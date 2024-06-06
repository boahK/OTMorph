import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks
from .loss import crossCorrelation3D, gradientLoss

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

class OTMorph(BaseModel):
    def name(self):
        return 'OTMorph'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = 128
        self.input_A = self.Tensor(nb, 1, size, size)
        self.input_B = self.Tensor(nb, 1, size, size)

        # load/define networks
        self.netG_R = networks.define_G(opt.input_nc, opt.encoder_nc, opt.decoder_nc, opt.which_model_net, opt.name, opt.init_type, self.gpu_ids)
        self.netOT_f, self.netOT_T = networks.define_NOT(opt.init_type, self.gpu_ids)

        if not self.isTrain:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_R, 'G_R', which_epoch)
            self.load_network(self.netOT_T, 'OT_T', which_epoch)
        if opt.continue_train:
            self.load_network(self.netOT_f, 'OT_f', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionRG = gradientLoss('l2')
            self.criterionCC = crossCorrelation3D(1, kernel=(9,9,9))
            self.criterionMSE = torch.nn.MSELoss() # mean reduction

            # initialize optimizers
            self.optimizer_GT = torch.optim.Adam(itertools.chain(self.netOT_T.parameters(), self.netG_R.parameters()), lr=opt.lr, weight_decay=1e-10)
            self.optimizer_f = torch.optim.Adam(self.netOT_f.parameters(), lr=opt.lr, weight_decay=1e-10)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_GT)
            self.optimizers.append(self.optimizer_f)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            real_B = Variable(self.input_B)
            optm_A = self.netOT_T(real_A)
            fake_B, flow_A = self.netG_R(torch.cat([optm_A, real_B, real_A], dim=1))
        self.flow_A = flow_A 
        self.fake_B = fake_B
        self.optm_A = optm_A

    def backward_G(self):
        lambda_R = self.opt.loss_lambda
        alpha = self.opt.loss_alpha

        # Neural optimal transport loss
        optm_A = self.netOT_T(self.real_A)
        OT_loss_T = self.criterionMSE(self.real_A, optm_A)
        OT_loss_f = self.netOT_f(optm_A).mean()
        OT_loss = OT_loss_T - OT_loss_f

        # Registration loss
        fake_B, flow_A = self.netG_R(torch.cat([optm_A, self.real_B, self.real_A], dim=1))
        lossA_RC = (self.criterionCC(fake_B, self.real_B))*alpha
        lossA_RL = (self.criterionRG(flow_A) * lambda_R)*alpha

        loss = OT_loss + lossA_RC + lossA_RL
        loss.backward()

        self.optm_A = optm_A.data
        self.flow_A  = flow_A.data
        self.fake_B = fake_B.data
        self.lossA_RC = lossA_RC.item()
        self.lossA_RL = lossA_RL.item()
        self.OT_loss_T = OT_loss_T.item()
        self.OT_loss_f = OT_loss_f
        self.loss_T = OT_loss.item()
        self.loss_f = 0
        self.loss = loss.item()

        del loss, optm_A, fake_B, flow_A; torch.cuda.empty_cache()


    def backward_f(self):
        # Neural optimal transport loss
        with torch.no_grad():
            optm_A = self.netOT_T(self.real_A)
        OT_loss = self.netOT_f(optm_A).mean() - self.netOT_f(self.real_B).mean()

        loss = OT_loss
        loss.backward()

        self.optm_A = optm_A.data
        self.loss_f = loss.item()
        self.loss_T = 0
        del loss, optm_A; torch.cuda.empty_cache()


    def optimize_parameters(self, istep, OTsteps=100):
        # forward
        self.forward()

        unfreeze(self.netOT_T); unfreeze(self.netG_R); freeze(self.netOT_f)
        self.optimizer_GT.zero_grad()
        self.backward_G()
        self.optimizer_GT.step()

        freeze(self.netOT_T); freeze(self.netG_R); unfreeze(self.netOT_f)
        if istep == OTsteps:
            self.optimizer_f.zero_grad()
            self.backward_f()
            self.optimizer_f.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('OT_T', self.loss_T), ('OT_f', self.loss_f),
                                  ('A_RC', self.lossA_RC), ('A_RL', self.lossA_RL),
                                  ('Tot', self.loss)])
        return ret_errors

    def get_current_visuals(self):
        realSize = self.input_A.shape

        real_A = util.tensor2im(self.input_A[0, 0, int(realSize[2]/2)])
        optm_A = util.tensor2im(self.optm_A[0, 0, int(realSize[2]/2)])
        flow_A = util.tensor2im(self.flow_A[0, :, int(realSize[2] / 2)])
        fake_B = util.tensor2im(self.fake_B[0, 0, int(realSize[2]/2)])
        real_B = util.tensor2im(self.input_B[0, 0, int(realSize[2]/2)])

        ret_visuals = OrderedDict([('real_A', real_A), ('flow_A', flow_A),
                                   ('fake_B', fake_B), ('optm_A', optm_A),
                                   ('real_B', real_B)])
        return ret_visuals

    def get_current_data(self):
        ret_visuals = OrderedDict([('flow_A', self.flow_A),('fake_B', self.fake_B),('optm_A', self.optm_A)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_R, 'G_R', label, self.gpu_ids)
        self.save_network(self.netOT_T, 'OT_T', label, self.gpu_ids)
        self.save_network(self.netOT_f, 'OT_f', label, self.gpu_ids)
