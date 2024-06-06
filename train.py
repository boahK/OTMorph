import time
from options.train_options import TrainOptions
from data.getDatabase import DataProvider
from models.models import create_model
from util.visualizer import Visualizer
from math import *
from util import util
import os

opt = TrainOptions().parse()
data_train     = DataProvider(opt.dataroot, mode="train")
dataset_size   = data_train.n_data
training_iters = int(ceil(data_train.n_data/float(opt.batchSize)))
print('#training images = %d' % dataset_size)

total_steps = 0
model = create_model(opt)
visualizer = Visualizer(opt)
check_dir = opt.checkpoints_dir
img_dir = os.path.join(check_dir, 'images')
util.mkdirs([check_dir, img_dir])

ot_iter = 1
for epoch in range(opt.epoch_count, opt.niter + 1):
    epoch_start_time = time.time()

    """ Train """
    for step in range(1, training_iters+1):
        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'B': batch_y, 'path': path}
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += 1
        model.set_input(data)

        model.optimize_parameters(ot_iter, OTsteps=opt.OTsteps)
        ot_iter += 1
        if ot_iter > opt.OTsteps:
            ot_iter = 1

        if step % opt.display_step == 0:
            for label, image_numpy in model.get_current_visuals().items():
                img_path = os.path.join(img_dir, '%s.png' % (label))
                util.save_image(image_numpy, img_path)
            
        if step % opt.plot_step == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, step, training_iters, errors, t, 'Train')

    if epoch % opt.save_epoch_img == 0:
        visualizer.save_current_results(model.get_current_visuals(), epoch, True)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))
