import os, glob
import numpy as np
from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
from data.getDatabase import DataProvider
from math import *
from util import util

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.nThreads = 1
    model_regist = create_model(opt)
    util.mkdirs(opt.results_dir)

    data_test   = DataProvider(opt.dataroot, mode=opt.phase)
    dataset_size   = data_test.n_data
    test_iters = int(data_test.n_data)
    print('#Test images = %d' % dataset_size)

    startNum = 0
    testTime = np.zeros(((dataset_size), 1))
    numsub = -1
    for isub in range(test_iters):
        batch_x, batch_y, path = data_test(opt.batchSize)

        test_data = {'A': batch_x, 'B': batch_y}
        model_regist.set_input(test_data)
        model_regist.test()
        visuals = model_regist.get_current_data()
        regist_data = visuals['fake_B'].cpu().numpy()[0,0].transpose(1, 2, 0)
        
        dataName = path[0][0].split('/')[-1]
        image = nib.load(path[0][0])
        nib.save(nib.Nifti1Image(regist_data, image.affine), os.path.join(opt.results_dir, 'regist_'+dataName))

