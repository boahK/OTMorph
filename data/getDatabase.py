from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import os
from .getDataLoader import BaseDataProvider
import nibabel as nib
import glob
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.measure import label  

def largest_connected_component(segmentation):
    labels = label(segmentation)
    largest_cc = labels == np.argmax(np.bincount(labels[segmentation]))
    return largest_cc

class DataProvider(BaseDataProvider):
    def __init__(self, path, mode=None):
        super(DataProvider, self).__init__()
        self.path       = path
        self.mode       = mode
        self.data_idx   = -1
        self.n_data     = self._load_data()

    def _load_data(self):
        self.imageNum = []

        dataList = sorted(glob.glob(os.path.join(self.path, 'CT', '*0000.nii.gz')))

        if self.mode == 'train':
            for dataName_ct in dataList[:1]:
                subNum = int(dataName_ct.split('/')[-1].split('_')[1])
                dataName_mr = os.path.join(self.path, 'MR', 'subject_%d_0000.nii.gz'%subNum)
                self.imageNum.append([dataName_ct, dataName_mr])
                np.random.shuffle(self.imageNum)

        else:
            for dataName_ct in dataList[1:]:
                subNum = int(dataName_ct.split('/')[-1].split('_')[1])
                dataName_mr = os.path.join(self.path, 'MR', 'subject_%d_0000.nii.gz'%subNum)
                self.imageNum.append([dataName_ct, dataName_mr])

        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode =="train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        dataPath = self.imageNum[self.data_idx]

        data_mov = nib.load(dataPath[0]).get_fdata()
        data_fix = nib.load(dataPath[1]).get_fdata()
        
        data_mov = np.clip(data_mov, -300, 300)
        data_mov[data_mov==0] += -300 
        data_mov_mask = data_mov > data_mov.min() 
        data_mov_mask = largest_connected_component(data_mov_mask)
        data_mov = data_mov * data_mov_mask
        data_mov[data_mov==0] += data_mov.min()
        data_mov -= data_mov.min()
        data_mov /= data_mov.max()

        data_fix = rescale_intensity(data_fix, in_range=tuple(np.percentile(data_fix, (1, 99)))) #1, 99
        data_fix -= data_fix.min()
        data_fix /= data_fix.max()

        nh, nw, nd = data_mov.shape
        
        if nd < 48:
            data_mov_ = np.zeros([nh, nw, 48])
            data_fix_ = np.zeros([nh, nw, 48])
            data_mov_[:, :, :nd] = data_mov
            data_fix_[:, :, :nd] = data_fix
            data_mov, data_fix = data_mov_, data_fix_
        nd_ = 48 
        data_mov = resize(data_mov, (320, 320, nd_))
        data_fix = resize(data_fix, (320, 320, nd_))

        return data_mov, data_fix, dataPath

    def _augment_data(self, data, label):
        if self.mode == "train":
            # Flip horizon / vertical
            op = np.random.randint(0, 3)
            if op < 2:
                data, label = np.flip(data, op), np.flip(label, op)

        return data, label


