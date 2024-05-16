import glob
from torch.utils.data import Dataset
from numpy.random import RandomState
import numpy as np
import cv2



class RainDataset(Dataset):
    def __init__(self, opt, is_eval=False, is_test=False):
        super(RainDataset, self).__init__()

        if is_test:
            self.dataset = opt.test_dataset
        elif is_eval:
            self.dataset = opt.eval_dataset
        else:
            self.dataset = opt.train_dataset
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
        self.rand_state = RandomState(66)   #是不是相当于seed
        self.patch_size = opt.patch_size
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]
        img_name_t = img_name.split('-')[-1]
        gt_name_t = gt_name.split('-')[-1]
        if img_name_t != gt_name_t:
            print('img_name',img_name)
            print('gt_name',gt_name)
            raise AssertionError('gt_name不匹配')

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)   #以原保存方式读入  （灰度还是彩色）
        img , gt  = self.crop(img, gt)  #crop
        if img.dtype == np.uint8:   #调用这里
            img = (img / 255.0).astype('float32')
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')

        return [img,gt]

    def crop(self, img_pair,gt):
        patch_size = self.patch_size
        h, w, c = img_pair.shape
        r = self.rand_state.randint(0, h - patch_size)
        c = self.rand_state.randint(0, w - patch_size)
        B = img_pair[r: r+patch_size, c: c+patch_size]
        GT = gt[r: r+patch_size, c: c+patch_size]

        return  B,GT
