import os
import time
import random
import h5py
import numpy as np
from scipy import misc
import utils

DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]]) # RGB
NUM_CLASS = 150

class DataSource:
    def __init__(self, config, random=True):
        self.image_dir = config["images"]
        self.ground_truth_dir = config["ground_truth"]

        self.im_list = utils.open_im_list(config["im_list"])
        
        self.random = random
        if not self.random:
            self.idx = -1

    def next_im(self):
        if self.random:
            idx = random.randint(0,len(self.im_list)-1)
            return self.im_list[idx]
        else:
            self.idx += 1
            if self.idx == len(self.im_list):
                raise Exception("Reached end of image list.")
            return self.im_list[self.idx]

    def get_image(self, im):
        img_path = os.path.join(self.image_dir, im)
        img = misc.imread(img_path)
        if img.ndim != 3:
            img = np.stack((img,img,img), axis=2)

        # Preprocess
        img = img - DATA_MEAN
        # RGB => BGR
        img = img[:,:,::-1]
        img = img.astype('float32')
        return img

    def get_ground_truth(self, im):
        gt_path = os.path.join(self.ground_truth_dir, im.replace('.jpg', '.png'))
        gt = misc.imread(gt_path)
        gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
        return gt



