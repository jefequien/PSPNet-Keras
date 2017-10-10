import os
import time
import random
import h5py
import numpy as np
from scipy import misc
import utils
import utils_image

NUM_CLASS = 150

class DataSource:
    def __init__(self, config, random=True):
        self.config = config

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

    # def random_im_with_category(self):
    #     idx = random.randint(0,len(self.im_list)-1)
    #     return self.im_list[idx]

    def get_image(self, im):
        image_dir = self.config["images"]
        img_path = os.path.join(image_dir, im)
        img = misc.imread(img_path)
        if img.ndim != 3:
            img = np.stack((img,img,img), axis=2)
        return utils_image.preprocess_image(img)

    def get_ground_truth(self, im):
        ground_truth_dir = self.config["ground_truth"]
        gt_path = os.path.join(ground_truth_dir, im.replace('.jpg', '.png'))
        gt = misc.imread(gt_path)
        gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
        return gt

    def get_prediction(self, im):
        all_probs_dir = os.path.join(self.config["predictions"], "all_prob")
        file_path = os.path.join(all_probs_dir, im.replace('.jpg', '.h5'))
        output = self.open_probs(file_path)
        output = np.transpose(output, (1,2,0))
        return output

    def open_probs(self, file_path):
        with h5py.File(file_path, 'r') as f:
                output = f['allprob'][:]
                if output.dtype == 'uint8':
                    return output.astype('float32')/255
                else:
                    return output.astype('float32')
