import os
import time
import h5py
import numpy as np
from scipy import misc

NUM_CLASS = 150

class DataSource:

    def __init__(self, config):
        self.config = config
        self.print_config()

    def print_config(self):
        print "Config"
        for key in self.config:
            print "{}: {}".format(key, self.config[key])

    def get_image(self, im):
        try:
            path = self.get_file_path(im, ftype="im")
            img = open_file(path, ftype="im")
            return img, path
        except:
            print "No image", im
            return None, None

    def get_category_mask(self, im):
        try:
            path = self.get_file_path(im, ftype="cm")
            img = open_file(path, ftype="cm")
            return img, path
        except:
            print "No category mask", im
            return None, None

    def get_prob_mask(self, im):
        try:
            path = self.get_file_path(im, ftype="pm")
            img = open_file(path, ftype="pm")
            return img, path
        except:
            print "No prob mask", im
            return None, None

    def get_ground_truth(self, im, one_hot=False):
        try:
            path = self.get_file_path(im, ftype="gt")
            img = open_file(path, ftype="gt")
            if one_hot:
                NUM_CLASS = 150
                gt_one_hot = (np.arange(NUM_CLASS) == img[:,:,None] - 1)
                img = gt_one_hot.transpose((2,0,1))
            return img, path
        except:
            print "No ground truth", im
            return None, None

    def get_all_prob(self, im):
        try:
            path = self.get_file_path(im, ftype="ap")
            img = open_file(path, ftype="ap")
            return img, path
        except:
            print "No all prob", im
            return None, None

    def get_file_path(self, im, ftype="im"):
        if ftype == "im":
            root = self.config["images"]
            return os.path.join(root, im)
        elif ftype == "gt":
            root = self.config["ground_truth"]
            return os.path.join(root, im.replace(".jpg",".png"))
        elif ftype == "cm":
            root = os.path.join(self.config["pspnet_prediction"], "category_mask")
            return os.path.join(root, im.replace(".jpg",".png"))
        elif ftype == "ap":
            root = os.path.join(self.config["pspnet_prediction"], "all_prob")
            fname = os.path.join(root, im.replace(".jpg", ".h5"))
            return fname
        elif ftype == "pm":
            root = os.path.join(self.config["pspnet_prediction"], "prob_mask")
            fname = os.path.join(root, im)
            return fname
        else:
            print "File type not found."
            raise Exception

def open_file(file_path, ftype="im"):
        if ftype == "im":
            return misc.imread(file_path)
        elif ftype == "gt":
            return misc.imread(file_path)
        elif ftype == "cm":
            return misc.imread(file_path)
        elif ftype == "ap":
            with h5py.File(file_path, 'r') as f:
                output = f['allprob'][:]
                if output.dtype == 'uint8':
                    return output.astype('float32')/255
                else:
                    return output.astype('float32')
        elif ftype == "pm":
            return misc.imread(file_path)
        else:
            print "File type not found."
            raise Exception
