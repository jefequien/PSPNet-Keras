import os
import h5py
from scipy import misc
import numpy as np

#
# Open files
#
def get_image(im, config):
    try:
        path = get_file_path(im, config, ftype="im")
        img = open_file(path, ftype="im")
        return img, path
    except:
        print "No image", im
        return None, None

def get_category_mask(im, config):
    try:
        path = get_file_path(im, config, ftype="cm")
        img = open_file(path, ftype="cm")
        return img, path
    except:
        print "No category mask", im
        return None, None

def get_prob_mask(im, config):
    try:
        path = get_file_path(im, config, ftype="pm")
        img = open_file(path, ftype="pm")
        return img, path
    except:
        print "No prob mask", im
        return None, None

def get_ground_truth(im, config):
    try:
        path = get_file_path(im, config, ftype="gt")
        img = open_file(path, ftype="gt")
        return img, path
    except:
        print "No ground truth", im
        return None, None

def get_all_prob(im, config):
    try:
        path = get_file_path(im, config, ftype="ap")
        img = open_file(path, ftype="ap")
        return img, path
    except:
        print "No all prob", im
        return None, None

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

def get_file_path(im, config, ftype="im"):
    if ftype == "im":
        root = config["images"]
        return os.path.join(root, im)
    elif ftype == "gt":
        root = config["ground_truth"]
        return os.path.join(root, im.replace(".jpg",".png"))
    elif ftype == "cm":
        root = os.path.join(config["pspnet_prediction"], "category_mask")
        return os.path.join(root, im.replace(".jpg",".png"))
    elif ftype == "ap":
        root = os.path.join(config["pspnet_prediction"], "all_prob")
        fname = os.path.join(root, im.replace(".jpg", ".h5"))
        return fname
    elif ftype == "pm":
        root = os.path.join(config["pspnet_prediction"], "prob_mask")
        fname = os.path.join(root, im)
        return fname
    else:
        print "File type not found."
        raise Exception


if __name__=="__main__":
    image_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/images/ADE_train_00000037.jpg"
    mask_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/annotations/ADE_train_00000037.png"
    category = 1
    image = misc.imread(image_path)
    mask = misc.imread(mask_path)

    category_mask = (mask == category)
    masked_image = apply_mask(image, category_mask)
