from __future__ import print_function
import os
import json
import colorsys
import numpy as np

PATH = os.path.dirname(__file__)

def add_color(img):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in xrange(1, 151):
        img_color[img == i] = to_color(i)
    return img_color


def to_color(category):
    # Maps each category a good distance away
    # from each other on the HSV color space
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v, 1, 1)

def open_im_list(im_list_txt):
    if ".txt" not in im_list_txt:
        project = im_list_txt
        CONFIG = get_config(project)
        im_list_txt = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(im_list_txt, 'r')]
    return im_list

def get_config(project):
    with open(os.path.join(PATH, "../LabelMe/data_config.json"), 'r') as f:
        data_config = json.load(f)
        config = data_config[project]
        return config

def get_latest_checkpoint(checkpoint_dir):
    # weights.00-1.52.hdf5
    latest_i = -1
    latest_fn = ""
    for fn in os.listdir(checkpoint_dir):
        split0 = fn.split('-')[0]
        i = int(split0.split('.')[1])
        if i > latest_i:
            latest_i = i
            latest_fn = fn

    if latest_i == -1:
        raise Exception("No checkpoint found.")
    return os.path.join(checkpoint_dir, latest_fn), latest_i+1

