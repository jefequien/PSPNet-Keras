from __future__ import print_function
import os
import json
import colorsys
import numpy as np

PATH = os.path.dirname(__file__)


def apply_mask(image, mask):
    masked_image = np.copy(image)

    masked_image[mask] = np.maximum(masked_image[mask], random.choice(COLORS))

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, (0, 0, 0), 2)
    return masked_image

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

def get_categories():
    categories = {}
    with open(os.path.join(PATH, "objectInfo150.txt"), 'r') as f:
        for line in f.readlines():
            split = line.split()
            cat = split[0]
            if cat.isdigit():
                categories[int(cat)] = split[4].replace(',','')
        return categories
categories = get_categories()
category_list = [categories[i] for i in range(1, len(categories)+1)]

def open_im_list(im_list_txt):
    if ".txt" not in im_list_txt:
        project = im_list_txt
        CONFIG = get_config(project)
        im_list_txt = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(im_list_txt, 'r')]
    return np.array(im_list)

def get_config(project):
    with open(os.path.join(PATH, "../../../LabelMe/data_config.json"), 'r') as f:
        data_config = json.load(f)
        if project in data_config:
            return data_config[project]
        else:
            raise Exception("Project not found: " + project)

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

