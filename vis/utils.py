import os
import json
import cv2
import colorsys
import random

PATH = os.path.dirname(__file__)

def to_color(category):
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v,1,1)


def apply_mask(image, mask):
    masked_image = np.copy(image)

    masked_image[mask] = np.maximum(masked_image[mask], random.choice(COLORS))

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, (0, 0, 0), 2)
    return masked_image

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

def get_config(project):
    with open(os.path.join(PATH, "../../LabelMe/data_config.json"), 'r') as f:
        data_config = json.load(f)
        config = data_config[project]
        return config

# Can also be project
def open_im_list(txt_im_list):
    if ".txt" not in txt_im_list:
        project = txt_im_list
        CONFIG = get_config(project)
        txt_im_list = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(txt_im_list, 'r')]
    return im_list