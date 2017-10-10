import os
import uuid
import time
import cv2
import argparse
import numpy as np
from scipy import misc

import file_utils
import utils

TMP_DIR = "tmp/"
IMAGES_DIR = "tmp/images/"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

class ImageVisualizer:

    def __init__(self, project, config):
        self.project = project
        self.config = config

    def visualize(self, im):
        # Get files
        _ , im_path = file_utils.get_image(im, self.config)
        cm, cm_path = file_utils.get_category_mask(im, self.config)
        gt, gt_path = file_utils.get_ground_truth(im, self.config)
        gt_one_hot, gt_path = file_utils.get_ground_truth(im, self.config, one_hot=True)
        pm, pm_path = file_utils.get_prob_mask(im, self.config)
        ap, ap_path = file_utils.get_all_prob(im, self.config)

        paths = {}
        paths["image"] = im_path

        if cm is not None:
            cm_color, cm_color_path = self.add_color(cm)
            paths["category_mask"] = cm_color_path

        if pm is not None:
            paths["prob_mask"] = pm_path

        if gt is not None:
            gt_color, gt_color_path = self.add_color(gt)
            paths["ground_truth"] = gt_color_path

        if cm is not None and gt is not None:
            diff = self.make_diff(cm, gt)
            diff_color, diff_color_path = self.add_color(diff)
            paths["diff"] = diff_color_path

        #
        # Open slices
        #
        order = None
        if gt_one_hot is not None:
            slices = self.open_slices(gt_one_hot, num=20)
            slices_path = self.save(slices)
            paths["gt_slices"] = slices_path

        if ap is not None:
            slices = self.open_slices(ap, num=20, threshold=True)
            slices_path = self.save(slices)
            paths["ap_slices_thresholded"] = slices_path

        if ap is not None:
            slices = self.open_slices(ap, num=20)
            slices_path = self.save(slices)
            paths["ap_slices"] = slices_path

        return paths

    def make_diff(self, cm, gt):
        mask = gt - cm
        mask = np.invert(mask.astype(bool))
        diff = np.copy(gt)
        diff[mask] = 0
        return diff

    def open_slices(self, ap, num=10, threshold=False):
        if threshold:
            ap = ap > 0.5

        sums = [np.sum(slic) for slic in ap]
        order = np.flip(np.argsort(sums), 0)
        #order = range(100)

        slices = []
        for i in order[:num]:
            c = i+1
            s = ap[i,:,:]

            labeled = self.label_img(s, c)
            slices.append(labeled)
        output = np.concatenate(slices, axis=1)
        return output

    def get_slice(self, ap, c):
        s = ap[c-1]
        # t = transformer.transform(s)
        # output = np.concatenate([s,t], axis=1)*255
        return s

    def blend(self, gt, ap, c):
        gt = gt[c-1]
        pr = ap[c-1]
        blends = transformer.blend(gt, pr, num=8)
        blends = np.concatenate(blends, axis=1)
        return blends

    def label_img(self, img, c):
        if img.dtype == bool:
            img = img.astype('float32')
        if np.ndim(img) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        color = utils.to_color(c)
        img[:50,:200,:] = color
        tag = "{} {}".format(str(c), utils.categories[c])
        cv2.putText(img, tag, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        return img

    def add_color(self, img):
        if img is None:
            return None, None

        h,w = img.shape
        img_color = np.zeros((h,w,3))
        for i in xrange(1,151):
            img_color[img == i] = utils.to_color(i)
        path = self.save(img_color)
        return img_color, path

    def save(self, img):
        fname = "{}.jpg".format(uuid.uuid4().hex)
        path = os.path.join(IMAGES_DIR, fname)
        misc.imsave(path, img)
        return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', required=True, help="Project name")
    parser.add_argument('-i', '--image', help="Image name")
    args = parser.parse_args()

    im = args.image
    if not args.image:
        im_list = utils.open_im_list(args.project)
        im = im_list[0]

    print args.project, im
    config = utils.get_config(args.project)
    vis = ImageVisualizer(args.project, config)
    paths = vis.visualize(im)
    print paths
