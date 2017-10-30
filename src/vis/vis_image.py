import os
import uuid
import time
import cv2
import argparse
import numpy as np
from scipy import misc
from collections import OrderedDict

import utils
from utils.datasource import DataSource

TMP_DIR = "tmp/"
IMAGES_DIR = "tmp/images/"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

class ImageVisualizer:

    def __init__(self, project, datasource):
        self.project = project
        self.datasource = datasource

    def visualize(self, im):
        # Get files
        img, im_path = self.datasource.get_image(im)
        cm, cm_path = self.datasource.get_category_mask(im)
        gt, gt_path = self.datasource.get_ground_truth(im)
        pm, pm_path = self.datasource.get_prob_mask(im)
        paths = OrderedDict()

        if im is not None:
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

        return paths

    def visualize_all_categories(self, im):
        gt, gt_path = self.datasource.get_ground_truth(im, one_hot=True)
        ap, ap_path = self.datasource.get_all_prob(im)
        paths = OrderedDict()

        order = self.get_order(gt, ap)
        paths["order"] = order
        if gt is not None:
            slices = self.open_slices(gt, order, num=20)
            slices_path = self.save(slices)
            paths["gt_slices"] = slices_path

        if ap is not None:
            slices = self.open_slices(ap, order, num=20, threshold=True)
            slices_path = self.save(slices)
            paths["ap_slices_thresholded"] = slices_path

        if ap is not None:
            slices = self.open_slices(ap, order, num=20)
            slices_path = self.save(slices)
            paths["ap_slices"] = slices_path
        return paths

    def visualize_category(self, im, category):
        gt, gt_path = self.datasource.get_ground_truth(im, one_hot=True)
        ap, ap_path = self.datasource.get_all_prob(im)
        paths = OrderedDict()
        
        if gt is not None:
            s = gt[category-1]
            s_path = self.save(s)
            paths["gt_slice"] = s_path
        if ap is not None:
            s = ap[category-1]
            s = s > 0.5
            s_path = self.save(s)
            paths["ap_slice_thresholded"] = s_path
        if ap is not None:
            s = ap[category-1]
            s_path = self.save(s)
            paths["ap_slice"] = s_path
        return paths

    def make_diff(self, cm, gt):
        mask = gt - cm
        mask = np.invert(mask.astype(bool))
        diff = np.copy(gt)
        diff[mask] = 0
        return diff

    def open_slices(self, ap, order, num=10, threshold=False):
        if threshold:
            ap = ap > 0.5

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

    def get_order(self, gt, ap):
        if gt is None and ap is None:
            return np.arange(0,150)

        gt_sums = np.zeros(150)
        ap_sums = np.zeros(150)
        if gt is not None:
            gt_sums = np.array([np.sum(s) for s in gt])
        if ap is not None:
            ap_sums = np.array([np.sum(s) for s in ap])

        sums = gt_sums + ap_sums
        order = np.flip(np.argsort(sums), 0)
        return order

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
    datasource = DataSource(config)
    vis = ImageVisualizer(args.project, datasource)
    paths = vis.visualize(im)
    print paths

