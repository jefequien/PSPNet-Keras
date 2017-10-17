import random
import numpy as np
from scipy import misc

from data_generator import threadsafe_generator
from disc import prepare_disc_data

from utils import image_utils
from utils.datasource import DataSource

@threadsafe_generator
def DiscDataGenerator(im_list, datasource, category):
    while True:
        im1 = random.choice(im_list)
        im2 = random.choice(im_list)
        img1, _ = datasource.get_image(im1)
        img2, _ = datasource.get_image(im2)
        img1 = image_utils.preprocess_image(img1)
        img2 = image_utils.preprocess_image(img2)
        gt1, _ = datasource.get_ground_truth(im1, one_hot=True)
        gt2, _ = datasource.get_ground_truth(im2, one_hot=True)
        ap1, _ = datasource.get_all_prob(im1)
        ap2, _ = datasource.get_all_prob(im2)

        # Training data
        img1 = image_utils.scale(img1, (473,473))
        img1 = image_utils.preprocess_image(img1)
        img2 = image_utils.scale(img2, (473,473))
        img2 = image_utils.preprocess_image(img2)

        d1 = prepare_disc_data(img1, gt1, category)
        d2 = prepare_disc_data(img1, ap1, category)
        d3 = prepare_disc_data(img2, gt2, category)
        d4 = prepare_disc_data(img2, ap2, category)
        s1 = d1[:,:,3] > 0
        s2 = d2[:,:,3] > 0
        s3 = d3[:,:,3] > 0
        s4 = d4[:,:,3] > 0
        l1 = np.max(s1)
        l2 = iou(s1, s2)
        l3 = np.max(s3)
        l4 = iou(s4, s4)
        data = [d1, d2, d3, d4]
        label = [l1, l2, l3, l4]

        # Augment with mismatched data
        b1 = prepare_disc_data(img1, gt2, category)
        b2 = prepare_disc_data(img1, ap2, category)
        b3 = prepare_disc_data(img2, gt1, category)
        b4 = prepare_disc_data(img2, ap1, category)
        bad_data = [b1,b2,b3,b4]
        bad_label = [0, 0, 0, 0]

        data = data + bad_data
        label = label + bad_label
        data = np.stack(data, axis=0)
        label = np.array(label)
        
        # save(data)
        yield (data, label)

def iou(gt_s, pr_s):
    gt_s = gt_s > 0
    pr_s = pr_s > 0
    intersection = np.logical_and(gt_s, pr_s)
    union = np.logical_or(gt_s, pr_s)
    if np.sum(union) == 0:
        return 0
    else:
        iou = 1.0*np.sum(intersection)/np.sum(union)
        return iou

def save(data):
    import uuid, h5py
    fname = "vis/tmp/{}.h5".format(uuid.uuid4().hex)
    with h5py.File(fname, 'w') as f:
        f.create_dataset('data', data=data)