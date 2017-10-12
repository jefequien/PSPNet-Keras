import random
import time
import threading
import numpy as np
from scipy import misc

import utils
from utils import image_utils
from utils.datasource import DataSource

NUM_CLASS = 150

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

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
        g1 = prep_disc_data(img1, gt1, category)
        b1 = prep_disc_data(img1, ap1, category)
        b2 = prep_disc_data(img1, gt2, category)
        b3 = prep_disc_data(img1, ap2, category)
        data = [g1, b1, b2, b3]
        label = [1, 0, 0, 0]

        data = np.concatenate(data, axis=0)
        label = label
        yield (data, label)

def prep_disc_data(img, pr, category):
    s = pr[category-1]
    s = s > 0.5
    data = np.concatenate((img, s[:,:,np.newaxis]), axis=2)
    return data

@threadsafe_generator
def DataGenerator(im_list, datasource, maxside=None):
    while True:
        im = random.choice(im_list)
        img, _ = datasource.get_image(im)
        gt, _ = datasource.get_ground_truth(im, one_hot=True)
        img = image_utils.preprocess_image(img)
        gt = gt.transpose((1,2,0)) # Make channel-last

        if maxside is None:
            img_s = image_utils.scale(img, (473,473))
            gt_s = image_utils.scale(gt, (473,473))
        else:
            img_s = image_utils.scale_maxside(img, maxside=maxside)
            gt_s = image_utils.scale_maxside(gt, maxside=maxside)
        
        box = image_utils.random_crop(img_s)
        data = image_utils.crop_array(img_s, box)
        label = image_utils.crop_array(gt_s, box)

        # Batch size of 1
        data = data[np.newaxis, ...]
        label = label[np.newaxis, ...]
        yield (data, label)


if __name__ == "__main__":
    project = "local"
    config = utils.get_config(project)
    im_list = utils.open_im_list(config["im_list"])
    datasource = DataSource(config)
    generator = DataGenerator(im_list, datasource, maxside=512)

    data, label = generator.next()
    print data.shape
    print label.shape

    data = data[0]
    label = label[0]

    unlabeled = np.max(label, axis=2) == 0
    gt = np.argmax(label, axis=2) + 1
    gt[unlabeled] = 0
    gt = utils.add_color(gt)

    misc.imsave("data.png", data)
    misc.imsave("label.png", gt)
