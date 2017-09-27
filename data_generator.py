import random
import time
import threading
import numpy as np
from scipy import misc

from image_processor import *
import utils
from datasource import DataSource

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
def DataGenerator(datasource, maxside=None):
    while True:
        im = datasource.next_im()
        img = datasource.get_image(im)
        img = datasource.preprocess_image(img)
        gt = datasource.get_ground_truth(im)

        if maxside is None:
            img_s = scale(img, (473,473))
            gt_s = scale(gt, (473,473))
        else:
            img_s = scale_maxside(img, maxside=maxside)
            gt_s = scale_maxside(gt, maxside=maxside)
        
        box = random_crop(img_s)
        data = crop_array(img_s, box)
        label = crop_array(gt_s, box)

        # Batch size of 1
        data = data[np.newaxis, ...]
        label = label[np.newaxis, ...]
        yield (data, label)


if __name__ == "__main__":
    project = "local"
    config = utils.get_config(project)
    datasource = DataSource(config, random=True)
    generator = DataGenerator(datasource, maxside=512)

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
