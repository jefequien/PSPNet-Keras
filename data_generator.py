import random
import time
import threading
import numpy as np
from scipy import misc

import image_processor
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
def DataGenerator(datasource):
    while True:
        # t = time.time()
        im = datasource.next_im()
        img = datasource.get_image(im)
        gt = datasource.get_ground_truth(im)
        outs = image_processor.scale_and_crop_imgs([img, gt])

        data = outs[0]
        label = outs[1]

        # Batch size of 1
        data = data[np.newaxis, ...]
        label = label[np.newaxis, ...]

        #print time.time() - t
        yield (data,label)


if __name__ == "__main__":

    project = "local"
    config = utils.get_config(project)
    datasource = DataSource(config, random=True)
    generator = DataGenerator(datasource)

    data, label = generator.next()
    print data.shape
    print label.shape

    data = data[0]
    label = np.argmax(label[0], axis=2)
    label = utils.add_color(label)

    misc.imsave("data.png", data)
    misc.imsave("label.png", label)
