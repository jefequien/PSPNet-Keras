import random
import time
import numpy as np
from scipy import misc

import utils
from utils import image_utils
from utils.data import DataSource, threadsafe_generator

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
