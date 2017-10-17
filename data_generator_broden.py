import re
import os
import numpy
from scipy.misc import imresize
from PIL import Image
import cv2
import loadseg
import random

from data_generator import threadsafe_generator

TEST_DIR = '/data/vision/torralba/deepscene/david_data/bulk/uniseg4_384'

params = {}
params['split'] = 'train'
params['mean'] = (109.5388, 118.6897, 124.6901)

class DataLayer:
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def __init__(self):
        """
        Setup data layer according to parameters:
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        """
        # config
        # params = eval(self.param_str)
        self.directory = TEST_DIR     # Really should be from param_str
        self.split = params['split']  # I have not implemented splits yet
        self.mean = numpy.array(params['mean'])
        self.random = params.get('randomize', True)
        self.random_flip = True #params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params.get('batch_size', 1)
        self.disp = 0
        self.categories = ['object','part','texture','material','color']
        # self.categories = ['object']
        self.categories_num_class = [584,234,47,32,11]
        self.segmentation_shape = params.get('segmentation_shape', None)
        self.splitmap = {
                'train': 1,
                'val': 2
                }

        # Convert to 2-dimensional shape.
        if self.segmentation_shape and len(
                numpy.shape(self.segmentation_shape)) == 0:
            self.segmentation_shape = (self.segmentation_shape,) * 2

        # Specific object classes to ignore.
        self.blacklist = {
                #'object': [1,2] # wall, floor, ceiling, sky: in uniseg: 4 become tree!!
            }

        # Thresholds to ignore: these classes and any ones rarer (higher).
        self.outliers = {
                'object': 537,    # brick occurs only 9 times in test_384.
                #'part': 155,      # porch occurs only 9 times in test_384

                                  # if switching to uniseg, switch 561->544.
                                  # because there are fewer object classes.
                                  # part classes remain the same.
            }

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False
            self.random_flip = False

        # Load up metadata for images and labels
        self.datasource = loadseg.SegmentationData(self.directory,
                categories=self.categories)
        self.prefetcher = loadseg.SegmentationPrefetcher(self.datasource,
                split=self.split, categories=['image'] + self.categories,
                segmentation_shape=self.segmentation_shape,
                batch_size=self.batch_size, randomize=self.random) # ahead=12)

        # Now make a blacklist map for blacklisted types
        self.zeromap = {}
        for k, z in self.blacklist.items():
            self.zeromap[k] = numpy.arange(self.datasource.label_size(k))
            self.zeromap[k][z] = 0
        for k, z in self.outliers.items():
            if k not in self.zeromap:
                self.zeromap[k] = numpy.arange(self.datasource.label_size(k))
            self.zeromap[k][numpy.arange(z, self.datasource.label_size(k))] = 0

        # Build our category map which merges the category map and the zeromap
        self.categorymap = {}
        for cat in self.categories:
            catmap = self.datasource.category_index_map(cat)
            if cat in self.zeromap:
                catmap = self.zeromap[cat][catmap]
            self.categorymap[cat] = catmap

    @threadsafe_generator
    def generator(self):
        yield (data, label)

    def reshape(self):
        # load image + label image set
        batch = self.prefetcher.fetch_batch()
        full_data = self.form_tensors(batch)
        
        self.image = full_data[0]
        
        ls = []
        for i in xrange(len(self.categories)):
            data = full_data[i+1]
            NUM_CLASS = self.categories_num_class[i]
            print NUM_CLASS, data.shape
            
            l = self.one_hot_encode(data - 1,NUM_CLASS)
            ls.append(l)
        
        self.label = numpy.concatenate(ls, axis=1)

        return self.image, self.label

    def one_hot_encode(self, a, NUM_CLASS):
        '''
        a.shape = n,1,h,w
        '''
        a = a[:,0,:,:]
        label = (numpy.arange(NUM_CLASS) == a[...,None])
        return label.transpose((0,3,1,2))

    def form_tensors(self, batch):
        # Assemble a batch in [{'cat': data,..},..] format into
        # an array of batch tensors, the first for the image, and the
        # remaining for each category in self.categories, in order.
        # This also applies a random flip if needed
        batches = [[] for _ in range(len(self.categories) + 1)]
        for record in batch:
            mirror = (random.randrange(2) * 2 - 1) if self.random_flip else 1
            flip = (slice(None), slice(None, None, mirror), slice(None))
            default_shape = (1, record['sh'], record['sw'])
            for c, cat in enumerate(['image'] + self.categories):
                if cat == 'image':
                    # Normalize image with right RGB order and mean
                    batches[c].append(self.normalize_image(record[cat])[flip])
                else:
                    batches[c].append(self.normalize_label(
                        cat, record[cat], default_shape)[flip])
        return [numpy.stack(b) for b in batches]

    def normalize_image(self, rgb_image):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = numpy.array(rgb_image, dtype=numpy.float32)
        if (in_.ndim == 2):
            in_ = numpy.repeat(in_[:,:,None], 3, axis = 2)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = imresize(in_, (473,473), interp='bilinear')
        in_ = in_.transpose((2,0,1))
        return in_

    def normalize_label(self, cat, label_data, shape):
        dims = len(numpy.shape(label_data))
        catmap = self.categorymap[cat]
        label = None
        if dims <= 2:
            # Scalar data on this channel: fill shape
            if dims == 1:
                label_data = label_data[0] if len(label_data) else 0
            mapped = catmap[label_data]
            label = numpy.full(shape, mapped, dtype=numpy.int16)
        else:
            if dims == 3:
                label_data = label_data[0]
            mapped = catmap[label_data]
            label = mapped[numpy.newaxis]
        label = cv2.resize(label.transpose((1,2,0)), (473,473), interpolation=cv2.INTER_NEAREST)
        return label[numpy.newaxis,:,:]

if __name__ == "__main__":
    generator = DataLayer().generator()

    data, label = generator.next()
    print data.shape
    print label.shape
    

