import random
import numpy as np

import loadseg
from data_generator import threadsafe_generator

TEST_DIR = '/data/vision/torralba/deepscene/david_data/bulk/uniseg4_384'
CATEGORIES = ['object','part','texture','material','color']

@threadsafe_generator
def BrodenDataGenerator():
    split = "train"
    batch_size = 8
    randomize = True

    while True:
        datasource = loadseg.SegmentationData(TEST_DIR,
                categories=categories)
        prefetcher = loadseg.SegmentationPrefetcher(datasource,
                split=split, categories=['image'] + categories,
                segmentation_shape=None,
                batch_size=batch_size, randomize=randomize, ahead=12)
        batch = prefetcher.fetch_batch()
        yield batch, None

def form_tensors(batch):
    # Assemble a batch in [{'cat': data,..},..] format into
    # an array of batch tensors, the first for the image, and the
    # remaining for each category in self.categories, in order.
    # This also applies a random flip if needed
    batches = [[] for _ in range(len(categories) + 1)]
    for record in batch:
        mirror = (random.randrange(2) * 2 - 1) if self.random_flip else 1
        flip = (slice(None), slice(None, None, mirror), slice(None))
        default_shape = (1, record['sh'], record['sw'])
        for c, cat in enumerate(['image'] + CATEGORIES):
            if cat == 'image':
                # Normalize image with right RGB order and mean
                batches[c].append(normalize_image(record[cat])[flip])
            else:
                batches[c].append(normalize_label(
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
    generator = BrodenDataGenerator()

    data, label = generator.next()
    print data.shape
    print label.shape
    


