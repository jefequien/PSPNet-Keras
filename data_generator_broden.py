import random
import numpy as np

import loadseg
from data_generator import threadsafe_generator

TEST_DIR = '/data/vision/torralba/deepscene/david_data/bulk/uniseg4_384'

@threadsafe_generator
def BrodenDataGenerator():
    categories = ['object','part','texture','material','color']
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

if __name__ == "__main__":
    generator = BrodenDataGenerator()

    data, label = generator.next()
    print data.shape
    print label.shape
    


