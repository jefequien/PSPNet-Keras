from functools import partial
import numpy
import os
import re
import random
import signal
from scipy.misc import imread
from unicsv import DictUnicodeReader
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom

class AbstractSegmentation:
    def all_names(self, category, j):
        raise NotImplementedError
    def size(self):
        return 0
    def filename(self, i):
        raise NotImplementedError
    def metadata(self, i):
        return self.filename(i)
    @classmethod
    def resolve_segmentation(cls, m):
        return {}

    def name(self, category, i):
        '''
        Default implemtnation for segmentation_data,
        utilizing all_names.
        '''
        all_names = self.all_names(category, i)
        return all_names[0] if len(all_names) else ''

    def segmentation_data(self, category, i, c=0, full=False):
        '''
        Default implemtnation for segmentation_data,
        utilizing metadata and resolve_segmentation.
        '''
        segs = self.resolve_segmentation(
                self.metadata(i), categories=[category])
        if category not in segs:
            return 0
        data = segs[category]
        if not full and len(data.shape) >= 3:
            return data[0]
        return data

class SegmentationData(AbstractSegmentation):
    '''
    Represents and loads a multi-channel segmentation represented with
    a series of csv files: index.csv lists the images together with
    any label data avilable in each category; category.csv lists
    the categories of segmentations available; and label.csv lists the
    numbers used to describe each label class. In addition, the categories
    each have a separate c_*.csv file describing a dense coding of labels.
    '''
    def __init__(self, directory, categories=None, require_all=False):
        directory = os.path.expanduser(directory)
        self.directory = directory
        with open(os.path.join(directory, 'index.csv')) as f:
            self.image = [decode_index_dict(r) for r in DictUnicodeReader(f)]
        with open(os.path.join(directory, 'category.csv')) as f:
            self.category = dict(
                    (row['name'], row) for row in DictUnicodeReader(f))
        with open(os.path.join(directory, 'label.csv')) as f:
            label_data = [decode_label_dict(r) for r in DictUnicodeReader(f)]
        self.label = build_dense_label_array(label_data)
        if categories is not None:
            # Filter out unused categories
            categories = set([c for c in categories if c in self.category])
            for cat in self.category.keys():
                if cat not in categories:
                    del self.category[cat]
        else:
            categories = self.category.keys()
        # Filter out images with insufficient data
        filter_fn = partial(
                index_has_all_data if require_all else index_has_any_data,
                categories=categories)
        self.image = [row for row in self.image if filter_fn(row)]
        # Build dense remapping arrays for labels, so that you can
        # get dense ranges of labels for each category.
        self.category_label = {}
        self.category_map = {}
        self.category_unmap = {}
        for cat in self.category:
            with open(os.path.join(directory, 'c_%s.csv' % cat)) as f:
                c_data = [decode_label_dict(r) for r in DictUnicodeReader(f)]
            self.category_unmap[cat], self.category_map[cat] = (
                    build_numpy_category_map(c_data))

    def all_names(self, category, j):
        '''All English synonyms for the given label'''
        if category is not None:
            j = self.category_unmap[category][j]
        return [self.label[j]['name']] + self.label[i]['syns']

    def size(self):
        '''The number of images in this data set.'''
        return len(self.image)

    def filename(self, i):
        '''The filename of the ith jpeg (original image).'''
        return os.path.join(self.directory, 'images', self.image[i]['image'])

    def split(self, i):
        '''Which split contains item i.'''
        return self.image[i]['split']

    def metadata(self, i):
        '''Extract metadata for image i, For efficient data loading.'''
        return self.directory, self.image[i]

    @classmethod
    def resolve_segmentation(cls, m, categories=None):
        '''
        Resolves a full segmentation, potentially in a differenct process,
        for efficient multiprocess data loading.
        '''
        directory, row = m
        result = {}
        for cat, d in row.items():
            if cat in ['image', 'split', 'ih', 'iw', 'sh', 'sw']:
                continue
            if not wants(cat, categories):
                continue
            if all(isinstance(data, int) for data in d):
                result[cat] = d
                continue
            out = numpy.empty((len(d), row['sh'], row['sw']), dtype=numpy.int16)
            for i, channel in enumerate(d):
                if isinstance(channel, int):
                    out[i] = channel
                else:
                    rgb = imread(os.path.join(directory, 'images', channel))
                    out[i] = rgb[:,:,0] + rgb[:,:,1] * 256
            result[cat] = out
        return result, (row['sh'], row['sw'])

    def label_size(self, category=None):
        '''
        Returns the number of distinct labels (plus zero), i.e., one
        more than the maximum label number.  If a category is specified,
        returns the number of distinct labels within that category.
        '''
        if category is None:
            return len(self.label)
        else:
            return len(self.category_unmap[category])

    def name(self, category, j):
        '''
        Returns an English name for the jth label.  If a category is
        specified, returns the name for the category-specific nubmer j.
        If category=None, then treats j as a fully unified index number.
        '''
        if category is not None:
            j = self.category_unmap[category][j]
        return self.label[j]['name']

    def segmentation_data(self, category, i, c=0, full=False, out=None):
        '''
        Returns a 2-d numpy matrix with segmentation data for the ith image,
        restricted to the given category.  By default, maps all label numbers
        to the category-specific dense mapping described in the c_*.csv
        listing; but can be asked to expose the fully unique indexing by
        using full=True.
        '''
        row = self.image[i]
        data_channels = row.get(category, ())
        if c >= len(data_channels):
            channel = 0  # Deal with unlabeled data in this category
        else:
            channel = data_channels[c]
        if out is None:
            out = numpy.empty((row['sh'], row['sw']), dtype=numpy.int16)
        if isinstance(channel, int):
            if not full:
                channel = self.category_map[category][channel]
            out[:,:] = channel  # Single-label for the whole image
            return out
        png = imread(os.path.join(self.directory, 'images', channel))
        if full:
            # Full case: just combine png channels.
            out[...] = png[:,:,0] + png[:,:,1] * 256
        else:
            # Dense case: combine png channels and apply the category map.
            catmap = self.category_map[category]
            out[...] = catmap[png[:,:,0] + png[:,:,1] * 256]
        return out

    def full_segmentation_data(self, i, shape,
            categories=None, max_depth=None, out=None):
        '''
        Returns a 3-d numpy tensor with segmentation data for the ith image,
        with multiple layers represnting multiple lables for each pixel.
        The depth is variable depending on available data but can be
        limited to max_depth.
        '''
        row = self.image[i]
        if categories:
            groups = [d for cat, d in row if cat in categories and d]
        else:
            groups = [d for cat, d in row if d and (
                cat not in ['image', 'split', 'ih', 'iw', 'sh', 'sw'])]
        depth = sum(len(c) for c in groups)
        if max_depth is not None:
            depth = min(depth, max_depth)
        # Allocate an array if not already allocated.
        if out is None:
            out = numpy.empty((depth, row['sh'], row['sw']), dtype=numpy.int16)
        i = 0
        # Stack up the result segmentation one channel at a time
        for group in groups:
            for channel in group:
                if isinstance(channel, int):
                    out[i] = channel
                else:
                    rgb = imread(channel)
                    out[i] = rgb[0] + rgb[1] * 256
                i += 1
                if i == depth:
                    return out
        # Return above when we get up to depth
        assert False

    def category_index_map(self, category):
        return numpy.array(self.category_map[category])

def build_dense_label_array(label_data, key='number', allow_none=False):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    result = [None] * (max([d[key] for d in label_data]) + 1)
    for d in label_data:
        result[d[key]] = d
    # Fill in none
    if not allow_none:
        example = label_data[0]
        def make_empty(k):
            return dict((c, k if c is key else type(v)())
                    for c, v in example.items())
        for i, d in enumerate(result):
            if d is None:
                result[i] = dict(make_empty(i))
    return result

def build_numpy_category_map(map_data, key1='code', key2='number'):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    results = list(numpy.zeros((max([d[key] for d in map_data]) + 1),
            dtype=numpy.int16) for key in (key1, key2))
    for d in map_data:
        results[0][d[key1]] = d[key2]
        results[1][d[key2]] = d[key1]
    return results

def decode_label_dict(row):
    result = {}
    for key, val in row.items():
        if key == 'category':
            result[key] = dict((c, int(n))
                for c, n in [re.match('^([^(]*)\(([^)]*)\)$', f).groups()
                    for f in val.split(';')])
        elif key == 'name':
            result[key] = val
        elif key == 'syns':
            result[key] = val.split(';')
        elif re.match('^\d+$', val):
            result[key] = int(val)
        elif re.match('^\d+\.\d*$', val):
            result[key] = float(val)
        else:
            result[key] = val
    return result

def decode_index_dict(row):
    result = {}
    for key, val in row.items():
        if key in ['image', 'split']:
            result[key] = val
        elif key in ['sw', 'sh', 'iw', 'ih']:
            result[key] = int(val)
        else:
            item = [s for s in val.split(';') if s]
            for i, v in enumerate(item):
                if re.match('^\d+$', v):
                    item[i] = int(v)
            result[key] = item
    return result

def index_has_any_data(row, categories):
    for c in categories:
        for data in row[c]:
            if data: return True
    return False

def index_has_all_data(row, categories):
    for c in categories:
        cat_has = False
        for data in row[c]:
            if data:
                cat_has = True
                break
        if not cat_has:
            return False
    return True

class SegmentationPrefetcher:
    '''
    SegmentationPrefetcher will prefetch a bunch of segmentation
    images using a multiprocessing pool, so you do not have to wait
    around while the files get opened and decoded.  Just request
    batches of images and segmentations calling fetch_batch().
    '''
    def __init__(self, segmentation, split=None, randomize=False,
            segmentation_shape=None, categories=None,
            batch_size=4, ahead=24):
        '''
        Constructor arguments:
        segmentation: The AbstractSegmentation to load.
        split: None for no filtering, or 'train' or 'val' etc.
        randomize: True to randomly shuffle order, or a random seed.
        categories: a list of categories to include in each batch.
        batch_size: number of data items for each batch.
        ahead: the number of data items to prefetch ahead.
        '''
        self.segmentation = segmentation
        self.split = split
        self.randomize = randomize
        self.random = random.Random()
        if randomize is not True:
            self.random.seed(randomize)
        self.categories = categories
        self.batch_size = batch_size
        self.ahead = ahead
        # Initialize the multiprocessing pool
        n_procs = cpu_count()
        original_sigint_handler = setup_sigint()
        self.pool = Pool(processes=n_procs, initializer=setup_sigint)
        restore_sigint(original_sigint_handler)
        # Prefilter the image indexes of interest
        self.indexes = range(segmentation.size())
        if split:
            self.indexes = [i for i in self.indexes
                    if segmentation.split(i) == split]
        self.random.shuffle(self.indexes)
        self.index = 0
        self.result_queue = []
        self.segmentation_shape = segmentation_shape

    def next_job(self):
        j = self.indexes[self.index]
        result = (j,
                self.segmentation.__class__,
                self.segmentation.metadata(j),
                self.segmentation.filename(j),
                self.categories,
                self.segmentation_shape)
        self.index += 1
        if self.index >= len(self.indexes):
            # Reshuffle every time through
            self.index = 0
            self.random.shuffle(self.indexes)
        return result

    def fetch_batch(self):
        try:
            self.refill_tasks()
            result = self.result_queue.pop(0)
            return result.get(31536000)
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            self.pool.terminate()
            raise

    def fetch_tensor_batch(self, bgr_mean=None):
        batch = self.fetch_batch()
        return self.form_caffe_tensors(batch, bgr_mean)

    def form_caffe_tensors(self, batch, bgr_mean=None):
        # Assemble a batch in [{'cat': data,..},..] format into
        # an array of batch tensors, the first for the image, and the
        # remaining for each category in self.categories, in order.
        # This also applies a random flip if needed
        batches = [[] for c in self.categories]
        for record in batch:
            default_shape = (1, record['sh'], record['sw'])
            for c, cat in enumerate(self.categories):
                if cat == 'image':
                    # Normalize image with right RGB order and mean
                    batches[c].append(self.normalize_image(
                        record[cat], bgr_mean))
                else:
                    batches[c].append(self.normalize_label(
                        cat, record[cat], default_shape))
        return [numpy.stack(b) for b in batches]

    def normalize_image(self, rgb_image, bgr_mean):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        img = numpy.array(rgb_image, dtype=numpy.float32)
        if (img.ndim == 2):
            img = numpy.repeat(img[:,:,None], 3, axis = 2)
        img = img[:,:,::-1]
        if bgr_mean is not None:
            img -= bgr_mean
        img = img.transpose((2,0,1))
        return img

    def normalize_label(self, cat, label_data, shape):
        dims = len(numpy.shape(label_data))
        if dims <= 2:
            # Scalar data on this channel: fill shape
            if dims == 1:
                label_data = label_data[0] if len(label_data) else 0
            return numpy.full(shape, label_data, dtype=numpy.int16)
        else:
            if dims == 3:
                label_data = label_data[0]
            return label_data[numpy.newaxis]

    def refill_tasks(self):
        # It will call the sequencer to ask for a sequence
        # of batch_size jobs (indexes with categories)
        # Then it will call pool.map_async
        while len(self.result_queue) < self.ahead:
            data = [self.next_job() for _ in range(self.batch_size)]
            self.result_queue.append(self.pool.map_async(prefetch_worker, data))

    def close():
        while len(self.result_queue):
            result = self.result_queue.pop(0)
            result.wait(0.001)
        self.pool.close()
        self.poool.cancel_join_thread()

def prefetch_worker(d):
    j, typ, m, fn, categories, segmentation_shape = d
    segs, shape = typ.resolve_segmentation(m, categories=categories)
    if segmentation_shape is not None:
        for k, v in segs.items():
            segs[k] = scale_segmentation(v, segmentation_shape)
        shape = segmentation_shape
    # Some additional metadata to provide
    segs['sh'], segs['sw'] = shape
    segs['i'] = j
    segs['fn'] = fn
    if categories is None or 'image' in categories:
        segs['image'] = imread(fn)
    return segs

def scale_segmentation(segmentation, dims, crop=False):
    '''
    Zooms a 2d or 3d segmentation to the given dims, using nearest neighbor.
    '''
    shape = numpy.shape(segmentation)
    if len(shape) < 2 or shape[-2:] == dims:
        return segmentation
    peel = (len(shape) == 2)
    if peel:
        segmentation = segmentation[numpy.newaxis]
    levels = segmentation.shape[0]
    result = numpy.zeros((levels, ) + dims,
            dtype=segmentation.dtype)
    ratio = (1,) + tuple(res / float(orig)
            for res, orig in zip(result.shape[1:], segmentation.shape[1:]))
    if not crop:
        safezoom(segmentation, ratio, output=result, order=0)
    else:
        ratio = max(ratio[1:])
        height = int(round(dims[0] / ratio))
        hmargin = (segmentation.shape[0] - height) // 2
        width = int(round(dims[1] / ratio))
        wmargin = (segmentation.shape[1] - height) // 2
        safezoom(segmentation[:, hmargin:hmargin+height,
            wmargin:wmargin+width],
            (1, ratio, ratio), output=result, order=0)
    if peel:
        result = result[0]
    return result

def safezoom(array, ratio, output=None, order=0):
    '''Like numpy.zoom, but does not crash when the first dimension
    of the array is of size 1, as happens often with segmentations'''
    dtype = array.dtype
    if array.dtype == numpy.float16:
        array = array.astype(numpy.float32)
    if array.shape[0] == 1:
        if output is not None:
            output = output[0,...]
        result = zoom(array[0,...], ratio[1:],
                output=output, order=order)
        if output is None:
            output = result[numpy.newaxis]
    else:
        result = zoom(array, ratio, output=output, order=order)
        if output is None:
            output = result
    return output.astype(dtype)

def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)

def restore_sigint(original):
    if original is None:
        original = signal.SIG_DFL
    signal.signal(signal.SIGINT, original)

def wants(what, option):
    if option is None:
        return True
    return what in option

if __name__ == '__main__':
    import numpy
    ds = SegmentationData('~/bulk/small_test', categories=['texture'])
    print 'sample has size', ds.size()
    i = ds.size() - 1
    print 'last filename is', ds.filename(i)
    print 'image shape read was', imread(ds.filename(i)).shape
    sd = ds.segmentation_data('texture', i)
    print 'texture seg shape read was', sd.shape
    print 'texture seg datatype', sd.dtype
    found = list(numpy.bincount(sd.ravel()).nonzero()[0])
    print 'unique textures seen', ';'.join(
            ds.name('texture', f) for f in found)
    fs, shape = ds.resolve_segmentation(ds.metadata(i))
    print 'full segmentation', shape, ', '.join(fs.keys())
    pf = SegmentationPrefetcher(ds, categories=['texture'],
            segmentation_shape=(96, 96))
    for i in range(5):
        dat = pf.fetch_batch()
        for d in dat:
            print i, d.keys()
    for i in range(5):
        dat = pf.fetch_tensor_batch()
        for d in dat:
            print i, numpy.shape(d)
    print 'all done'
