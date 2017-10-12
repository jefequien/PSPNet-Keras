import random
import numpy as np
import itertools
from scipy import misc, ndimage
import cv2

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order
INPUT_SIZE = 473

def preprocess_image(img):
    """Preprocess an image as input."""
    float_img = img.astype('float16')
    centered_image = float_img - DATA_MEAN
    bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
    return bgr_image

def build_sliding_window(img, stride_rate, input_shape=(473,473)):
    '''
    Returns sliding window patches as a batch.
    '''
    h,w = img.shape[:2]
    boxes = sliding_window_tiles(img, stride_rate)
    n = len(boxes)

    tiles = np.zeros((n, input_shape[0], input_shape[1], 3))
    for i in xrange(n):
        box = boxes[i]
        tiles[i] = crop_array(img, box)
    return tiles

def assemble_sliding_window_tiles(img, stride_rate, tiles):
    '''
    Combines sliding window predictions into one image
    '''
    h, w = img.shape[:2]
    n, h_t, w_t, k = tiles.shape

    probs = np.zeros((h, w, k), dtype=np.float32)
    cnts = np.zeros((h,w,1))

    boxes = sliding_window_tiles(img, stride_rate, tile_size=h_t)
    n = len(boxes)
    for i in xrange(n):
        sh,eh,sw,ew = boxes[i]
        tile = tiles[i]
        probs[sh:eh,sw:ew,:] += tile[0:eh-sh,0:ew-sw,:]
        cnts[sh:eh,sw:ew,0] += 1

    assert cnts.min()>=1
    probs /= cnts
    assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())
    return probs

def sliding_window_tiles(img, stride_rate, tile_size=473):
    '''
    Sliding window crop box locations
    '''
    # Get top-left corners
    h, w = img.shape[:2]
    stride = tile_size * stride_rate

    hs_upper = max(1,h-(tile_size-stride))
    ws_upper = max(1,w-(tile_size-stride))
    hs = np.arange(0,hs_upper,stride, dtype=int)
    ws = np.arange(0,ws_upper,stride, dtype=int)
    # Fix last stride
    if len(hs) > 1:
        remainder = h - hs[-1]
        delta = tile_size - remainder
        hs[-1] = hs[-1] - delta
    if len(ws) > 1:
        remainder = w - ws[-1]
        delta = tile_size - remainder
        ws[-1] = ws[-1] - delta
    crop_locs = list(itertools.product(hs,ws))

    boxes = []
    for loc in crop_locs:
        sh,sw = loc
        eh = min(h, sh + INPUT_SIZE)
        ew = min(w, sw + INPUT_SIZE)
        box = (sh,eh,sw,ew)
        boxes.append(box)
    return boxes

def crop_array(a, box):
    h,w,c = a.shape
    sh,eh,sw,ew = box
    crop = np.zeros((INPUT_SIZE,INPUT_SIZE, c))
    crop[0:eh-sh,0:ew-sw] = a[sh:eh,sw:ew]
    return crop

def random_crop(img):
    h,w,_ = img.shape
    sh = 0
    sw = 0
    if h > INPUT_SIZE:
        sh = random.randint(0,h-INPUT_SIZE)
    if w > INPUT_SIZE:
        sw = random.randint(0,w-INPUT_SIZE)
    eh = min(h,sh + INPUT_SIZE)
    ew = min(w,sw + INPUT_SIZE)
    box = (sh,eh,sw,ew)
    return box

def scale_maxside(a, maxside=512):
    h,w = a.shape[:2]
    long_side = max(h, w)
    r = 1.*maxside/long_side # Make long_side == scale_size

    h_t = h*r
    w_t = w*r
    return scale(a, (h_t,w_t))

def scale(a, shape):
    h_t,w_t = shape[:2]
    h,w = a.shape[:2]
    r_h = 1.*h_t/h
    r_w = 1.*w_t/w

    if np.ndim(a) == 3 and a.shape[2] == 3:
        # Image, use bilinear
        return ndimage.zoom(a, (r_h,r_w,1.), order=1, prefilter=False)
    elif np.ndim(a) == 3 and a.dtype == 'float32':
        # Probs, use bilinear
        return ndimage.zoom(a, (r_h,r_w,1.), order=1, prefilter=False)
    else:
        # Ground truth, use nearest
        if np.ndim(a) == 2:
            return ndimage.zoom(a, (r_h,r_w), order=0, prefilter=False)
        else:
            return ndimage.zoom(a, (r_h,r_w,1.), order=0, prefilter=False)
