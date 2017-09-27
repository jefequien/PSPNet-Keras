import random
import numpy as np
import itertools
from scipy import misc, ndimage
import cv2

INPUT_SIZE = 473

def build_sliding_window(img, stride_rate, input_shape=(473,473)):
    '''
    Returns sliding window patches as a batch.
    '''
    h,w = img.shape[:2]
    crop_boxes = sliding_window_tiles(img, stride_rate)
    n = len(crop_boxes)

    crops = np.zeros((n, input_shape[0], input_shape[1], 3))
    for i in xrange(n):
        box = crop_boxes[i]
        crops[i] = crop_array(img, box)
    return crops

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
        crop_prob = crop_probs[i]

        probs[sh:eh,sw:ew,:] += crop_prob[0:eh-sh,0:ew-sw,:]
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
    crop_locs = list(itertools.product(hs,ws))

    boxes = []
    for loc in crop_locs:
        sh,sw = loc
        eh = min(h, sh + INPUT_SIZE)
        ew = min(w, sw + INPUT_SIZE)
        box = (sh,eh,sw,ew)
        boxes.append(box)
    return boxes


#####

def scale_and_crop_imgs(imgs):
    '''
    Scales and returns a random crop of images
    '''
    box = None
    outs = []
    for img in imgs:
        out = scale_maxside(img, maxside=512)
        if box is None:
            box = random_crop(out)
        out = crop_array(out, box)
        outs.append(out)
    return outs

def scale_imgs(imgs):
    '''
    Scale to 473x473
    '''
    outs = []
    for img in imgs:
        out = scale(img, (INPUT_SIZE,INPUT_SIZE))
        outs.append(out)
    return outs

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
