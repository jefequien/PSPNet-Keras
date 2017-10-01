import os
from os.path import splitext, join, isfile, isdir
from os import environ, makedirs
import sys
import argparse
import numpy as np
import h5py
from scipy import misc

from datasource import DataSource
import image_processor
import utils

def open_h5file(file_path):
    with h5py.File(file_path, 'r') as f:
            output = f['allprob'][:]
            if output.dtype == 'uint8':
                return output.astype('float32')/255
            else:
                return output.astype('float32')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--project', type=str, required=True, help="Project name")
    parser.add_argument("-r", '--randomize', action='store_true', default=False, help="Randomize image list")
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint to use')
    parser.add_argument('-s', '--scale', type=str, default='normal',
                        help='Scale to use',
                        choices=['normal',
                                 'medium',
                                 'big',
                                 'single'])
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    config = utils.get_config(args.project)
    datasource = DataSource(config, random=True)

    im_list = datasource.im_list
    if args.randomize:
        random.seed(3)
        random.shuffle(im_list)

    # Output directory
    root_result = "predictions/default/{}/{}".format(params['activation'], args.scale)
    if args.checkpoint is not None:
        model = os.path.dirname(args.checkpoint)
        version = os.path.basename(args.checkpoint).split('-')[0]
        root_result = "predictions/{}/{}/{}".format(model, version, args.scale)
    print "Outputting to ", root_result

    root_allprob = os.path.join(root_result, 'all_prob')
    root_fix = os.path.join(root_result, 'fixed')

    for im in im_list:
        print im

        fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))
        fn_fix = os.path.join(root_fix, im.replace('.jpg', '.h5'))

        if not os.path.exists(fn_allprob):
            print "Not done."
            continue
        if os.path.exists(fn_fix):
            print "Already done."
            continue

        # make paths if not exist
        if not os.path.exists(os.path.dirname(fn_fix)):
            os.makedirs(os.path.dirname(fn_fix))

        probs = open_h5file(fn_allprob)
        all_prob = np.array(probs*255+0.5, dtype='uint8')
        with h5py.File(fn_fix, 'w') as f:
            f.create_dataset('allprob', data=all_prob)

