import os
from os.path import splitext, join, isfile, isdir
from os import environ, makedirs
import sys
import argparse
import numpy as np
import h5py
from scipy import misc

from keras import backend as K
import tensorflow as tf

from pspnet import PSPNet50, predict_sliding
from datasource import DataSource
import image_processor
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument('-r', '--randomize', action='store_true', default=False, help="Randomize image list")
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint to use')
    parser.add_argument('-s', '--scale', type=str, default='normal',
                        help='Scale to use',
                        choices=['normal',
                                 'medium',
                                 'big',
                                 'single'])
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    config = utils.get_config(args.project)
    datasource = DataSource(config, random=True)

    im_list = datasource.im_list
    if args.randomize:
        random.seed(3)
        random.shuffle(im_list)

    params = {}
    params['activation'] = 'softmax'

    # Output directory
    root_result = "predictions/default/{}/{}".format(params['activation'], args.scale)
    if args.checkpoint is not None:
        model = os.path.dirname(args.checkpoint)
        version = os.path.basename(args.checkpoint).split('-')[0]
        root_result = "predictions/{}/{}/{}".format(model, version, args.scale)
    print "Outputting to ", root_result

    root_mask = os.path.join(root_result, 'category_mask')
    root_prob = os.path.join(root_result, 'prob_mask')
    root_maxprob = os.path.join(root_result, 'max_prob')
    root_allprob = os.path.join(root_result, 'all_prob')

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        pspnet = PSPNet50(activation=params['activation'],
                            checkpoint=args.checkpoint)

        for im in im_list:
            print im

            fn_maxprob = os.path.join(root_maxprob, im.replace('.jpg', '.h5'))
            fn_mask = os.path.join(root_mask, im.replace('.jpg', '.png'))
            fn_prob = os.path.join(root_prob, im)
            fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))

            if os.path.exists(fn_maxprob):
                print "Already done."
                continue

            # make paths if not exist
            if not os.path.exists(os.path.dirname(fn_maxprob)):
                os.makedirs(os.path.dirname(fn_maxprob))
            if not os.path.exists(os.path.dirname(fn_mask)):
                os.makedirs(os.path.dirname(fn_mask))
            if not os.path.exists(os.path.dirname(fn_prob)):
                os.makedirs(os.path.dirname(fn_prob))
            if not os.path.exists(os.path.dirname(fn_allprob)):
                os.makedirs(os.path.dirname(fn_allprob))

            img = datasource.get_image(im)
            probs = None
            if args.scale == "single":
                probs = pspnet.predict(img)
            elif args.scale == "normal":
                img_s = image_processor.scale_maxside(img, maxside=512)
                probs_s = pspnet.predict_sliding(img_s)
                probs = image_processor.scale(probs_s, img.shape)
            elif args.scale == "medium":
                img_s = image_processor.scale_maxside(img, maxside=1028)
                probs_s = pspnet.predict_sliding(img_s)
                probs = image_processor.scale(probs_s, img.shape)
            elif args.scale == "big":
                img_s = image_processor.scale_maxside(img, maxside=2048)
                probs_s = pspnet.predict_sliding(img_s)
                probs = image_processor.scale(probs_s, img.shape)

            # probs is 150 x h x w
            probs = np.transpose(probs, (2,0,1))

            # Write output
            pred_mask = np.array(np.argmax(probs, axis=0) + 1, dtype='uint8')
            prob_mask = np.array(np.max(probs, axis=0)*255, dtype='uint8')
            max_prob = np.max(probs, axis=(1,2))
            all_prob = np.array(probs*255+0.5, dtype='uint8')
            
            # write to file
            misc.imsave(fn_mask, pred_mask)
            misc.imsave(fn_prob, prob_mask)
            with h5py.File(fn_maxprob, 'w') as f:
                f.create_dataset('maxprob', data=max_prob)
            with h5py.File(fn_allprob, 'w') as f:
                f.create_dataset('allprob', data=all_prob)



