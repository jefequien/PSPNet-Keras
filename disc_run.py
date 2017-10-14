from os import environ, makedirs
from os.path import join, isfile, isdir, dirname, basename
import argparse
import numpy as np

from keras import backend as K
import tensorflow as tf

import utils
from utils.datasource import DataSource
from utils.evaluator import Evaluator
from utils.recorder import Recorder

from disc import Discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help="Name identifier")
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Checkpoint to use')
    parser.add_argument('--prediction', type=str, required=True, help='Checkpoint to use')
    parser.add_argument('--category', type=int, required=True, help='Checkpoint to use')
    parser.add_argument('-r', '--restart', action='store_true', default=False, help="Restart")
    parser.add_argument('--randomize', action='store_true', default=False, help="Randomize image list")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()
    print(args)

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    # Data
    config = utils.get_config(args.project)
    config["pspnet_prediction"] = args.prediction
    datasource = DataSource(config)

    # Image list
    evaluator = Evaluator(args.name, args.project, datasource)
    im_list = evaluator.get_im_list_by_category(args.category)
    print args.category, len(im_list)
    if args.randomize:
        random.seed(3)
        random.shuffle(im_list)

    # Output directory
    recorder = Recorder("{}-{}-{}.txt".format(args.name, args.project, args.category), restart=args.restart)

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        disc = Discriminator(checkpoint=args.checkpoint)

        for im in im_list:
            if recorder.contains(im):
                print im, recorder.get(im), "Done."
                continue

            img, _ = datasource.get_image(im)
            gt, _ = datasource.get_ground_truth(im, one_hot=True)
            ap, _ = datasource.get_all_prob(im)
            pr_prob = disc.predict(img, ap, args.category)
            gt_prob = disc.predict(img, gt, args.category)
            
            print "{} {} {}".format(im, pr_prob, gt_prob)
            recorder.save(im,[pr_prob, gt_prob])
