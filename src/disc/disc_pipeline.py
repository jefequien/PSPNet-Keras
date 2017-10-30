
from os.path import splitext, join, isfile, isdir
from os import environ, makedirs
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import utils
from utils.data import DataSource
from utils.evaluator import Evaluator

from disc import Discriminator
from data_generator_disc import DiscDataGenerator
import plot

CHECKPOINT_DIR = "../disc_checkpoints/"
if not isdir(CHECKPOINT_DIR):
    makedirs(CHECKPOINT_DIR)

class Pipeline:

    def __init__(self, name, lr=1e-3):
        config = utils.get_config("ade20k")
        config["pspnet_prediction"] = "../pspnet_prediction/sigmoid_normal3/weights.57/normal/"
        self.datasource = DataSource(config)

        self.name = name
        self.lr = lr

        self.main_dir = join(CHECKPOINT_DIR, self.name)

    def evaluate(self, category):
        evaluated_train = Evaluator("sigmoid_normal3", "ade20k", self.datasource)
        evaluated_val = Evaluator("sigmoid_normal3", "ade20k_val", self.datasource)
        im_list_train = evaluated_train.get_im_list_by_category(category)
        im_list_val = evaluated_val.get_im_list_by_category(category)

        print "Training..."
        self.train(category, im_list_train)
        print "Training Done"

        print "Running on validation data"
        output_fn = "{}-ade20k_val.txt".format(category)
        fn_val = self.run(category, im_list_val, output_fn=output_fn)
        plot.scatterplot(fn_val, evaluated_val)

        print "Running on training data"
        output_fn = "{}-ade20k.txt".format(category)
        fn_train = self.run(category, im_list_train, output_fn=output_fn)
        plot.scatterplot(fn_train, evaluated_train)

    def train(self, category, im_list):
        disc = self.get_latest_disc(category)

        data_generator = DiscDataGenerator(im_list, self.datasource, category)

        # Checkpoint callback
        checkpoint_dir = join(self.main_dir, "checkpoints")
        checkpoint_name = "weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5"
        checkpoint_path = join(checkpoint_dir, checkpoint_name)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss')
        callbacks_list = [checkpoint]
        disc.model.fit_generator(data_generator, 1000, epochs=100, callbacks=callbacks_list,
                 verbose=1, workers=6, use_multiprocessing=True, initial_epoch=epoch)

    def run(self, category, im_list, output_fn="tmp.txt"):
        disc = self.get_latest_disc(category)

        output_dir = join(self.main_dir, "predictions")
        output_path = join(output_dir, output_fn)
        recorder = Recorder(output_path, restart=False)

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
        return output_path


    def get_latest_disc(self, category):
        # Make checkpoint directory
        checkpoint_dir = join(self.main_dir, category, self.lr)
        if not isdir(checkpoint_dir):
            makedirs(checkpoint_dir)
        checkpoint, epoch = get_latest_checkpoint(category)
        disc = Discriminator(lr=self.lr, checkpoint=checkpoint)
        return disc


def get_latest_checkpoint(checkpoint_dir):
    # weights.00-1.52.hdf5
    epoch = 0
    filename = None
    for fn in os.listdir(checkpoint_dir):
        num = fn.split('-')[0].split('.')[1]
        e = int(num) + 1
        if e > epoch:
            epoch = e
            filename = fn

    if filename is None:
        return None, epoch
    else:
        return join(checkpoint_dir, filename), epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=int, required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print(args)

        name = "run1"
        pipeline = Pipeline(name, lr=args.learning_rate)
        pipeline.evaluate(args.category)



