
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
from utils.datasource import DataSource
from utils.evaluator import Evaluator

from disc import Discriminator
from data_generator import DiscDataGenerator

def train(disc, data_generator, checkpoint_dir, initial_epoch=0):
    filename = "weights.{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint_path = join(checkpoint_dir, filename)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss')
    callbacks_list = [checkpoint]

    print("Training...")
    disc.model.fit_generator(data_generator, 1000, epochs=100, callbacks=callbacks_list,
             verbose=1, workers=6, initial_epoch=initial_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help="Name to identify this model")
    parser.add_argument('-c', '--category', type=int, required=True)
    parser.add_argument('-p', '--prediction', type=str, required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, required=True, default=1e-4)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    # Checkpoint handling
    checkpoint_name = "{}-{}-{}".format(args.name, args.learning_rate, args.category)
    checkpoint_dir = join("weights", "checkpoints", "disc", checkpoint_name)
    if not isdir(checkpoint_dir):
        makedirs(checkpoint_dir)

    checkpoint, epoch = (None, 0)
    if args.resume:
        checkpoint, epoch = utils.get_latest_checkpoint(checkpoint_dir)

    # Data handling
    project = "ade20k"
    config = utils.get_config(project)
    config["pspnet_prediction"] = args.prediction
    datasource = DataSource(config)

    # Image list
    evaluator = Evaluator(args.name, project, datasource)
    im_list = evaluator.get_im_list_by_category(args.category)
    print args.category, len(im_list)

    data_generator = DiscDataGenerator(im_list, datasource, args.category)

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print(args)

        disc = Discriminator(lr=args.learning_rate, checkpoint=checkpoint)
        train(disc, data_generator, checkpoint_dir, initial_epoch=epoch)


