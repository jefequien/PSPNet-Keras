
from os.path import splitext, join, isfile, isdir
from os import environ, makedirs
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from disc import Discriminator
from datasource import DataSource
from data_generator import DataGenerator
import utils

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
    parser.add_argument('-c', '--category', required=True, type=int)
    parser.add_argument('-p', '--predictions', required=True, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    project = "ade20k"
    config = utils.get_config(project)
    config["predictions"] = args.predictions
    datasource = DataSource(config, random=True)

    checkpoint_dir = join("weights", "checkpoints", "disc", args.name)
    if not isdir(checkpoint_dir):
        makedirs(checkpoint_dir)

    checkpoint, epoch = (None, 0)
    if args.resume:
        checkpoint, epoch = utils.get_latest_checkpoint(checkpoint_dir)

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        disc = Discriminator(category, checkpoint=checkpoint)

        data_generator = DiscDataGenerator(datasource, args.category)
        train(pspnet, data_generator, checkpoint_dir, initial_epoch=epoch)


