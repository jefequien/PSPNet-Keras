
from os.path import splitext, join, isfile, isdir
from os import environ, makedirs
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from pspnet import PSPNet50
from datasource import DataSource
from data_generator import DataGenerator
import utils

def train(net, datasource, checkpoint_dir, initial_epoch=0):
    filename = "weights.{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint_path = join(checkpoint_dir, filename)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss')
    callbacks_list = [checkpoint]

    print("Training...")
    net.fit_generator(DataGenerator(datasource), 1000, epochs=100, callbacks=callbacks_list,
             verbose=1, workers=1, initial_epoch=initial_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    project = "local"
    config = utils.get_config(project)
    datasource = DataSource(config, random=True)

    checkpoint_dir = join("weights", "checkpoints", args.model)
    if not isdir(checkpoint_dir):
        makedirs(checkpoint_dir)

    checkpoint, epoch = (None,0)
    if args.resume:
        checkpoint,epoch = utils.get_latest_checkpoint(checkpoint_dir)

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)

        if "pspnet50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights="pspnet50_ade20k",
                              checkpoint=checkpoint)

        train(pspnet.model, datasource, checkpoint_dir, initial_epoch=epoch)


