#!/usr/bin/env python
"""
This module is a Keras/Tensorflow based implementation of Pyramid Scene Parsing Networks.

Original paper & code published by Hengshuang Zhao et al. (2017)
"""
from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf

import utils
from utils import image_utils
from utils.datasource import open_file

class Discriminator(object):
    """Discriminator for classes"""

    def __init__(self, lr=1e-4, checkpoint=None):
        print("checkpoint %s" % checkpoint)
        """Instanciate a Resnet discriminator"""

        if checkpoint is None:
            print("Building Resnet discriminator")
            self.model = self.build_model(lr)
        else:
            print("Loading from checkpoint %s" % checkpoint)
            self.model = load_model(checkpoint)

        self.input_shape = (473,473)

    def build_model(self, lr):
        inp = Input((473,473,4))
        resnet = ResNet50(input_tensor=inp, weights=None, include_top=False)
        x = Flatten()(resnet.outputs[0])
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=output)

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                        loss="binary_crossentropy",
                        metrics=['accuracy'])
        return model

    def predict(self, img, prediction, category):
        """
        Predict segmentation for an image.

        Arguments:
            img: must be rowsxcolsx3
            prediction: must be rowsxcolsxN-1
            category: must be 1 ... N
        """
        img_resized = misc.imresize(img, self.input_shape)
        img_preprocessed = image_utils.preprocess_image(img_resized)
        
        input_data = prepare_disc_data(img_preprocessed, prediction, category)
        input_data = input_data[np.newaxis, :, :, :]  # Append sample dimension for keras

        prediction = self.model.predict(input_data)[0]
        return prediction

def prepare_disc_data(img, prediction, category):
    s = prediction[category-1]
    s = image_utils.scale(s, img.shape[:2])
    s = s > 0.5
    data = np.concatenate((img, s[:,:,np.newaxis]), axis=2)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-p', '--prediction', type=str, default='example_results/ade20k.h5',
                        help='Path to output')
    parser.add_argument('-c', '--category', type=int, default='1',
                        help='Model/Weights to use')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        prediction = open_file(args.prediction, ftype="ap")
        print(args)

        disc = Discriminator()

        prob = disc.predict(img, prediction, args.category)
        print(prob)

