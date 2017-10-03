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
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

import utils
import image_processor
#import matplotlib.pyplot as plt

__author__ = "Vlad Kryvoruchko, Chaoyue Wang, Jeffrey Hu & Julian Tatsch"


# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class Discriminator(object):
    """Discriminator for classes"""

    def __init__(self, category, checkpoint=None):
        print("checkpoint %s" % checkpoint)
        """Instanciate a Resnet discriminator"""

        if checkpoint is None:
            print("Building Resnet discriminator")
            self.model = self.build_model()
        else:
            print("Loading from checkpoint %s" % checkpoint)
            self.model = load_model(checkpoint)

        self.category = category
        self.

    def build_model(self):
        inp = Input((473,473,4))
        resnet = ResNet50(input_tensor=inp, weights=None, include_top=False)
        x = Flatten()(resnet.outputs[0])
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=output)

        sgd = SGD(lr=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                        loss="binary_crossentropy",
                        metrics=['accuracy'])
        return model

    def predict(self, img):
        """
        Predict segmentation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]
        if img.shape[0:2] != self.input_shape:
            print("Input %s not fitting for network size %s, resizing. You may want to try sliding prediction for better results." % (img.shape[0:2], self.input_shape))
            img = misc.imresize(img, self.input_shape)
        input_data = self.preprocess_image(img)
        input_data = input_data[np.newaxis, :, :, :]  # Append sample dimension for keras
        # utils.debug(self.model, input_data)

        prediction = self.model.predict(input_data)[0]

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = prediction.shape[:2]
            prediction = ndimage.zoom(prediction, (1.*h_ori/h, 1.*w_ori/w, 1.),
                                      order=1, prefilter=False)
        return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, default='1', help='Model/Weights to use')
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        print(args)

        disc = Discriminator(args.category)

        probs = pspnet.predict(img)

        save(probs, output_path=args.output_path)

