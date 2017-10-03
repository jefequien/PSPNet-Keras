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
import layers_builder as layers
import utils
import utils_image
import utils_pspnet
#import matplotlib.pyplot as plt

__author__ = "Vlad Kryvoruchko, Chaoyue Wang, Jeffrey Hu & Julian Tatsch"


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017."""

    def __init__(self, params, checkpoint=None):
        print("params %s" % params)
        print("checkpoint %s" % checkpoint)
        """Instanciate a PSPNet."""
        self.input_shape = params['input_shape']

        if checkpoint is not None:
            print("Loading from checkpoint %s" % checkpoint)
            self.model = load_model(checkpoint)
        else:
            # Load cached keras model
            model_path = join("weights", "keras", params['name'] + "_" + params['activation'] + ".hdf5")
            if isfile(model_path):
               print("Cached Keras model found, loading %s" % model_path)
               self.model = load_model(model_path)
            else:
                print("No Keras model found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=params['nb_classes'],
                                                 resnet_layers=params['resnet_layers'],
                                                 input_shape=params['input_shape'],
                                                 activation=params['activation'])
                self.set_npy_weights(params['name'], model_path)

    def predict(self, img):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]
        if img.shape[0:2] != self.input_shape:
            print("Input %s not fitting for network size %s, resizing. You may want to try sliding prediction for better results." % (img.shape[0:2], self.input_shape))
            img = misc.imresize(img, self.input_shape)
        input_data = utils_image.preprocess_image(img, self.input_shape)
        input_data = input_data[np.newaxis, :, :, :]  # Append sample dimension for keras
        # utils_pspnet.debug(self.model, input_data)

        prediction = self.model.predict(input_data)[0]

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = prediction.shape[:2]
            prediction = ndimage.zoom(prediction, (1.*h_ori/h, 1.*w_ori/w, 1.),
                                      order=1, prefilter=False)
        return prediction

    def predict_sliding(self, img):
        input_data = preprocess_sliding_image(img)
        n = input_data.shape[0]
        print("Needs %i prediction tiles" % n)

        predictions = []
        batch_size = 8
        for i in range(0, n, batch_size):
            print("Predicting tiles %i to %i" % (i, min(i + batch_size, n) - 1))
            batch = input_data[i:i + batch_size]
            predictions.append(self.model.predict(batch))
        prediction = np.concatenate(predictions, axis=0)
        prediction = postprocess_sliding_image(img, prediction)
        return prediction

    def set_npy_weights(self, name, output_path):
        """Set weights from the intermediary npy file."""
        npy_weights_path = join("weights", "npy", name + ".npy")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path).item()

        whitelist = ["InputLayer", "Activation", "ZeroPadding2D", "Add", "MaxPooling2D", "AveragePooling2D", "Lambda", "Concatenate", "Dropout"]

        weights_set = 0
        for layer in self.model.layers:
            print("Processing %s" % layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name]['mean'].reshape(-1)
                variance = weights[layer.name]['variance'].reshape(-1)
                scale = weights[layer.name]['scale'].reshape(-1)
                offset = weights[layer.name]['offset'].reshape(-1)

                self.model.get_layer(layer.name).set_weights([mean, variance,
                                                             scale, offset])
                weights_set += 1
            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name]['weights']
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception:
                    biases = weights[layer.name]['biases']
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                 biases])
                weights_set += 1
            elif layer.__class__.__name__ in whitelist:
                # print("Nothing to set in %s" % layer.__class__.__name__)
                pass
            else:
                print("Warning: Did not find weights for keras layer %s in numpy weights" % layer)

        print("Set a total of %i weights" % weights_set)

        print('Finished importing weights.')

        print("Writing keras model")
        self.model.save(output_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes=150, name="pspnet50_ade20k", input_shape=(473, 473), activation="softmax", checkpoint=None):
        """Instanciate a PSPNet50."""
        params = {'nb_classes': nb_classes,
                    'input_shape': input_shape,
                    'name': name,
                    'resnet_layers': 50,
                    'activation': activation}
        PSPNet.__init__(self, params, checkpoint=checkpoint)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, name, input_shape, activation="softmax", checkpoint=None):
        """Instanciate a PSPNet101."""
        params = {'nb_classes': nb_classes,
                    'input_shape': input_shape,
                    'name': name,
                    'resnet_layers': 101,
                    'activation': activation}
        PSPNet.__init__(self, params, checkpoint=checkpoint)

def preprocess_sliding_image(img, input_shape):
    stride_rate = 2./3
    preprocessed = utils_image.preprocess_image(img)
    input_data = utils_image.build_sliding_window(preprocessed, stride_rate, input_shape=input_shape)
    return input_data

def postprocess_sliding_image(img, prediction):
    stride_rate = 2./3
    prediction = utils_image.assemble_sliding_window_tiles(img, stride_rate, prediction)
    return prediction


# def pad_image(img, target_size):
#     """Pad an image up to the target size."""
#     rows_missing = target_size[0] - img.shape[0]
#     cols_missing = target_size[1] - img.shape[1]
#     padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
#     return padded_img


# def visualize_prediction(prediction):
#     """Visualize prediction."""
#     cm = np.argmax(prediction, axis=2) + 1
#     color_cm = utils.add_color(cm)
#     plt.imshow(color_cm)
#     plt.show()

# def predict_sliding(full_image, net):
#     """Predict on tiles of exactly the network input shape so nothing gets squeezed."""
#     tile_size = net.input_shape
#     classes = net.model.outputs[0].shape[3]
#     overlap = 1/3

#     stride = ceil(tile_size[0] * (1 - overlap))
#     tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
#     tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
#     print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
#     full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
#     count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
#     tile_counter = 0
#     for row in range(tile_rows):
#         for col in range(tile_cols):
#             x1 = int(col * stride)
#             y1 = int(row * stride)
#             x2 = min(x1 + tile_size[1], full_image.shape[1])
#             y2 = min(y1 + tile_size[0], full_image.shape[0])
#             x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
#             y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

#             img = full_image[y1:y2, x1:x2]
#             padded_img = pad_image(img, tile_size)
#             # plt.imshow(padded_img)
#             # plt.show()
#             tile_counter += 1
#             print("Predicting tile %i" % tile_counter)
#             padded_prediction = net.predict(padded_img)
#             prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
#             count_predictions[y1:y2, x1:x2] += 1
#             full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

#     # average the predictions in the overlapping regions
#     full_probs /= count_predictions
#     # visualize normalization Weights
#     # plt.imshow(np.mean(count_predictions, axis=2))
#     # plt.show()
#     return full_probs

def save(pred, output_path="out.jpg"):
    print("Writing results...")
    cm = np.argmax(probs, axis=2) + 1
    pm = np.max(probs, axis=2)
    color_cm = utils.add_color(cm)
    # color cm is [0.0-1.0] img is [0-255]
    alpha_blended = 0.5 * color_cm * 255 + 0.5 * img
    filename, ext = splitext(output_path)
    misc.imsave(filename + "_seg" + ext, color_cm)
    misc.imsave(filename + "_probs" + ext, pm)
    misc.imsave(filename + "_seg_blended" + ext, alpha_blended)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('-s', '--sliding', action='store_true',
                        help="Whether the network should be slided over the original image for prediction.")
    args = parser.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        print(args)

        if "pspnet50" in args.model:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              name=args.model)
        elif "pspnet101" in args.model:
            if "cityscapes" in args.model:
                pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                   name=args.model)
            if "voc2012" in args.model:
                pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                   name=args.model)
        else:
            print("Network architecture not implemented.")

        if args.sliding:
            probs = pspnet.predict_sliding(img)
        else:
            probs = pspnet.predict(img)

        save(probs, output_path=args.output_path)

