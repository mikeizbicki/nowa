from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from models.mnist.common import *


def inference2(images,_):
    network = input_data(placeholder=images)
    network = tf.reshape(network,[-1,28,28,1])
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    #network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    #network = dropout(network, 0.8)
    #network = fully_connected(network, 10, activation='softmax')
    return network

def inference(images,_):
    #input_layer = tflearn.input_data(shape=[None, 784])
    input_layer = tflearn.input_data(placeholder=images)
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')
    return softmax
