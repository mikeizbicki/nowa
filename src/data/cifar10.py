from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import data_provider
from tensorflow.contrib.slim.python.slim.data import parallel_reader

import preprocessing.cifarnet_preprocessing as preprocessing

NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_COLORS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_COLORS

########################################

def testing_data(FLAGS):
    print('constructing testing input for cifar10')
    datafiles=['cifar10_test.tfrecord']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    _,dataqueue = parallel_reader.parallel_read(
        datapaths,
        tf.TFRecordReader,
        seed=FLAGS.seed,
        num_epochs=1
        )
    image,label=load_data_from_files(dataqueue)
    image=preprocessing.preprocess_image(image,IMAGE_HEIGHT,IMAGE_WIDTH,is_training=False)
    return image,label

def training_data(FLAGS):
    print('constructing training input for cifar10')
    datafiles=['train.tfrecords']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    _,dataqueue = parallel_reader.parallel_read(
        datapaths,
        tf.TFRecordReader,
        seed=FLAGS.seed
        )
    image,label=load_data_from_files(dataqueue)
    image=preprocessing.preprocess_image(image,IMAGE_HEIGHT,IMAGE_WIDTH,is_training=True)
    return image,label

def load_data_from_files(dataqueue):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [],
            tf.int64,
            default_value=tf.zeros([],dtype=tf.int64)
            ),
        }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    [image,label] = decoder.decode(dataqueue,['image','label'])

    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image,label
