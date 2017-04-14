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

import preprocessing.inception_preprocessing as preprocessing

NUM_CLASSES = 1001
IMAGE_SIZE = 224
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_COLORS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_COLORS
PADDING = 4

########################################

def testing_data(FLAGS):
    print('constructing testing input for imagenet')
    #datafiles=['validation-00001-of-00128']
    datafiles=['validation-%05d-of-00128'%i for i in range(128)]
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
    #image = tf.to_float(image)
    #image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    #image = tf.image.per_image_standardization(image)
    return image,label

def training_data(FLAGS):
    print('constructing training input for imagenet')
    datafiles=['train-%05d-of-01024'%i for i in range(1024)]
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
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    [image,label] = decoder.decode(dataqueue,['image','label'])

    return image,label

