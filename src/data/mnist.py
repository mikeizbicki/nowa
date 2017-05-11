from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
IMAGE_COLORS = 1

########################################

def testing_data(FLAGS):
    print('constructing testing input for mnist')
    datafiles=['test.tfrecords']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    filename_queue = tf.train.string_input_producer(datapaths,num_epochs=1)
    return load_data_from_files(filename_queue)

def training_data(FLAGS):
    print('constructing training input for mnist')
    datafiles=['train.tfrecords']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    filename_queue = tf.train.string_input_producer(datapaths)
    return load_data_from_files(filename_queue)

def load_data_from_files(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)

    #image,label=tf.cond(
        #tf.equal(
            #tf.string_to_hash_bucket_strong(key,FLAGS.numproc,[0,FLAGS.seed]),
            #FLAGS.procid
            #),
        #lambda: [tf.expand_dims(image,0),tf.expand_dims(label,0)],
        #lambda: [tf.zeros([0]+image.get_shape().as_list()),tf.zeros([0],dtype=tf.int32)]
        #)

    return image,label

