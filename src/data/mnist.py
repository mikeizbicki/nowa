from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.training import queue_runner
from tensorflow.contrib.distributions import Bernoulli

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
IMAGE_COLORS = 1

TRAINING_SIZE = 55000

########################################

def testing_data(FLAGS):
    print('constructing testing input for mnist')
    datafiles=['test.tfrecords']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    return load_data_from_files(datapaths)

def training_data(FLAGS):
    # load raw data
    print('constructing training input for mnist')
    datafiles=['train.tfrecords']
    datadir=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    datapaths = [ os.path.join(datadir, datafile) for datafile in datafiles ]
    image,label = load_data_from_files(datapaths)

    # filter data for procid
    numdp = TRAINING_SIZE
    uidqueue = tf.FIFOQueue(1000,tf.int32)
    uidcounter = tf.Variable(0,name='uidcounter',dtype=tf.int32)
    enqueue_op = uidqueue.enqueue(uidcounter.assign((uidcounter+1)%numdp))
    qr = tf.train.QueueRunner(uidqueue,[enqueue_op])
    queue_runner.add_queue_runner(qr)
    uid = uidqueue.dequeue()

    shufq = tf.FIFOQueue(
        1000,
        [image.dtype,label.dtype,uid.dtype],
        shapes=[image.shape,label.shape,label.shape],
        )
    enqueue_op = tf.cond(
        tf.equal(uid%FLAGS.numproc,FLAGS.procid),
        lambda: shufq.enqueue([image,label,uid]),
        lambda: tf.no_op()
        )
    qr = tf.train.QueueRunner(shufq,[enqueue_op])
    queue_runner.add_queue_runner(qr)
    [image,label,uid]=shufq.dequeue()

    return image,label

def load_data_from_files(datapaths):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
        datapaths,
        shuffle=False
        )
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

    #label = tf.cond(
        #tf.equal(uid%1000,0),
        #lambda: tf.Print(label,[uid,label],message='uid,label='),
        #lambda: label
        #)

    return image,label
