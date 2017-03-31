#!/usr/bin/python

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import importlib

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from common import *
from models.common import *

# Basic model parameters as external flags.
FLAGS = None

################################################################################

def main(_):

    # process command line args
    if FLAGS.numproc <= FLAGS.procid or FLAGS.procid < 0:
        print("procid/numproc combination invalid", file=sys.stderr)
        sys.exit(1)
    model = importlib.import_module("models.%s" % FLAGS.model)

    # prepare logging
    local_log_dir=os.path.join(FLAGS.log_dir_out, '%d-%s.%s-%d-%d'%(FLAGS.seed,FLAGS.same_seed,FLAGS.model,FLAGS.numproc,FLAGS.procid))
    if tf.gfile.Exists(local_log_dir):
        tf.gfile.DeleteRecursively(local_log_dir)
    tf.gfile.MakeDirs(local_log_dir)

    # load and sample data
    data_sets = read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    np.random.seed(FLAGS.seed)
    np.random.shuffle(data_sets.train._images)
    np.random.seed(FLAGS.seed)
    np.random.shuffle(data_sets.train._labels)
    np.random.seed(FLAGS.seed+1)
    np.random.shuffle(data_sets.validation._images)
    np.random.seed(FLAGS.seed+1)
    np.random.shuffle(data_sets.validation._labels)

    trainnm = data_sets.train._images.shape[0]
    trainn  = int(trainnm / FLAGS.numproc)
    data_sets.train._images=data_sets.train._images[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)]
    data_sets.train._labels=data_sets.train._labels[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)]
    data_sets.train._num_examples=trainn

    validationnm = data_sets.validation._images.shape[0]
    validationn  = int(validationnm / FLAGS.numproc)
    data_sets.validation._images=data_sets.validation._images[validationn*FLAGS.procid:validationn*(FLAGS.procid+1)]
    data_sets.validation._labels=data_sets.validation._labels[validationn*FLAGS.procid:validationn*(FLAGS.procid+1)]
    data_sets.validation._num_examples=validationn

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # ensure that tf will be deterministic
        seed=FLAGS.seed
        if not FLAGS.same_seed:
            seed+=FLAGS.procid
        tf.set_random_seed(seed)

        # create the graph
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        dropout_rate = tf.placeholder(tf.float32)
        logits = model.inference(images_placeholder,dropout_rate)
        loss = model.loss(logits, labels_placeholder)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(logits, labels_placeholder)
        
        # create a session for running Ops on the Graph.
        if FLAGS.maxcpu==0:
            session_conf=tf.ConfigProto()
        else:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
                )
        sess = tf.Session(config=session_conf)

        # train model
        trainmodel(FLAGS,sess,data_sets.train,data_sets.test,loss,train_op,eval_correct)

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.0001,
            help='Initial learning rate.'
    )
    parser.add_argument(
            '--max_steps',
            type=int,
            default=2000,
            help='Number of steps to run trainer.'
    )
    parser.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
            '--input_data_dir',
            type=str,
            default='data',
            help='Directory to put the input data.'
    )
    parser.add_argument(
            '--log_dir_out',
            type=str,
            default='log/local',
            help='Directory to put the log data.'
    )
    parser.add_argument(
            '--fake_data',
            default=False,
            help='If true, uses fake data for unit testing.',
            action='store_true'
    )
    parser.add_argument(
            '--numproc',
            type=int,
            default=1
            )
    parser.add_argument(
            '--procid',
            type=int,
            default=0
            )
    parser.add_argument(
            '--seed',
            type=int,
            default=0
            )
    parser.add_argument(
            '--model',
            type=str,
            default=''
            )
    parser.add_argument(
            '--same_seed',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--maxcpu',
            type=int,
            default=0
            )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
