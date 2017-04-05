#!/usr/bin/env python2.7

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import sys
import time
import importlib

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from common import *

################################################################################

def main(_):

    # process command line args
    if FLAGS.numproc <= FLAGS.procid or FLAGS.procid < 0:
        print("procid/numproc combination invalid", file=sys.stderr)
        sys.exit(1)

    module_dataset='models.%s.common'%FLAGS.dataset
    print('loading module %s'%module_dataset)
    datainfo=importlib.import_module(module_dataset)

    module_model='models.%s.%s'%(FLAGS.dataset,FLAGS.model)
    print('loading module %s'%module_model)
    model=importlib.import_module(module_model)

    # load and sample data
    (dataset_train,dataset_valid,dataset_test) = model.loaddata(FLAGS)

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
            session_conf=tf.ConfigProto(
                )
        else:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
                )
        sess = tf.Session(config=session_conf)

        # train model
        trainmodel(FLAGS,sess,dataset_train,dataset_test,train_op,eval_correct,{})

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.001,
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
            '--dataset',
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
    parser.add_argument(
            '--induced_bias',
            type=float,
            default=0
            )

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print('unparsed arguments: ',unparsed)
        exit(1)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
