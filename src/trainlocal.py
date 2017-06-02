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
import tflearn

from common import *
from parser import *

################################################################################

if __name__ == '__main__':

    FLAGS=parseArgs()

    # process command line args
    if FLAGS.numproc <= FLAGS.procid or FLAGS.procid < 0:
        print("procid/numproc combination invalid", file=sys.stderr)
        sys.exit(1)

    module_dataset='data.%s'%FLAGS.dataset
    print('loading module %s'%module_dataset)
    datainfo=importlib.import_module(module_dataset)

    module_model='models.%s'%FLAGS.model
    print('loading module %s'%module_model)
    model=importlib.import_module(module_model)

    # train model
    with tf.Graph().as_default():

        # make computations as deterministic as possible
        # because tf uses lots of parallelism, there will still be some nondeterminism
        seed=FLAGS.seed
        if not FLAGS.same_seed:
            seed+=FLAGS.procid
        tf.set_random_seed(seed)

        # create inputs
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                x,y = datainfo.training_data(FLAGS)
                X,Y = tf.train.shuffle_batch(
                    [x,y],
                    batch_size=FLAGS.batch_size,
                    num_threads=16,
                    capacity=5*FLAGS.batch_size,
                    seed=FLAGS.seed,
                    min_after_dequeue=4*FLAGS.batch_size
                    )

        # create computations
        print('creating graph')
        logits = model.inference(X,datainfo)
        loss = model.loss(logits, Y)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(logits, Y)

        # train model
        print('training model')
        trainmodel(FLAGS,[],[],train_op,eval_correct,{})

