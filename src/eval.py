#!/usr/bin/env python2.7

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import importlib
import re
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import tflearn

from common import *
from parser import *

################################################################################

########################################

if __name__ == '__main__':

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

    # test model
    with tf.Graph().as_default():

        # make computations as deterministic as possible
        # because tf uses lots of parallelism, there will still be some nondeterminism
        seed=FLAGS.seed
        if not FLAGS.same_seed:
            seed+=FLAGS.procid
        tf.set_random_seed(seed)

        # create graph
        with tf.name_scope('input'):
            x,y = datainfo.testing_data(FLAGS)
            X,Y = tf.train.batch(
                [x,y],
                batch_size=FLAGS.batch_size,
                num_threads=16,
                capacity=5*FLAGS.batch_size
                )
        logits = model.inference(X,datainfo,is_training=False)
        eval_correct = model.evaluation(logits, Y)

        # get list of checkpoints
        modeldir=flags2logdir(FLAGS)
        checkpoints=sorted(list(set(map(int,sum(
            map(
                lambda x:re.findall('\d+',x),
                sum(map(lambda x:re.findall(r'model.ckpt.\d+',x), os.listdir(modeldir)),[])
                ),
            [],
            )))))
        print('checkpoints for %s'%modeldir)

        # process each checkpoint
        for checkpoint in checkpoints:

            sess=tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(modeldir,'model.ckpt-%d'%checkpoint))

            # eval loop
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step=0
                tot=0
                while not coord.should_stop():
                    tot+=sess.run(eval_correct)
                    step+=1
            except tf.errors.OutOfRangeError:
                pass
                #print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))

            ave=float(tot)#/(step*FLAGS.batch_size)
            print('  %8d: %f'%(checkpoint,ave))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            sess.close()
