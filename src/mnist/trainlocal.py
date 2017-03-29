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
from eval import do_eval
from models.common import *

# Basic model parameters as external flags.
FLAGS = None

def main(_):

    # process command line args
    if FLAGS.numproc <= FLAGS.procid or FLAGS.procid < 0:
        print("procid/numproc combination invalid", file=sys.stderr)
        sys.exit(1)
    model = importlib.import_module("models.%s" % FLAGS.model)

    # prepare logging
    local_log_dir=os.path.join(FLAGS.log_dir, '%d-%s.%s-%d-%d'%(FLAGS.seed,FLAGS.same_seed,FLAGS.model,FLAGS.numproc,FLAGS.procid))
    if tf.gfile.Exists(local_log_dir):
        tf.gfile.DeleteRecursively(local_log_dir)
    tf.gfile.MakeDirs(local_log_dir)

    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    data_sets = read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # subspample data
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
        # Ensure that tf will be deterministic
        seed=FLAGS.seed
        if not FLAGS.same_seed:
            seed+=FLAGS.procid
        tf.set_random_seed(seed)

        # Generate placeholders for the images and labels.
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        dropout_rate = tf.placeholder(tf.float32)

        # Build a Graph that computes predictions from the inference model.
        logits = model.inference(images_placeholder,dropout_rate)

        # Add to the Graph the Ops for loss calculation.
        loss = model.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = model.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(local_log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size,FLAGS.fake_data)
            feed_dict = {
                    images_placeholder: images_feed,
                    labels_placeholder: labels_feed,
                    dropout_rate: 0.4,
            }

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(local_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                print('Training Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.train,FLAGS)

                print('Validation Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.validation,FLAGS)

                print('Test Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.test,FLAGS)

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
            '--log_dir',
            type=str,
            default='log',
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)