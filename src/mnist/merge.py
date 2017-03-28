#!/usr/bin/python

import argparse
import importlib
import sys

import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

################################################################################

def main(_):
    # process command line args
    model = importlib.import_module("models.%s" % FLAGS.model)

    # load the data
    data_sets = read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    with tf.Graph().as_default():

        # Ensure that tf will be deterministic
        tf.set_random_seed(FLAGS.seed)
        
        # create the graph
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        logits = model.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        loss = model.loss(logits, labels_placeholder)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(logits, labels_placeholder)


        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, "log/mnist-256-15.0/model-256-15.ckpt")

        print('Training Data Eval:')
        do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.train)

        print('Validation Data Eval:')
        do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.validation)

        print('Test Data Eval:')
        do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.test)



    print('done.')

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help='Initial learning rate.'
    )
    parser.add_argument(
            '--max_steps',
            type=int,
            default=2000,
            help='Number of steps to run trainer.'
    )
    parser.add_argument(
            '--hidden1',
            type=int,
            default=128,
            help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
            '--hidden2',
            type=int,
            default=32,
            help='Number of units in hidden layer 2.'
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
            '--seed',
            type=int,
            default=0
            )
    parser.add_argument(
            '--model',
            type=str,
            default=''
            )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
