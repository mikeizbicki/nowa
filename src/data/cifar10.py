from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tflearn.datasets import cifar10
from tflearn.data_utils import to_categorical

NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def loaddata(FLAGS):
    print('loading dataset cifar10')
    datadir_dataset=os.path.join(FLAGS.input_data_dir,FLAGS.dataset)
    (X, Y), (X_test, Y_test) = cifar10.load_data(dirname=datadir_dataset)
    Y = to_categorical(Y, 10)
    Y_test = to_categorical(Y_test, 10)

    np.random.seed(FLAGS.seed)
    np.random.shuffle(X)
    np.random.seed(FLAGS.seed)
    np.random.shuffle(Y)
    #np.random.seed(FLAGS.seed+1)
    #np.random.shuffle(datasets.validation.data)
    #np.random.seed(FLAGS.seed+1)
    #np.random.shuffle(datasets.validation.target)

    trainnm = X.shape[0]
    trainn  = int(trainnm / FLAGS.numproc)
    datasets_train=DataSet(
        data=X[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)],
        target=Y[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)]
        )
    #datasets.train.data=datasets.train.data[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)]
    #datasets.train.target=datasets.train.target[trainn*FLAGS.procid:trainn*(FLAGS.procid+1)]
    #datasets.train._num_examples=trainn
    print('  train n=%d'%trainn)

    #validationnm = datasets.validation.data.shape[0]
    #validationn  = int(validationnm / FLAGS.numproc)
    datasets_validation=base.Dataset([],[])
    #datasets.validation.data=datasets.validation.data[validationn*FLAGS.procid:validationn*(FLAGS.procid+1)]
    #datasets.validation.target=datasets.validation.target[validationn*FLAGS.procid:validationn*(FLAGS.procid+1)]
    #datasets.validation._num_examples=validationn

    #print('  valid n=%d'%validationn)

    datasets_test=base.Dataset(
        data=X_test,
        target=Y_test
        )

    #return (datasets.train,datasets.validation,datasets.test)
    return (datasets_train,datasets_validation,datasets_test)

########################################

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
