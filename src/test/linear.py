#!/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########################################

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#sess = tf.InteractiveSession()

################################################################################

def inference(images, hidden1_units, hidden2_units):
    # Hidden 1
    with tf.name_scope('hidden1'):
      weights = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                              stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
          name='weights')
      biases = tf.Variable(tf.zeros([hidden1_units]),
                           name='biases')
      hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
      weights = tf.Variable(
          tf.truncated_normal([hidden1_units, hidden2_units],
                              stddev=1.0 / math.sqrt(float(hidden1_units))),
          name='weights')
      biases = tf.Variable(tf.zeros([hidden2_units]),
                           name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
      weights = tf.Variable(
          tf.truncated_normal([hidden2_units, NUM_CLASSES],
                              stddev=1.0 / math.sqrt(float(hidden2_units))),
          name='weights')
      biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
      logits = tf.matmul(hidden2, weights) + biases
    return logits

########################################

def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

########################################

def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

#def buildmodel():
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b
    #return y

#y = buildmodel()
y_ = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

