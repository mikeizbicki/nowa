''' Derived from https://www.tensorflow.org/get_started/mnist/pros
'''
import tensorflow as tf
from common import *

def inference(images,_):
    #x_images=tf.reshape(images,[100,28,28])
    #W = tf.Variable(tf.zeros([28,28,10]))
    #b = tf.Variable(tf.zeros([10]))
    #y = tf.tensordot(x_images,W,[[1,2],[0,1]]) + b

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(images,W) + b
    return y