''' Derived from https://www.tensorflow.org/get_started/mnist/pros
'''
import tensorflow as tf
from common import *

def inference(images,_):
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(images,W) + b
    return y
