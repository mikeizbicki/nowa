''' Derived from https://www.tensorflow.org/get_started/mnist/pros
'''
import tensorflow as tf
from models.common_images import *

def inference(images,datainfo):
    x_images=tf.reshape(images,[-1,datainfo.IMAGE_SIZE,datainfo.IMAGE_SIZE,datainfo.IMAGE_COLORS])
    W = tf.Variable(tf.zeros([datainfo.IMAGE_SIZE,datainfo.IMAGE_SIZE,datainfo.IMAGE_COLORS,datainfo.NUM_CLASSES]))
    b = tf.Variable(tf.zeros([datainfo.NUM_CLASSES]))
    y = tf.tensordot(x_images,W,[[1,2,3],[0,1,2]]) + b

    #W = tf.Variable(tf.zeros([datainfo.IMAGE_PIXELS,datainfo.NUM_CLASSES]))
    #b = tf.Variable(tf.zeros([datainfo.NUM_CLASSES]))
    #y = tf.matmul(images,W) + b
    return y
