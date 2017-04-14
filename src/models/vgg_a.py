from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.common_images import *
from models.slim.vgg import *

def inference(input,datainfo,weight_decay=0.0,is_training=True):
    arg_scope = vgg_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits,_ = vgg_a(input, datainfo.NUM_CLASSES, is_training=is_training)
        return logits

