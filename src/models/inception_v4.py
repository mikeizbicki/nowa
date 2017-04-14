from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.common_images import *
from models.slim.inception_v4 import *

def inference(input,datainfo,weight_decay=0.0,is_training=True):
    arg_scope = inception_v4_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits,_ = inception_v4(input, datainfo.NUM_CLASSES, is_training=is_training)
        return logits

