from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.slim.alexnet import *
from models.common_images import *

def inference(input,datainfo,weight_decay=0.0,is_training=True):
    arg_scope = alexnet_v2_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits,_ = alexnet_v2(input, datainfo.NUM_CLASSES, is_training=is_training)
        return logits
