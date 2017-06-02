#!/usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import *
from parser import *

################################################################################

if __name__ == '__main__':

    FLAGS=parseArgs(require_model=False)

    # process command line args
    if FLAGS.numproc <= FLAGS.procid or FLAGS.procid < 0:
        print("procid/numproc combination invalid", file=sys.stderr)
        sys.exit(1)

    # load data

    for example in tf.python_io.tf_record_iterator("data/mnist/train.tfrecords"):
        print('example=',example)
        #result = tf.train.Example.FromString(example)
