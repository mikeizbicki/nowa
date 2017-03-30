#!/usr/bin/python

from __future__ import print_function

import argparse
from collections import defaultdict
import importlib
import os.path
import os
import re
import sys

import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from common import *

################################################################################

def main(_):
    # process command line args
    model = importlib.import_module("models.%s" % FLAGS.model)
    np.random.seed(FLAGS.seed)

    results=dict()

    ########################################
    print('preparing data')
    data_sets = read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    validn=FLAGS.validn
    if validn==0:
        validn=data_sets.validation._num_examples
    data_sets.validation._images=data_sets.train._images[0:validn]
    data_sets.validation._labels=data_sets.train._labels[0:validn]
    data_sets.validation._num_examples=validn

    ######################################## 
    print('creating local model graph')
    graph_local = tf.Graph()
    with graph_local.as_default(): 

        # Ensure that tf will be deterministic
        tf.set_random_seed(FLAGS.seed)

        # create the graph
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        dropout_rate = tf.placeholder(tf.float32)
        logits = model.inference(images_placeholder,dropout_rate)
        loss = model.loss(logits, labels_placeholder)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(logits, labels_placeholder)

    ######################################## 
    print('loading local models')
    with graph_local.as_default(): 
        tensorlists=defaultdict(list)
        saver = tf.train.Saver()
        sess_local = dict()
        for procid in range(FLAGS.maxproc):
            sess_local[procid]=tf.Session()

            modeldir=os.path.join(
                FLAGS.log_dir_in,
                '%d-%s.%s-%d-%d'%(FLAGS.seed,FLAGS.same_seed,FLAGS.model,FLAGS.numproc,procid)
                )
            maxitr=max(map(int,sum(
                    map(
                        lambda x:re.findall('\d+',x),
                        sum(map(lambda x:re.findall(r'model.ckpt.\d+',x), os.listdir(modeldir)),[])
                        ),
                    [],
                    )))
            print('  %s --- %d'%(modeldir,maxitr))
            saver.restore(sess_local[procid], os.path.join(modeldir,'model.ckpt-%d'%maxitr))

    results['local'] = do_eval(sess_local[0],eval_correct,data_sets.test,FLAGS)

    ######################################## 
    print('creating average model')
    augtensors=dict()
    with graph_local.as_default(): 
        sess_ave = tf.Session()
        for v in tf.trainable_variables():
            print('  %s:'%v.name)
            augtensors[v.name]=np.stack([tf.convert_to_tensor(v).eval(session=sess_local[i]) for i in sess_local.keys()],axis=0)
            sess_ave.run(tf.assign(v,augtensors[v.name].mean(axis=0)))

        results['ave']=do_eval(sess_ave,eval_correct,data_sets.test,FLAGS)

    ######################################## 

    def mk_nowa_graph(deep):
        graph_nowa = tf.Graph()
        with graph_nowa.as_default():
            alpha_W=None
            alpha_bias=None
            if not deep:
                alpha_W=tf.Variable(
                    tf.ones([FLAGS.maxproc])/FLAGS.maxproc,
                    name='alpha_W'
                    )
                alpha_bias=tf.Variable(
                    tf.zeros([1]),
                    name='alpha_bias'
                    )

            vars = graph_local.get_collection('trainable_variables')
            for v in vars:
                vname=v.name[:v.name.index(':')]
                if deep:
                    alpha_W=tf.Variable(
                        tf.ones([FLAGS.maxproc])/FLAGS.maxproc,
                        name='alpha_W/'+vname
                        )
                    alpha_bias=tf.Variable(
                        tf.zeros([1]),
                        name='alpha_bias/'+vname
                        )
                v2=tf.Variable(
                    augtensors[v.name],
                    False,
                    name='nowa/'+vname
                    )
                tf.identity(tf.tensordot(alpha_W,v2,1)+alpha_bias,name=vname)
                    
            nowa_images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
            nowa_labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
            nowa_dropout_rate = tf.placeholder(tf.float32)
#
            nowa_logits=tf.contrib.copy_graph.copy_op_to_graph(logits,graph_nowa,vars)
            #nowa_logits = model.inference(nowa_images_placeholder,nowa_dropout_rate)
            nowa_loss = model.loss(nowa_logits, nowa_labels_placeholder)
            nowa_train_op = model.training(nowa_loss, FLAGS.learning_rate)
            nowa_eval_correct = model.evaluation(nowa_logits, nowa_labels_placeholder)

            sess_nowa = tf.Session()
            trainmodel(FLAGS,sess_nowa,data_sets.validation,data_sets.test,nowa_train_op,nowa_eval_correct)
            if not deep:
                results['nowa']=do_eval(sess_nowa,nowa_eval_correct,data_sets.test,FLAGS)
            else:
                results['dnowa']=do_eval(sess_nowa,nowa_eval_correct,data_sets.test,FLAGS)
        
        return graph_nowa

    print('creating nowa model')
    mk_nowa_graph(False)

    print('creating dnowa model')
    mk_nowa_graph(True)

    ######################################## 
    print('writing results to disk')

    with open("results.txt", "a") as myfile:
        print(
            ' ', FLAGS.seed,
            ' ', FLAGS.same_seed,
            ' ', FLAGS.numproc,
            ' ', FLAGS.maxproc,
            ' ', FLAGS.validn,
            ' ', results['local'],
            ' ', results['ave'],
            ' ', results['nowa'],
            ' ', results['dnowa'],
            file=myfile
            )

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
            '--log_dir_in',
            type=str,
            default='log/local',
            help='Directory to put the log data.'
    )
    parser.add_argument(
            '--log_dir_out',
            type=str,
            default='log/merge',
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
            '--maxproc',
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
    parser.add_argument(
            '--same_seed',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--validn',
            type=int,
            default=0
            )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
