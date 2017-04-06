#!/usr/bin/env python2.7

from __future__ import print_function

import argparse
from collections import defaultdict
import errno
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

    ########################################
    module_dataset='models.%s.common'%FLAGS.dataset
    print('loading %s'%module_dataset)
    datainfo=importlib.import_module(module_dataset)

    module_model='models.%s.%s'%(FLAGS.dataset,FLAGS.model)
    print('loading %s'%module_model)
    model=importlib.import_module(module_model)

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

        count_params()

    ########################################
    print('loading local models')
    with graph_local.as_default():
        tensorlists=defaultdict(list)
        saver = tf.train.Saver()
        sess_local = dict()
        for procid in range(0,FLAGS.maxproc):
            modeldir=os.path.join(
                FLAGS.log_dir_in,
                '%s-%s.%d-%1.2f-%s.%d-%d'%(FLAGS.dataset,FLAGS.model,FLAGS.seed,FLAGS.induced_bias,FLAGS.same_seed,FLAGS.numproc,procid)
                )
            maxitr=max(map(int,sum(
                    map(
                        lambda x:re.findall('\d+',x),
                        sum(map(lambda x:re.findall(r'model.ckpt.\d+',x), os.listdir(modeldir)),[])
                        ),
                    [],
                    )))
            print('  %s --- %d'%(modeldir,maxitr))
            sess_local[procid]=tf.Session()
            saver.restore(sess_local[procid], os.path.join(modeldir,'model.ckpt-%d'%maxitr))

    ########################################
    print('creating augtensors')
    augtensors=dict()
    with graph_local.as_default():
        for v in tf.trainable_variables():
            print('  %s:'%v.name)
            augtensors[v.name]=np.stack([tf.convert_to_tensor(v).eval(session=sess_local[i]) for i in sess_local.keys()],axis=0)

    ########################################
    print('closing local sessions')
    for i in sess_local.keys():
        sess_local[i].close()

    ########################################
    if FLAGS.ave:
        print('creating average model')
        with graph_local.as_default():
            sess_ave = tf.Session()
            for v in tf.trainable_variables():
                sess_ave.run(tf.assign(v,augtensors[v.name].mean(axis=0)))
            results['ave']=[do_eval(sess_ave,eval_correct,data_sets.test,FLAGS)]

    ########################################
    print('creating virtual procs')
    with graph_local.as_default():
        for v in tf.trainable_variables():
            print('  %s:'%v.name)
            x=np.random.standard_normal([FLAGS.virtual_procs]+v.get_shape().as_list()).astype(np.float32)
            augtensors[v.name]=np.append(augtensors[v.name],x,axis=0)
    totproc=FLAGS.maxproc+FLAGS.virtual_procs

    ########################################
    def mk_ops_and_train():
        nowa_images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,model.IMAGE_PIXELS))
        nowa_labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        nowa_dropout_rate = tf.placeholder(tf.float32)
#
        nowa_logits=tf.contrib.copy_graph.copy_op_to_graph(logits,tf.get_default_graph(),vars)
        #nowa_logits = model.inference(nowa_images_placeholder,nowa_dropout_rate)
        nowa_loss = model.loss(nowa_logits, nowa_labels_placeholder)
        nowa_train_op = model.training(nowa_loss, FLAGS.learning_rate)
        nowa_eval_correct = model.evaluation(nowa_logits, nowa_labels_placeholder)

        sess = tf.Session()
        return trainmodel(FLAGS,sess,data_sets.validation,data_sets.test,nowa_train_op,nowa_eval_correct,augtensors)
        #return do_eval(sess,nowa_eval_correct,data_sets.test,FLAGS)

    ####################
    if FLAGS.owa:
        print('creating owa model')
        with tf.Graph().as_default():
            with tf.name_scope('owa') as scope_owa:
                alpha_weights=tf.Variable(
                    tf.ones([totproc])/totproc,
                    name='alpha_weights'
                    )
                alpha_bias=tf.Variable(
                    tf.zeros([1]),
                    name='alpha_bias'
                    )
            vars = graph_local.get_collection('trainable_variables')
            for v in vars:
                vname=v.name[:v.name.index(':')]
                with tf.name_scope(scope_owa):
                    with tf.name_scope(vname):
                        placeholder=tf.placeholder(
                            tf.float32,
                            augtensors[v.name].shape,
                            name='placeholder'
                            )
                        v2=tf.Variable(
                            placeholder,
                            False,
                            name='augtensor'
                            )
                        res=tf.add(
                            tf.tensordot(alpha_weights,v2,1),
                            tf.ones(v.get_shape())*alpha_bias,
                            name=vname
                            )
                tf.identity(res,name=vname)
            count_params()
            results['owa']=mk_ops_and_train()

    ####################
    if FLAGS.dowa:
        print('creating dowa model')
        with tf.Graph().as_default():
            with tf.name_scope('owa') as scope_owa:
                pass
            vars = graph_local.get_collection('trainable_variables')
            for v in vars:
                vname=v.name[:v.name.index(':')]
                with tf.name_scope(scope_owa):
                    with tf.name_scope(vname):
                        alpha_weights=tf.Variable(
                            tf.ones([totproc])/totproc,
                            name='alpha_weights'
                            )
                        alpha_bias=tf.Variable(
                            tf.zeros([1]),
                            name='alpha_bias'
                            )
                        placeholder=tf.placeholder(
                            tf.float32,
                            augtensors[v.name].shape,
                            name='placeholder'
                            )
                        v2=tf.Variable(
                            placeholder,
                            False,
                            name='augtensor'
                            )
                        res=tf.add(
                            tf.tensordot(alpha_weights,v2,1),
                            tf.multiply(
                                tf.ones(v.get_shape()),
                                alpha_bias,
                                ),
                            name=vname
                            )
                tf.identity(res,name=vname)
            count_params()
            results['dowa']=mk_ops_and_train()

    ####################
    if FLAGS.anowa:
        print('creating anowa model')
        with tf.Graph().as_default():
            vars = graph_local.get_collection('trainable_variables')
            for v in vars:
                vname=v.name[:v.name.index(':')]

                hidden_units=1
                alpha1_weights=tf.Variable(
                    tf.add(
                        tf.ones([hidden_units,totproc])/totproc,
                        tf.truncated_normal([hidden_units,totproc],stddev=0.1)
                        ),
                    name=vname+'/alpha1/weights'
                    )
                alpha1_bias=tf.Variable(
                    tf.zeros([hidden_units]),
                    name=vname+'/alpha1/bias'
                    )
                v2=tf.Variable(
                    augtensors[v.name],
                    False,
                    name=vname+'/augtensor'
                    )
                hidden1=tf.nn.relu(tf.add(
                    tf.tensordot(alpha1_weights,v2,[[1],[0]]),
                    tf.tensordot(
                        tf.expand_dims(alpha1_bias,0),
                        tf.expand_dims(tf.ones(v.get_shape()),0),
                        [[0],[0]]
                        ),
                    name=vname+'/hidden1'
                    ))

                alpha2_weights=tf.Variable(
                    tf.add(
                        tf.ones([hidden_units])/hidden_units,
                        tf.truncated_normal([hidden_units],stddev=0.1)
                        ),
                    name=vname+'/alpha2/weights'
                    )
                tf.tensordot(hidden1,alpha2_weights,[[0],[0]],name=vname)
                #alpha2_bias=tf.Variable(
                    #tf.zeros([1]),
                    #name=vname+'/alpha2/bias'
                    #)
                #tf.add(
                    #tf.tensordot(hidden1,alpha2_weights,[[0],[0]]),
                    #tf.tensordot(
                        #tf.expand_dims(alpha1_bias,0),
                        #tf.expand_dims(tf.ones(v.get_shape()),0),
                        #[[0],[0]]
                        #),
                    #name=vname
                    #)
            count_params()
            #results['dnowa']=mk_ops_and_train()

    ########################################
    print('writing results to disk')

    try:
        os.makedirs('results')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    for method in results.keys():
        if FLAGS.same_seed:
            same_seed='same'
        else:
            same_seed='diff'
        with open('results/%s-%s-%s'%(FLAGS.model,method,same_seed), 'a') as myfile:
            print(
                ' ', FLAGS.seed,
                ' ', FLAGS.learning_rate,
                ' ', FLAGS.max_steps,
                ' ', FLAGS.batch_size,
                ' ', FLAGS.numproc,
                ' ', FLAGS.maxproc,
                ' ', FLAGS.virtual_procs,
                ' ', FLAGS.validn,
                ' ', 'RESULTS',
                ' ', ' '.join(map(str,results[method])),
                file=myfile
                )

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.001,
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
            '--dataset',
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
    parser.add_argument(
            '--virtual_procs',
            type=int,
            default=0
            )
    parser.add_argument(
            '--ave',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--owa',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--dowa',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--anowa',
            default=False,
            action='store_true'
            )
    parser.add_argument(
            '--induced_bias',
            type=float,
            default=0
            )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.procid=FLAGS.maxproc
    if unparsed:
        print('unparsed arguments: ',unparsed)
        exit(1)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
