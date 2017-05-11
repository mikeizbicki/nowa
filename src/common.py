import argparse
import importlib
import os.path
import re
import sys
import time

import tensorflow as tf

################################################################################

def count_params():
    tot=0
    for v in tf.trainable_variables():
        name=v.name[:v.name.index(':')]
        shape=v.get_shape()
        params=reduce(lambda x,y: x*y,shape)
        tot=tot+params.value
        print('  [%s] %d = %s'%(name,params,shape))
    print('  total %s'%tot)
    return tot

def flags2logdir(FLAGS):
    return os.path.join(
        FLAGS.log_dir_out,
        '%s-%s.%d-%1.2f-%s.%d-%d'%(
            FLAGS.dataset,
            FLAGS.model,
            FLAGS.seed,
            FLAGS.induced_bias,
            FLAGS.same_seed,
            FLAGS.numproc,
            FLAGS.procid
            )
        )

################################################################################

def do_eval(sess,eval_correct,data_set,FLAGS):
    """Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
        feed_dict = {
                'Placeholder:0': images_feed,
                'Placeholder_1:0': labels_feed,
                'Placeholder_2:0': 1,
        }
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    if num_examples>0:
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))
    return precision

################################################################################

def trainmodel(FLAGS,train_set,test_set,train_op,metric,augtensors):
    # ensure reproducibility
    tf.set_random_seed(0)
    #tf.reset_default_graph()

    augtensors2=dict()
    for k in augtensors.keys():
        name=k[:k.index(':')]
        index=k[k.index(':'):]
        augtensors2['owa/'+name+'/placeholder'+index]=augtensors[k]

    # create session
    if FLAGS.maxcpu==0:
        session_conf=tf.ConfigProto(
            )
    else:
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
            )
    sess = tf.Session(config=session_conf)
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            ),
        feed_dict=augtensors2
        )

    # prepare logging
    if FLAGS.allcheckpoints:
        saver = tf.train.Saver(max_to_keep=None)
    else:
        saver = tf.train.Saver(max_to_keep=1)

    local_log_dir=flags2logdir(FLAGS)
    if tf.gfile.Exists(local_log_dir):
        tf.gfile.DeleteRecursively(local_log_dir)
        #checkpoints=sorted(list(set(map(int,sum(
            #map(
                #lambda x:re.findall('\d+',x),
                #sum(map(lambda x:re.findall(r'model.ckpt.\d+',x), os.listdir(local_log_dir)),[])
                #),
            #[],
            #)))))
        #checkpoint=max(checkpoints)
        #saver.restore(sess, os.path.join(local_log_dir,'model.ckpt-%d'%checkpoint))
    #else:
    checkpoint=0
    tf.gfile.MakeDirs(local_log_dir)

    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(local_log_dir, sess.graph)

    # training loop
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    evals=[]
    tot=0
    start_time = time.time()
    for step in map(lambda x:x+checkpoint,xrange(FLAGS.max_steps)):

        _, metric_value = sess.run([train_op, metric])
        tot+=metric_value

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            print('  step %d: metric = %.2f (%.3f sec)' % (step, tot, duration))
            tot=0
            start_time = time.time()
            summary_str = sess.run(summary)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(local_log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

    sess.close()
    return evals
