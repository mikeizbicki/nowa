import argparse
import importlib
import os.path
import sys
import time

import tensorflow as tf

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

def trainmodel(FLAGS,sess,train_set,test_set,train_op,metric):
    tf.set_random_seed(0)
    #tf.reset_default_graph()

    local_log_dir=FLAGS.log_dir_out
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(local_log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        images_feed, labels_feed = train_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
        feed_dict = {
                'Placeholder:0': images_feed,
                'Placeholder_1:0': labels_feed,
                'Placeholder_2:0': 0.4,
        }

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, metric_value = sess.run([train_op, metric],feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            print('  step %d: metric = %.2f (%.3f sec)' % (step, metric_value, duration))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(local_log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

            #print('Test Data Eval:')
            do_eval(sess,metric,test_set,FLAGS)
