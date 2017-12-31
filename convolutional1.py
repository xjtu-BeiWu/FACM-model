# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.metrics import hamming_loss

import hammingloss as hl
import errorRate

WORK_DIRECTORY = 'data'
LOG_PATH = '/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/middle_result/log/1'
GRAMPATH = '/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/middle_result/3-gram.txt'
PIXEL_DEPTH = 255

NUM_N = 3  # 对应的待卷积词数
# NUM_K = 100
xdata = 200
ydata = 50
beta = 6
epsi = 26

# NUM_POOL = 350
# NUM_OUT = xdata-(NUM_N-1)-(NUM_POOL-1)
NUM_OUT = 28
NUM_HIDDEN = 7

# IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
VALIDATION_SIZE = 640
SEED = 66478
# SEED = 12345
BATCH_SIZE = 150
NUM_EPOCHS = 200
EVAL_BATCH_SIZE = 150
EVAL_FREQUENCY = 100

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name,  tf.reduce_min(var))
        tf.histogram_summary(name, var)


def error_rate(predictions, labels):
    # return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])
    # print('--------------------------hamming_loss---------------------------------')
    # print(predictions)
    result = numpy.zeros(shape=(len(predictions), len(predictions[0])), dtype=numpy.uint8)
    for n in range(len(predictions)):
        log = predictions[n]
        res = hl.label_easy(log)
        result[n] = res
    return 100.0 * hamming_loss(labels, result)


def main(argv=None):  # pylint: disable=unused-argument
    # Get the data.

    start_time = time.time()
    train_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/npy/train_data.npy').\
        astype(dtype=numpy.float32)
    train_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/npy/train_labels.npy').\
        astype(dtype=numpy.int64)
    test_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/npy/test_data.npy').\
        astype(dtype=numpy.float32)
    test_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/newnorm/npy/test_labels.npy').\
        astype(dtype=numpy.int64)
    elapsed_time = time.time() - start_time
    print('read data time: ', elapsed_time, 's')

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]  # 读取训练集大小

    # 训练样本数据和标签用于图。设置模型的内部参数：占位符用于外部训练数据输入，变量定义模型的参数
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, xdata, ydata, NUM_CHANNELS), name='train_data')
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10), name='train_label')
    eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, xdata, ydata, NUM_CHANNELS), name='eval_data')

    with tf.name_scope('layer1'):
        with tf.name_scope('weights'):
            conv1_weights = tf.Variable(tf.truncated_normal([NUM_N, 50, NUM_CHANNELS, 30], stddev=0.1, seed=SEED))
            variable_summaries(conv1_weights, 'layer1/weights')
        with tf.name_scope('biases'):
            conv1_biases = tf.Variable(tf.zeros([30]))
            variable_summaries(conv1_biases, 'layer1/biases')

    print("NUM_OUT:", NUM_OUT)
    with tf.name_scope('hidden'):
        with tf.name_scope('weights'):
            fc1_weights = tf.Variable(tf.truncated_normal([NUM_OUT*1*30, NUM_HIDDEN], stddev=0.1, seed=SEED))
            variable_summaries(fc1_weights, 'hidden/weights')
        with tf.name_scope('biases'):
            fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_HIDDEN]))
            variable_summaries(fc1_biases, 'hidden/biases')

    with tf.name_scope('softmax_linear'):
        with tf.name_scope('weights'):
            fc2_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1, seed=SEED))
            variable_summaries(fc2_weights, 'softmax_linear/weights')
        with tf.name_scope('biases'):
            fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
            variable_summaries(fc2_biases, 'softmax_linear/biases')

    def model1(data, train=False):
        with tf.name_scope('layer1'):
            conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name='relu')
            pool = tf.nn.max_pool(relu, ksize=[1, epsi, 1, 1], strides=[1, beta, 1, 1], padding='VALID', name='pool')
        with tf.name_scope('reshape'):
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(
                pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]], name='shape')
        return reshape

    def model2(reshape, train=False):
        # 全连接层。注意‘+’操作会自动传播偏置。
        with tf.name_scope('hidden'):
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # 只在 “训练” 的时候，增加一个dropout丢包操作。
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        with tf.name_scope('softmax_linear'):
            logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        return logits

    logits = model2(model1(train_data_node, True), True)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, train_labels_node), name='loss')

    # L2 regularization for the fully connected parameters.
    with tf.name_scope('L2Loss'):
        regularizers = (tf.nn.l2_loss(fc1_weights, name='hidden_weights') +
                        tf.nn.l2_loss(fc1_biases, name='hidden_biases') +
                        tf.nn.l2_loss(fc2_weights, name='softmax_linear_weights') +
                        tf.nn.l2_loss(fc2_biases, name='softmax_linear_biases'))
    loss += 5e-4 * regularizers
    batch = tf.Variable(0)
#    learning_rate = tf.train.exponential_decay(
#        0.001,  # Base learning rate.
#        batch * BATCH_SIZE,  # Current index into the dataset.
#        train_size,  # Decay step.
#        0.95,  # Decay rate.
#        staircase=True,
#        name='learning_rate')
    with tf.name_scope('train'):
#        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
		optimizer = tf.train.AdamOptimizer_init_(learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 0.00001, use_locking=False, name='Adam')
    train_prediction = tf.nn.softmax(logits)

    reshape1 = model1(eval_data)
    eval_prediction = tf.nn.softmax(model2(model1(eval_data)))

    # 添加loss和learning_rate为scalar_summary，用于画出训练过程中两个变量的变化过程
    
    tf.scalar_summary('loss', loss)
#    tf.scalar_summary('learning_rate', learning_rate)
    tf.scalar_summary('Adam', optimizer)
    merged_summary_op = tf.merge_all_summaries()
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(LOG_PATH, sess.graph)

    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches. """
        size = data.shape[0]  # 数据集行数
        if size < EVAL_BATCH_SIZE:  # 批训练大小不能超过数据集的大小
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):  # 0-size之间每次取批训练数据
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def eval_in_batches_test(data, sess):
        """Get all predictions for a dataset by running it in small batches. """
        size = data.shape[0]  # 数据集行数
        if size < EVAL_BATCH_SIZE:  # 批训练大小不能超过数据集的大小
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        result = numpy.zeros([1, NUM_OUT*28], dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):  # 0-size之间每次取批训练数据
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                a = sess.run(reshape1, feed_dict={eval_data: data[begin:end, ...]})
                result = numpy.vstack((result, a))
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        numpy.savetxt(GRAMPATH, result)
        return predictions

    # Create a local session to run the training.
    # 创建一个局部会话运行训练过程
    start_time = time.time()
    s_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        # 运行所有变量的初始化准备可训练的参数
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Loop through training steps.
        # 训练阶段进行循环   num_epochs = 10   train_size = 1190   BATCH_SIZE = 5
        # int(num_epochs * train_size) // BATCH_SIZE = 1190*10/5 = 2380
        # print(int(num_epochs * train_size) // BATCH_SIZE)  # 2300
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # print(step)  # 0-2300
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            # print(offset)    # 0-5-10-15-20-...
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, l, lr, predictions = sess.run(
                [optimizer, loss, train_prediction],
                feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:  # EVAL_FREQUENCY = 100,每批训练100次输出一次信息

                # 运行summaries
                summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

                elapsed_time = time.time() - start_time
                start_time = time.time()
                # print(sess.run(logits, feed_dict=feed_dict))  # 5*10的CNN模型训练的浮点数矩阵
                # print(batch_labels)    # 5*10的0-1标签矩阵
                print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: ', l)
                print('learning rate: %.6f' % (lr))

                # print('------------------------')
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                # print('Minibatch precision: %.1f%%' % errorRate.precision_twodim(errorRate.label_twodims(predictions),
                #                                                                  batch_labels))
                # print('Minibatch recall: %.1f%%' % errorRate.recall_twodim(errorRate.label_twodims(predictions),
                #                                                            batch_labels))

                # print('------------------------')
                print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
                # print('Validation precision: %.1f%%' % errorRate.precision_twodim(errorRate.label_twodims(
                #     eval_in_batches(validation_data, sess)), validation_labels))
                # print('Validation recall: %.1f%%' % errorRate.recall_twodim(errorRate.label_twodims(
                #     eval_in_batches(validation_data, sess)), validation_labels))

                sys.stdout.flush()

        # Finally print the result!
        # 循环结束，打印最终错误率
        test_error = error_rate(eval_in_batches_test(test_data, sess), test_labels)  # error_rate函数计算测试集的错误率
        print('Test error: %.1f%%' % test_error)
        # print('Test precision: %.1f%%' % errorRate.precision_twodim(errorRate.label_twodims(
        #     eval_in_batches(test_data, sess)), test_labels))
        # print('Test recall: %.1f%%' % errorRate.recall_twodim(errorRate.label_twodims(
        #     eval_in_batches(test_data, sess)), test_labels))

        if FLAGS.self_test:
            print('test_error', test_error)
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)
    
    elapsed_time = time.time() - s_time
    print('time cost: ', elapsed_time, 's')       


if __name__ == '__main__':
    tf.app.run()
