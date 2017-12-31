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
from sklearn import cross_validation

import tensorflow as tf


# import sklearn.metrics
# import skmultilearn

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
LOG_PATH = 'logs2'
PIXEL_DEPTH = 255

xdata = 30
ydata = 30
# xlabel = 1
# ylabel = 1

xd = ((xdata-4)//2-4)//2
yd = ((ydata-4)//2-4)//2

IMAGE_SIZE = 30
NUM_CHANNELS = 3
# NUM_CHANNELS = 1
# NUM_LABELS = 10
NUM_LABELS = 2
VALIDATION_SIZE = 4000
SEED = 66478
# SEED = 12345
BATCH_SIZE = 100
NUM_EPOCHS = 100
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 100

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS
FilePath = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/precision/3-gram/Batch100/E100.txt"


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
    """Return the error rate based on dense predictions and sparse labels.
    返回基于标准标签和预测标签的错误率

    numpy.argmax(predictions, 1) ---> predictions每一行最大值的下标数组
    labels 是每张图片的标准标签
    numpy.argmax(predictions, 1) == labels，如果两个值相同，返回true，否则返回false
    numpy.sum(numpy.argmax(predictions, 1) == labels) 统计sum里面数组true的个数
    predictions.shape[0] ---> 读取矩阵第一维长度，即矩阵行数
    相除的结果是正确率， 100-*** 的结果是错误率
    """
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
    start_time = time.time()
    train_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/data_train.npy').\
        astype(dtype=numpy.float32)
    train_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/label_train.npy').\
        astype(dtype=numpy.int64)
    test_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/data_test.npy').\
        astype(dtype=numpy.float32)
    test_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/label_test.npy').\
        astype(dtype=numpy.int64)
    elapsed_time = time.time() - start_time
    print('read data time: ', elapsed_time, 's')

        # Generate a validation set.
        # 0-5000是验证集，5000-60000是训练集
    # train_data, validation_data, train_labels, validation_labels = \
    #     cross_validation.train_test_split(train_data, train_labels, test_size=0.2, random_state=SEED)
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]  # 读取训练集大小

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    # 训练样本数据和标签用于图。这些占位符节点在每一步训练的时候使用 {feed_dict} 参数
    # 传参数给训练数据并被调用
    #
    # BATCH_SIZE = 64  IMAGE_SIZE = 28  NUM_CHANNELS = 1  EVAL_BATCH_SIZE = 64
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, xdata, ydata, NUM_CHANNELS), name='train_data')
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,), name='train_label')
    eval_data = tf.placeholder(
        tf.float32,
        shape=(EVAL_BATCH_SIZE, xdata, ydata, NUM_CHANNELS), name='eval_data')

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    # 下面的这些变量是所有需要训练的权重参数。当我们调用{tf.initialize_all_variables().run()}时，
    # 这些参数会被分配初始值。
    # truncated_normal 返回一个tensor其中的元素服从截断正态分布
    # NUM_CHANNELS = 1   SEED = 66478   NUM_LABELS = 10
    with tf.name_scope('conv1'):
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, featureMap_out 32.
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv1_biases = tf.Variable(tf.zeros([32]), name='biases')
    with tf.name_scope('conv2'):
        conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=0.1,
                                seed=SEED), name='weights')
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
    print(xd * yd * 64)
    with tf.name_scope('hidden'):
        fc1_weights = tf.Variable(  # fully connected, depth (input+output)/2.
            tf.truncated_normal(
                [xd * yd * 64, 161],
                stddev=0.1,
                seed=SEED), name='weights')
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[161]), name='biases')
    with tf.name_scope('softmax_linear'):
        fc2_weights = tf.Variable(
            tf.truncated_normal([161, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED), name='weights')
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='biases')

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    # 我们会复制这个模型结构给训练子图和评估子图，同时共享训练参数
    # 模型返回的是训练模型在训练数据下的标签或者测试模型在测试数据下的标签
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # 2维卷积，使用"SAME "padding参数，输出特征层和输入有相同的大小。
        # {Stride} 是一个4维数组，它的维度匹配数据层：[image index, y, x, depth]。
        with tf.name_scope('conv1'):
            conv = tf.nn.conv2d(data,
                                conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='conv')
            # Bias and rectified linear non-linearity. 偏置，线性修正函数relu作为激活函数
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name='relu')
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            # 池化层。核大小{ksize}也跟随数据层。
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pool')
        with tf.name_scope('conv2'):
            conv = tf.nn.conv2d(pool,
                                conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases), name='relu')
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pool')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        # reshape特征层立方体形状变为一个2维的形状用于全连接层。
        with tf.name_scope('reshape'):
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(
                pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]], name='shape')
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        # 全连接层。注意‘+’操作会自动传播偏置。
        with tf.name_scope('hidden'):
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # 只在 “训练” 的时候，增加一个dropout丢包操作。
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        with tf.name_scope('softmax_linear'):
            logits = tf.matmul(hidden, fc2_weights) + fc2_biases
        return logits

    # Training computation: logits + cross-entropy loss.
    # 训练计算：逻辑斯特 + 交叉熵损失函数。
    # 调用前一步定义的模型，后一个参数设置为True，因为是训练过程，允许dropout操作
    #
    # logits和训练标签节点train_labels_node的交叉熵的最小均方误差作为损失函数。
    #
    # 因为 logits = model(train_data_node, True) 最终返回的是训练模型在训练数据下的标签
    # 所以它可以和标准的标签数据 train_labels_node 进行误差的比较和对比。
    # print(logits)
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, train_labels_node), name='loss')
    # 采用hamming loss，主要针对多标签分类
    # loss = sklearn.metrics.hamming_loss(logits, train_labels_node)

    # L2 regularization for the fully connected parameters.
    # 全连接层的 L2 正则化参数
    with tf.name_scope('L2Loss'):
        regularizers = (tf.nn.l2_loss(fc1_weights, name='hidden_weights') +
                        tf.nn.l2_loss(fc1_biases, name='hidden_biases') +
                        tf.nn.l2_loss(fc2_weights, name='softmax_linear_weights') +
                        tf.nn.l2_loss(fc2_biases, name='softmax_linear_biases'))
    # Add the regularization term to the loss.
    # 损失函数值 在原来的基础上加上损失函数的值，防止过拟合或欠拟合等问题
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    # 优化器：设置一个变量，它在每一次批训练的时候增大并且控制学习速率的衰退
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    # 学习速率：每次迭代减小，使用一个指数计划减少，初始值为0.01
#    learning_rate = tf.train.exponential_decay(
#        0.01,  # Base learning rate.
#        batch * BATCH_SIZE,  # Current index into the dataset.
#        train_size,  # Decay step.
#        0.95,  # Decay rate.
#        staircase=True,
#        name='learning_rate')
    # Use simple momentum for the optimization.
    # 使用简单的 momentum 作为训练的优化器
#    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, name='MomentumOptimizer').minimize(loss, global_step=batch)
	optimizer = tf.train.AdamOptimizer_init_(learning_rate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 0.00001, use_locking=False, name='Adam')

    # Predictions for the current training minibatch.
    # 当前批训练的训练数据的预测值
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    # 测试集和验证集的预测值，我们计算的比较少
    eval_prediction = tf.nn.softmax(model(eval_data))

    # 添加loss和learning_rate为scalar_summary，用于画出训练过程中两个变量的变化过程
    # tf.scalar_summary('xentropy', cross_entropy)
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('Adam', optimizer)
    merged_summary_op = tf.merge_all_summaries()
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(LOG_PATH, sess.graph)

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    # 小实用函数：使用批数据{eval_data}评估一个数据集并且拉取{eval_predictions}的结果
    # 保存到内存并且使其可以在小GPUs上面运行。
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches.
        得到一个数据集的所有预测值，通过在小批训练数据上运行
        """
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
                # print("predictions:", sess.run(predictions))
                # print("eval_predictions:", eval_prediction)
                # print (predictions)
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def eval_in_batches_test(data, sess):
        """Get all predictions for a dataset by running it in small batches.
        得到一个数据集的所有预测值，通过在小批训练数据上运行
        """
        size = data.shape[0]  # 数据集行数
        print(size)
        if size < EVAL_BATCH_SIZE:  # 批训练大小不能超过数据集的大小
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        result = numpy.zeros([1, 1], dtype=numpy.int64)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):  # 0-size之间每次取批训练数据
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
                print("predictions:", predictions[begin:end, :])
                p = numpy.argmax(predictions[begin:end, :], 1)
                p1 = numpy.reshape(p, (EVAL_BATCH_SIZE, 1))
                print("p1:", p1)
                result = numpy.vstack((result, p1))
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        # print("result:", result)
        # file1 = open('/home/att/PycharmProjects/CNN_WB/code/1.txt', 'w')
        # file1.write(result)
        # file1.close()
        numpy.savetxt(FilePath, result, fmt='%.0e')
        return predictions

    # Create a local session to run the training.
    # 创建一个局部会话运行训练过程
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        # 运行所有变量的初始化准备可训练的参数
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Loop through training steps.
        # 训练阶段进行循环   num_epochs = 10   train_size = 60000   BATCH_SIZE = 64
        # int(num_epochs * train_size) // BATCH_SIZE = 60000*10/64 = 9375
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            # 下面这个词典，将批训练数据（numpy数组）映射成tf图中对应赋值的节点
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            # 运行图并且取出一些节点
            _, l, lr, predictions = sess.run(
                [optimizer, loss, train_prediction],
                feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:  # EVAL_FREQUENCY = 100,每批训练100次输出一次信息

                # 运行summaries
                summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))  # error_rate函数计算批训练的错误率
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))  # error_rate函数计算验证集的错误率
                sys.stdout.flush()
        # Finally print the result!
        # 循环结束，打印最终错误率
        test_error = error_rate(eval_in_batches_test(test_data, sess), test_labels)  # error_rate函数计算测试集的错误率
        print('Test error: %.1f%%' % test_error)
        if FLAGS.self_test:
            print('test_error', test_error)
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                test_error,)


if __name__ == '__main__':
    tf.app.run()
