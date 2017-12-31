# -*- coding: utf-8 -*-
import time

import numpy

xdata = 200
ydata = 50
trian_num = 54000 # train and validation
test_num = 10000


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    print('Extracting', filename)
    data = numpy.loadtxt(filename)   # 从文件读取数据，存为numpy数组
    data = numpy.frombuffer(data, dtype=numpy.float).astype(numpy.float32)   # 改变数组元素从float变为float32类型，符合CNN输入的要求
    data = data.reshape(num_images, xdata, ydata, 1)  # 所有元素
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    labels = numpy.loadtxt(filename)
    labels = numpy.frombuffer(labels, dtype=float).astype(numpy.int64)
    # print([labels])
    labels = labels.reshape(num_images)  # 标签是10个数据为其标签
    # print([labels])
    return labels


# train_data_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/data2-1.txt"
# train_labels_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/label2.txt"
# test_data_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/data2-1.txt"
# test_labels_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/label2.txt"

train_data_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/baseline/sim_cnn/"
train_labels_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/ForSim/label3_train.txt"
test_data_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/ForSim/3-gram_sim_test.txt"
test_labels_filename = "/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/ForSim/label3_test.txt"

start_time = time.time()
# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, trian_num) # train and validation
train_labels = extract_labels(train_labels_filename, trian_num)
test_data = extract_data(test_data_filename, test_num)
test_labels = extract_labels(test_labels_filename, test_num)
elapsed_time = time.time() - start_time
print(elapsed_time, 's')

start_time = time.time()
# Extract it into numpy arrays.
# train_data = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN1/train_data.npy', train_data)
# train_labels = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN1/train_labels.npy', train_labels)
# test_data = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN1/test_data.npy', test_data)
# test_labels = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN1/test_labels.npy', test_labels)
train_data = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/data_train.npy', train_data)
train_labels = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/label_train.npy', train_labels)
test_data = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/data_test.npy', test_data)
test_labels = numpy.save('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/new/npy/CNN3/label_test.npy', test_labels)
print(train_data)
elapsed_time = time.time() - start_time
print(elapsed_time, 's')


# 加载数据
# start_time = time.time()
# train_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/npy/train_data.npy').astype(dtype=numpy.float32)
# train_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/npy/train_labels.npy').astype(dtype=numpy.int64)
# test_data = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/npy/test_data.npy').astype(dtype=numpy.float32)
# test_labels = numpy.load('/home/ubuntu/anaconda2/envs/tfgpu/workspace/wu/data/npy/test_labels.npy').astype(dtype=numpy.int64)
# print(train_data)
# elapsed_time = time.time() - start_time
# print(elapsed_time, 's')
