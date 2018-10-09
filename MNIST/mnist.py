#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-14 09:36:34
# @Author  : WangGuo (guo_wang_113@163.com)
# @Link    : ${link}
# @Version : $Id$


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

max_step = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例

data_dir = 'logs/'   # 样本数据存储的路径
log_dir = 'logs/'    # 输出日志保存的路径

#mnist = input_data.read_data_sets(data_dir,one_hot=True)
mnist=input_data.read_data_sets('data/',one_hot=True)
#创建tensorflow的默认会话：
sess = tf.InteractiveSession()
#创建输入数据的占位符，分别创建特征数据x，标签数据y_ 
#在tf.placeholder()函数中传入了3个参数，第一个是定义数据类型为float32；
#第二个是数据的大小，特征数据是大小784的向量，标签数据是大小为10的向量，
#None表示不定死大小，到时候可以传入任何数量的样本；第3个参数是这个占位符的名称。
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  
# 使用tf.summary.image保存图像信息 
# 特征数据其实就是图像的像素数据拉升成一个1*784的向量，现在如果想在tensorboard上
# 还原出输入的特征数据对应的图片，就需要将拉升的向量转变成28 * 28 * 1的原始像素了，
# 于是可以用tf.reshape()直接重新调整特征数据的维度： 

# 将输入的数据转换成[28 * 28 * 1]的shape，存储成另一个tensor，命名为image_shaped_input。 
# 为了能使图片在tensorbord上展示出来，使用tf.summary.image将图片数据汇总给tensorbord。 
# tf.summary.image（）中传入的第一个参数是命名，第二个是图片数据，第三个是最多展示的张数，此处为10张
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 100)

#创建初始化权重w的方法，生成大小等于传入的shape参数，标准差为0.1,正态分布的随机数，并且将它转换成tensorflow中的variable返回。
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#创建初始换偏执项b的方法，生成大小为传入参数shape的常数0.1，并将其转换成tensorflow的variable并返回
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#收录每次的参数信息
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):

		# 计算参数的均值，并使用tf.summary.scaler记录
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)

		# 计算参数的标准差
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			# 使用tf.summary.scaler记录记录下标准差，最大值，最小值
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			# 用直方图记录参数的分布
			tf.summary.histogram('histogram', var)
#构建神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # 设置命名空间
    with tf.name_scope(layer_name):
      # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
			variable_summaries(weights)
		# 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
		with tf.name_scope('biases'):
			biases = bias_variable([output_dim])
			variable_summaries(biases)
		# 执行wx+b的线性计算，并且用直方图记录下来
		with tf.name_scope('linear_compute'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			tf.summary.histogram('linear', preactivate)
		# 将线性输出经过激励函数，并将输出也用直方图记录下来
			activations = act(preactivate, name='activation')
			tf.summary.histogram('activations', activations)

		# 返回激励层的最终输出
		return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)
	tf.summary.scalar('dropout_keep_probability', keep_prob)
	dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

#损失函数
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      # 计算所有样本交叉熵损失的均值
	    cross_entropy = tf.reduce_mean(diff)

tf.summary.scalar('loss', cross_entropy)

#使用AdamOptimizer优化器训练模型，最小化交叉熵损失
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)

#计算准确率,并用tf.summary.scalar记录准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      # 求均值即为准确率
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# summaries合并；合并summary operation, 运行初始化变量
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
		xs, ys = mnist.train.next_batch(100)
		k = dropout
    else:
		xs, ys = mnist.test.images, mnist.test.labels
		k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

#训练
# 每隔10步，就进行一次merge, 并打印一次测试数据集的准确率，然后将测试数据集的各种summary信息写进日志中。 
# 每隔100步，记录原信息 
# 其他每一步时都记录下训练集的summary信息并写到日志中。
for i in range(max_step):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
		summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
		test_writer.add_summary(summary, i)
		print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        if i % 100 == 99:  # Record execution stats
	        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	        run_metadata = tf.RunMetadata()
	        summary, _ = sess.run([merged, train_step],
	                              feed_dict=feed_dict(True),
	                              options=run_options,
	                              run_metadata=run_metadata)
	        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
	        train_writer.add_summary(summary, i)
	        print('Adding run metadata for', i)
        else:  # Record a summary
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()