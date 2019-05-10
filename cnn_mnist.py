import tensorflow as tf
import numpy as np

#下载并安装mnist手写数字识别库（55000*28*28）
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data',one_hot=True)
'''
one_hot 独热码
0 1 2 3 4 5 6 7 8 9
0 1000000000
1 0100000000...
'''

#/255归一化，把灰度区间放到01之间，方便优化器找到其中最小值
input_x = tf.placeholder(tf.float32,[None,28*28]) / 255  #输入
output_y = tf.placeholder(tf.float32,[None,10]) #输出十个数字的标签

input_x_images = tf.reshape(input_x,[-1,28,28,1])  #改变形状之后的输入

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

#搭建卷积神经网络 cnn
#第一层卷积层
conv1 = tf.layers.conv2d(
    inputs=input_x_images,#形状28 28 1
    filters=32,           #过滤器个数，输出depth 32
    kernel_size=[5,5],    #过滤器在二维大小
    strides=1,            #步长 1
    padding='same',       #输出的图像大小不变
    activation=tf.nn.relu #激活函数
)#形状[28,28,32]

#第一层池化层（亚采样层）

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,      #形状[28 28 32]
    pool_size=[2,2],    #过滤器在二维的大小[2,2]
    strides=2           #步长是2
)#性状[14 14 32]

#
conv2 = tf.layers.conv2d(
    inputs= pool1,
    filters = 64,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)#[14 14 64]

#2 池化层
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2
)#[7 7 64]

#平坦化（flat） 降维

flat = tf.reshape(pool2,[-1,7*7*64]) #形状 7 7 64
#1024个神经元全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
