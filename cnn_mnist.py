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

nput_x_images = tf.reshape(input_x,[-1,28,28,3])  #改变形状之后的输入
