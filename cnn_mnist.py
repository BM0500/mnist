import tensorflow as tf
import numpy as np

#下载并安装mnist手写数字识别库（55000*28*28）
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data',one_hot=True)
