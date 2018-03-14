import os
import tensorflow as tf
import numpy as np

# 关闭warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = np.array([[1, 2, 3], [4, 5, 6]])
a_shape = tf.shape(a)
b = a.reshape(-1, 2)  # 转成2列的数据

# 定义自适应匹配的shape
bias = tf.get_variable("b", shape=[10, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
bias_shape = bias.shape

with tf.Session() as sess:
    print(sess.run(a_shape))
    print(sess.run(b))
    print(sess.run(bias_shape))


# 完美诠释了字符/词语/句子之间的表示
arr_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = tf.reshape(arr_a, [-1, 6])
with tf.Session() as sess:
    print(sess.run(b))
