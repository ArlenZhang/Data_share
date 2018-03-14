"""
    desc: 对矩阵乘法的相关理解
    author: LongYinZ
    date: 2018.1.30
"""
import numpy as np
import tensorflow as tf
arr_a = np.array([1, 2, 3, 4])
arr_b = arr_a
# section 1 直接相乘就和np的multiply是对应位乘积
print(arr_a*arr_b)
print(np.multiply(arr_a, arr_b))

# section 2 dot转成矩阵乘积
print(np.dot(arr_a, arr_b))

# section 3 矩阵和数字相乘
temp = arr_a[:] * 2
print(temp)

print("=====================================================")

# 在tensorflow里面对应操作
const_arr_a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a', dtype=tf.int32)
const_arr_b = tf.constant([[1], [2], [3]], name='b', dtype=tf.int32)
result_a = tf.multiply(arr_a, arr_b)
# 乘法两边必须是matrix而不能是vector，这里对应的是矩阵乘法
result_b = tf.matmul(const_arr_a, const_arr_b)

with tf.Session() as sess:
    print(sess.run(result_a))
    print(sess.run(result_b))



