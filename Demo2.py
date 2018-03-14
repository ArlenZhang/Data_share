import os
import tensorflow as tf
# 关闭warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================== operation 1
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('../graphs/Demo2', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('graphs', sess.graph)
    print(sess.run(x))
    print(x.eval())  # 意味着
writer.close()
'''
Go to terminal run:
    tensorboard --logdir="graphs/Demo2" --port 6006
    open browser  http://ArlenIAC:6006
'''
# =============================== operation 2 Constants
a = tf.constant([1, 2], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")
x = tf.multiply(a, b, name="mul")
with tf.Session() as sess:
    print("变量可以是数组，乘法对应位相乘~")
    print(sess.run(x))

# =============================== operation 3 specific value
shape = [2, 3]
zeros_arr = tf.zeros(shape, tf.int32)
ones_arr = tf.ones(shape, dtype=tf.float32, name=None)
# ones_like_arr = tf.ones_like(input_tensor, dtype=None, name=None, optimal=None)

# create a tensor filled with a scalar value.
scaled_arr = tf.fill(shape, 8)

# tf.lin_space(start, stop, num, name=None)
# tf.lin_space(10.0, 13.0, 4) ==> [10. 11. 12. 13.]

# tf.range(start, limit=None, delta=1, dtype=None, name='range')
# tf.range(3, 18, 3) ==> [3 6 9 12 15]
# tf.range(5) ==> [0 1 2 3 4]

# read slides
