import tensorflow as tf
def add_place_holders():
    # create a placeholder for a vector of 3 elements, type tf.float32
    a = tf.placeholder(tf.float32, shape=[3])
    b = tf.constant([5, 5, 5], tf.float32)
    c = a + b   # tf.add(a, b)的简写
    with tf.Session() as sess:
        print(sess.run(c, feed_dict={a: [1, 2, 3]}))   # 需要对placeholder部分进行数据填充
        # print(sess.run(c, {a: [1, 2, 3]}))   # same

add_place_holders()

'''
    tips:
        tf.placeholder(tf.float32, shape=None) means that tensors of any shape will be accepted
'''

print("--------------------------feedable")
a = tf.add(2, 5)
b = tf.multiply(a, 3)
with tf.Session() as sess:
    sess.run(b, feed_dict={a: 15})
    print(b.eval())


