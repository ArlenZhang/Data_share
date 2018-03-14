# Demo1

import tensorflow as tf
a = tf.add(3, 5)     # x = 3, y = 5
print(a)

# nodes : operators, variables, constants
# edges : tensors

# create a session, assign it into variable sess so we can call it later
# within the session, evaluate the graph to fetch the value of a
with tf.Session() as sess:
    print(sess.run(a))


# slide 46 多节点操作同时进行  注： pow(x, y)代表x的y次方
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z, not_useless = sess.run([pow_op, useless])
    print("pow_op: ", z)
    print("useless: ", not_useless)

# 将神经元计算结合分布式系统，用不同cpu计算各个节点 e.g. AlexNet
# 不同节点之间通过机柜内交换机进行IO数据传输
# 苏大集群，怎么将计算节点上的程序同时运行起来？

