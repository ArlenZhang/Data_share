import tensorflow as tf
# create variables with tf.Variable
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0, 1], [2, 3]], name="matrix")
W = tf.Variable(tf.zeros([784, 10]))

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

'''
    tf.constant is an op
    tf.Variable is a class with many ops
'''
# x = tf.Variable(...)
# x.initializer # init op
# x.value() # read op
# x.assign(...) # write op
# x.assign_add(...) # and more

with tf.Session() as sess:
    # print(sess.run(W)) FailedPreconditionError: Attempting to use uninitialized value Variable
    # The easiest way is initializing all variables at once:
    sess.run(tf.global_variables_initializer())

    print(W.eval())   # == print(sess.run(W))

    # Initialize only a subset of variables:
    # sess.run(tf.variables_initializer([a, b]))

    # Initialize a single variable
    # sess.run(W.initializer)

# ============================ assign
v = tf.Variable(10, name="v1")
v.assign(100)
with tf.Session() as sess:
    sess.run(v.initializer)
    print(v.eval())
    # >> 10 v.assign(100) creates an assign op.
    # That op needs to be executed in a session to take effect.

v = tf.Variable(10, name="v2")
op = v.assign(100)
with tf.Session() as sess:
    sess.run(v.initializer)
    sess.run(op)
    print(v.eval())

# ============================ interesting assign in Fibonacci
first = tf.Variable(1, name="first")
second = tf.Variable(1, name="second")
temp = tf.Variable(1, name="temp")
# ops
t = temp.assign(second)
s1 = second.assign(first + second)
s2 = first.assign(t)
with tf.Session() as sess:
    sess.run(first.initializer)
    sess.run(second.initializer)

    # 输出前10个Fibonacci
    n = 10
    while n > 0:
        n -= 1
        sess.run([t, s1, s2])
        print(first.eval())

# =========================== different assign
v = tf.Variable(20, name="v3")
op1 = v.assign_add(2)
op2 = v.assign_sub(1)
with tf.Session() as sess:
    sess.run(v.initializer)
    print("------------same session 才会同步 -------------")
    print(v.eval())
    sess.run(op1)
    print(v.eval())
    sess.run(op2)
    print(v.eval())
