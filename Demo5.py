# Defer creating/initializing an object until it is needed
import tensorflow as tf

x = tf.Variable(10, name="x")
y = tf.Variable(20, name="y")

# Normal loading
# z = tf.add(x, y)   # 在使用之前就创建节点
writer = tf.summary.FileWriter('../graphs/Demo5', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        # sess.run(z)
        # Lazy loading
        sess.run(tf.add(x, y))  # 调用add操作的时候才建立add节点, 差别见TensorBoard
writer.close()

'''
Go to terminal run:
    tensorboard --logdir="graphs/Demo5" --port 6006
    http://ArlenIAC:6006
    当计算一个像add这样的操作成千上万次，图形变得臃肿并且加载昂贵！github上最常见的不是bug的bug.
    
'''