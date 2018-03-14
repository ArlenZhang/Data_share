"""
    desc: Starter code for simple linear regression example using tf.data
    线性拟合略显智障，我们在这里找到曲线拟合效果, Y = ax^2 + bx + c
    author: LongYinZ
    date: 2018.1.30
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from advanced_util import huber_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_FILE = '../data/birth_life_2010.txt'

# Step 1: 读数据
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: 创建 data set 和 iterator遍历器
data_set = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
iterator = data_set.make_initializable_iterator()

X, Y = iterator.get_next()

# Step 3: create weight and bias, initialized to 0
w1 = tf.get_variable('weight1', initializer=tf.constant(0.0))
w2 = tf.get_variable('weight2', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: build model to predict Y
Y_predicted = X**2 * w1 + X * w2 + b

# Step 5: use the square error as the loss function
# loss = tf.square(Y - Y_predicted, name='loss')
loss = huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
start = time.time()
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('../graphs/Demo8', sess.graph)
    # Step 8: train the model for 100 epochs
    for i in range(100):
        sess.run(iterator.initializer)  # initialize the iterator
        total_loss = 0
        try:
            while True:
                _, loss_ = sess.run([optimizer, loss])  # 不需要feed_dict，因为数据在tensorflow中封装好了
                total_loss += loss_
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()
    w1_out, w2_out, b_out = sess.run([w1, w2, b])
    print('w1: %f,w2: %f, b: %f' % (w1_out, w2_out, b_out))
print('Took: %f seconds' % (time.time() - start))

# uncomment the following lines to see the plot
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')

# 构建画图数据
x_arr = data[:, 0]
x_arr = np.sort(x_arr)
y_arr = []
for item in x_arr:
    y_arr.append(item**2*w1_out + item*w2_out + b_out)
plt.plot(x_arr, y_arr, color='red', label='Predicted data')
plt.legend()
plt.show()
"""
    tensorboard --logdir="graphs/Demo8" --port 6006
    http://ArlenIAC:6006
"""
