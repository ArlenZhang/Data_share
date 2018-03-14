"""
    desc: Starter code for simple linear regression example using placeholders
    placeholder + feed_dict 模式
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

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
X = tf.placeholder(tf.float32, shape=None, name="birth_rate")
Y = tf.placeholder(tf.float32, shape=None, name="life_expectancy")

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w1 = tf.get_variable(name="weight1", dtype=tf.float32,  initializer=tf.constant(0.0))
w2 = tf.get_variable(name="weight2", dtype=tf.float32,  initializer=tf.constant(0.0))
b = tf.get_variable(name="bias", dtype=tf.float32,  initializer=tf.constant(0.0))

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w1 * X**2 + w2 * X + b

# Step 5: use the square error as the loss function
# loss = tf.square(Y - Y_predicted, name="loss")
loss = huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

with tf.Session() as sess:
    # Create a file writer to write the model's graph to TensorBoard
    writer = tf.summary.FileWriter('../graphs/Demo7', sess.graph)

    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # close the writer when you're done using it
    writer.close()
    
    # Step 9: output the values of w and b
    w1_out, w2_out, b_out = w1.eval(), w2.eval(), b.eval()
    print("w1_out: ", w1_out)
    print("w2_out: ", w2_out)
    print("b_out: ", b_out)

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
    tensorboard --logdir="graphs/Demo7" --port 6006
    http://ArlenIAC:6006
"""
