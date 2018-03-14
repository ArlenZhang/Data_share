"""
    desc: 逻辑回归 手写体识别算法
    author: LongYinZ
    date: 2018.1.30
"""
import os
import tensorflow as tf
import time
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 200
n_train = 55000
n_test = 10000

# Step 1: Read in data
mnist_folder = '../data/mnist'

# 获取到训练集 验证集 测试集的图像 和 前两个的标签信息
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: 创建数据集和
# create training Dataset and batch it.  train.form : (data[:, 0], data[:, 1])
train_data = tf.data.Dataset.from_tensor_slices(train)
# 打乱数据顺序，据说训练效果会更好
train_data = train_data.shuffle(55000)
# 直接根据批处理的数据量大小将数据分批次摆好
train_data = train_data.batch(batch_size)
# 根据上面过程创建训练数据的数据集并批次化该数据集
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name="weights", shape=(784, 10), initializer=tf.random_normal_initializer(0, 1))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
logits = tf.matmul(img, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_min(entropy, name="loss")
# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Step 7: 计算准确率，这部分在模型训练之后进行计算 笔记本
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))  # 计数树别结果和训练标签一致的数量
"""
    理解 argmax:
        test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
        np.argmax(test, 0)  # 输出: array([3, 3, 1]) 所有数组都要互相比较，只比较相同位置上面的数
        np.argmax(test, 1)  # 输出: array([2, 2, 0, 0]) 比较每个数组内数的大小，会根据有几个数组产生几个结果
    所以上面的函数equal里面的两个参数分别是预测结果向量组和label标签的向量组
    通过计算得到的每个向量组中每个向量的标签结果
    equal之后得到的就是当前批次中预测结果和label结果相同与否[True, False, True ...]
    reduce_sum 得到的是结果中 True的个数
"""
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
print(accuracy)
writer = tf.summary.FileWriter('../graphs/Demo11', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            # 剖析这里的while True是因为用data_set 和iterate不断遍历各批次的数据替换了原来的手写代码分批词
            # 产生feed_dict数据，因为不知道一共有多少批，所以只能while True，直到data_set中批处理数据用完
            # 产生越界异常，结束训练。
            while True:
                _, loss_ = sess.run([optimizer, loss])
                total_loss += loss_
                n_batches += 1
        except tf.errors.OutOfRangeError:
            print("各个批次处理完毕，越界异常跳出while True循环！")
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    # 模型评测
    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
"""
    tensorboard --logdir="graphs/Demo11" --port 6006
    http://ArlenIAC:6006
"""
