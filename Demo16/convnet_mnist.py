"""
    Using convolutional net on MNIST dataset of handwritten digits
    MNIST dataset: http://yann.lecun.com/exdb/mnist/
    Author: ArlenZhang
    Date: 2018.2.28
"""
import os
import time
import tensorflow as tf
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
    ================ CNN的convolution, pooling, full_conn部分设计 ==================
    parameters
        input : 输入图像
        filters : 过滤器长度
        k_size : 窗口尺寸
        stride : 窗口移动步长
        padding : 是否用0补全空位
        scope_name : 你懂得
"""
def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    """
        A method that does convolution + relu on inputs
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]  # 输入数据，图像input转成一列，所以列数对应的channels, RGB就是三种颜色3列

        # 初始化kernel数据
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

        # convolution在tensorflow中定义
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)

        # relu 规范化过程定义
        result = tf.nn.relu(conv + bias, name=scope.name)
    return result

"""
    池化层, pooling过程也有convolution的过程
    parameters
        inputs : 输入矩阵（上一层Relu之后的结果作为输入）
        k_size : 窗口尺寸
        stride : pooling 窗口的步长
        padding : pooling过程的pad类型
        scope_name : 你懂得
"""
def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    """
        A method that does max pooling on inputs
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool

"""
    全连接层，对pooling的结果到标签之间建立全连接
    parameters
        inputs : pooling的结果作为输入
        out_dim : 输出数据的维度, 标签个数
        scope_name : 你懂得
"""
def fully_connected(inputs, out_dim, scope_name='fc'):
    """
        A fully connected linear layer on inputs
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]  # 列数作为pooling层输出数据个数 ?
        w = tf.get_variable('weights', [in_dim, out_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

"""
    ================ 神经网络部分 ==================
"""
class ConvNet(object):
    def __init__(self):
        self.accuracy = self.summary_op = self.opt = self.loss = self.logits = self.test_init = self.train_init = \
            self.img = self.label = None
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            mnist_folder = '../../data/mnist'
            train_data, test_data = utils.get_mnist_dataset(self.batch_size, mnist_folder=mnist_folder)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # 上面生成的iterator通过下面函数进行下一步迭代，run epoch关键就在这里, iterator每次切换用train和test初始化过程中
            # self.img分别切换为train和test的iterator并执行next使得当前img卫队鹰操作需要数据的下一批次数据
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for test_data

    def create_logits(self):
        """
            数据流 or 模型逻辑结构
        """
        # 建立两层convolution + max_pooling
        conv1 = conv_relu(inputs=self.img,
                          filters=32,
                          k_size=5,        # 感受野为5*5
                          stride=1,        # 步长
                          padding='SAME',  # 补
                          scope_name='conv1')
        print(self.img.shape)
        print(conv1.shape)
        input()

        pool1 = maxpool(inputs=conv1,
                        ksize=2,
                        stride=2,
                        padding='VALID',
                        scope_name='pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=64,
                          k_size=5,
                          stride=1,
                          padding='SAME',
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        print("feature_dim: ", feature_dim)
        pool2 = tf.reshape(pool2, [-1, feature_dim])  # 转成feature_dim列的数据
        full_c = fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(full_c), self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')
        print(self.logits.shape)
        print(self.label.shape)
        input("shape")

    def create_loss(self):
        """
            定义损失函数
        """
        with tf.name_scope('loss'):
            print(self.label.shape)
            print(self.logits.shape)
            input("不一致的shape是什么鬼")
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            input("不一致?")
            self.loss = tf.reduce_mean(entropy, name='loss')
    
    def create_optimize(self):
        """
            反响传播，训练最优参数
        """
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def summary(self):
        """
            数据的展示汇总
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        """
            Count the number of right predictions in a batch
        """
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        """
            Build the computation graph
        """
        self.get_data()
        self.create_logits()
        self.create_loss()
        self.create_optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        print(self.label)
        print(self.logits)
        input()
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, '../../checkpoints/Demo16/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        """
            The train function alternates between training one epoch and evaluating
        """
        input("训练过程")
        utils.safe_mkdir('../../checkpoints')
        utils.safe_mkdir('../../checkpoints/Demo16')
        writer = tf.summary.FileWriter('../../graphs/Demo16', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../../checkpoints/Demo16/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=10)
    """
        tensorboard --logdir="graphs/Demo16"
        http://ArlenIAC:6006
    """
