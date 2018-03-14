"""
    desc: Eager Executive
    Starter code for a simple regression example using eager execution.
    author: LongYinZ
    date: 2018.1.30
"""
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import utils
DATA_FILE = '../data/birth_life_2010.txt'
# must be called at the very beginning of a TensorFlow program.
tfe.enable_eager_execution()

class eager_executive:
    def __init__(self):
        # Read the data into a dataset.
        self.data, self.n_samples = utils.read_birth_life_data(DATA_FILE)
        self.data_set = tf.data.Dataset.from_tensor_slices((self.data[:, 0], self.data[:, 1]))

        # Create variables.
        self.w = tfe.Variable(0.0)
        self.b = tfe.Variable(0.0)

    # 模型预测值
    def prediction(self, x):
        return x * self.w + self.b

    # 代价函数 在Demo8中我也用了huber_loss进行代价计算
    def huber_loss(self, x, y, m=1.0):
        y_predicted = self.prediction(x)
        t = y - y_predicted
        return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

    # 设计训练过程 Train a regression model evaluated using `loss_fn`.
    def train(self):
        loss_fn = self.huber_loss
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        # 梯度更新函数,返回代价和代价函数对各个参数的求导梯度
        grad_fn = tfe.implicit_value_and_gradients(loss_fn)
        start = time.time()
        # 训练次数（梯度更新次数）：100
        # 不进行批处理，每次迭代进行
        for epoch in range(100):
            total_loss = 0.0
            for x_i, y_i in tfe.Iterator(self.data_set):
                loss, gradients = grad_fn(x_i, y_i)
                # Take an optimization step and update variables.
                self.optimizer.apply_gradients(gradients)
                total_loss += loss
            if epoch % 10 == 0:
                print('Epoch {0}: {1}'.format(epoch, total_loss / self.n_samples))
        print('Took: %f seconds' % (time.time() - start))
        """
            'Eager execution exhibits significant overhead per operation. '
            'As you increase your batch size, the impact of the overhead will '
            'become less noticeable. Eager execution is under active development: '
            'expect performance to increase substantially in the near future!'
        """

    # 画图
    def draw_plot(self):
        plt.plot(self.data[:, 0], self.data[:, 1], 'bo')
        plt.plot(self.data[:, 0], self.data[:, 0] * self.w.numpy() + self.b.numpy(), 'r',
                 label="huber regression")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    e_obj = eager_executive()
    e_obj.train()
    e_obj.draw_plot()
