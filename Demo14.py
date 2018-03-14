"""
    word2vec skip-gram model with NCE loss and
    这套代码似乎更合理，运用create xxx 将一些函数过程传递给类，作为属性
    问题：怎么运行两套学习率代码比较曲线 toggle run
    run tensorboard --logdir='/home/arlenzhang/Desktop/Workstation/Period2/visualization'
        tensorboard --logdir=name1:'/home/arlenzhang/Desktop/Workstation/Period2/graphs/word2vec/lr1.0',
        name2:'/home/arlenzhang/Desktop/Workstation/Period2/graphs/word2vec/lr0.5'

    Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。
    而CBOW是给定上下文，来预测input word。本篇文章仅讲解Skip-Gram模型。
"""
import os
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import utils
import word2vec_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Model 超参数

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # number of negative examples to sample
LEARNING_RATE = 0.7
NUM_TRAIN_STEPS = 100000  # 训练10000次
VISUAL_FLD = '../visualization/Demo14'
SKIP_STEP = 5000  # 每5000次输出一次训练情况 打印loss等

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000  # number of tokens to visualize


class SkipGramModel:
    """ Build the graph for word2vec model """

    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP
        self.dataset = dataset
        # 定义权重和偏移量
        self.nce_weight = tf.get_variable('nce_weight',
                                          shape=[self.vocab_size, self.embed_size],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / (self.embed_size ** 0.5))
                                          )
        self.nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

    def _import_data(self):
        """
            Step 1: import data
        """
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        """
            Step 2 : embedding lookup.
            In word2vec, it's actually the weights that we care about
        """
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable('embed_matrix',
                                                shape=[self.vocab_size, self.embed_size],
                                                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embedding')

    def _create_loss(self):
        with tf.name_scope('loss'):
            """ Step 3: define the loss function """
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                                      biases=self.nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                             global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
            Build the graph for our model
            建立模型大概就这几个模块分布考虑一下
        """
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()
        initial_step = 0
        utils.safe_mkdir('../checkpoints')
        with tf.Session() as sess:
            # dataset 这套代码需要初始化iterator迭代器
            sess.run(self.iterator.initializer)
            # 全局初始化
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/Demo14/checkpoint'))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # 学习和训练的过程
            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('../graphs/Demo14/lr' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, '../checkpoints/Demo14/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    def visualize(self, visual_fld, num_visualize):
        # create the list of num_variable most common words to visualize
        word2vec_utils.most_common_words(visual_fld, num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/Demo14/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)


def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE,
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)


def main():
    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VISUAL_FLD, NUM_VISUALIZE)


if __name__ == '__main__':
    main()
""" 
    run tensorboard --logdir='visualization/Demo14'
    run tensorboard --logdir='graphs/Demo14/lr0.5'
    run tensorboard --logdir='graphs/Demo14/lr0.7'
    http://ArlenIAC:6006
"""
