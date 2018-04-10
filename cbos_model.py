"""
    学习的是edu表示，实际上还是对词向量的继续学习
    张
    2018.4.2
"""
import tensorflow as tf
from utils.cbos_util import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class cbos_model:
    def __init__(self, dataset, dataset_dev):
        self.global_step = tf.get_variable('global_step_cbos', initializer=tf.constant(0), trainable=False)
        self.dataset = dataset
        self.dataset_dev = dataset_dev
        load_embedding_ = load_data(VOC_EMBED_PATH)  # 注意，这里的ids2vec下标即为ids存成list数据
        self.embedding = load_embedding_.astype(np.float32)
        self.check_p = 'checkpoints/checkpoint'
        self.lr_p = 'graphs/lr'
        self.sess_p = 'checkpoints/cbos'
        self.train_loss_list = []
        self.dev_loss_list = []
        self.dev_loss_comp_list = []

    def _import_data(self):
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.context_sents, self.target_sents, self.target_sents_tag = self.iterator.get_next()
            self.iterator_dev = self.dataset_dev.make_initializable_iterator()
            self.context_sents_dev, self.target_sents_dev, self.target_sents_tag_dev = self.iterator_dev.get_next()

    def _create_sent_embedding(self):
        with tf.name_scope('embed'):
            # init_zero = np.ones(shape=(self.embedding.shape[0], self.embedding.shape[1]), dtype=np.float32)
            self.voc_embed_matrix = tf.get_variable(
                "embedding_glove",
                initializer=self.embedding,
                # shape=[self.embedding.shape[0], self.embedding.shape[1]],
                # initializer=tf.random_normal_initializer(),
                dtype=tf.float32,
                trainable=True
            )

            self.compare_matrix = tf.get_variable(
                initializer=self.embedding,
                name="compare_matrix",
                dtype=tf.float32,
                trainable=False
            )

            # 批数，上下文跨度，句子长度，词向量维度 (batch_size, 5, 200, 50)再对第3维度的200词的向量进行求和(batch_size, 5, 50)
            context_sents_embed = tf.nn.embedding_lookup(self.voc_embed_matrix, self.context_sents,
                                                         name='context_sents_embedding')
            context_sents_embed = tf.reduce_sum(context_sents_embed, 2)
            # 到上下文的句子表示求和取平均得到(batch_size, 50)，转shape到 (batch_size, 1, 50)
            context_sents_embed = tf.reduce_sum(context_sents_embed, 1)
            context_sents_embed = tf.div(context_sents_embed, SKIP_WINDOW * 2 + 1)
            self.context_sents_embed = tf.reshape(context_sents_embed, shape=(BATCH_SIZE, 1, EMBED_SIZE))

            # (batch_size, 5, 200, 50)再对第3维度的200词的向量进行求和->(batch_size, 5, 50) 4个neg 1个positive
            target_sents_embed = tf.nn.embedding_lookup(self.voc_embed_matrix, self.target_sents,
                                                        name='target_sents_embedding')
            self.target_sents_embed = tf.reduce_sum(target_sents_embed, 2)

            # dev验证集的数据映射
            context_sents_dev_embed = tf.nn.embedding_lookup(self.voc_embed_matrix, self.context_sents_dev, name='context_sents_dev_embedding')
            context_sents_dev_embed = tf.reduce_sum(context_sents_dev_embed, 2)
            context_sents_dev_embed = tf.reduce_sum(context_sents_dev_embed, 1)
            context_sents_dev_embed = tf.div(context_sents_dev_embed, SKIP_WINDOW * 2 + 1)
            self.context_sents_dev_embed = tf.reshape(context_sents_dev_embed, shape=(BATCH_SIZE, 1, EMBED_SIZE))
            target_sents_dev_embed = tf.nn.embedding_lookup(self.voc_embed_matrix, self.target_sents_dev, name='target_sents_dev_embedding')
            self.target_sents_dev_embed = tf.reduce_sum(target_sents_dev_embed, 2)

    def _create_loss(self):
        with tf.name_scope('loss'):
            # 计算余弦相似度
            cosin_dis = tf.divide(tf.reduce_sum(tf.multiply(self.context_sents_embed, self.target_sents_embed), 2), tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.context_sents_embed), 2)), tf.sqrt(tf.reduce_sum(tf.square(self.target_sents_embed), 2))))
            # 得到(batch_size, 5)
            self.logits = tf.get_variable(initializer=cosin_dis, name="logits")
            # target_sents_tag shape (batch_size, 5)  [[0, 0, 0, 0, 1] ...])
            self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_sents_tag, name="entropy")
            self.loss = tf.reduce_sum(self.entropy, name="loss")
            # w b
            # print(self.logits.shape)
            self.w = tf.get_variable(name="weight", shape=(5, 5), initializer=tf.random_normal_initializer(0, 1))
            self.b = tf.get_variable(name="bias", shape=(BATCH_SIZE, 1), initializer=tf.zeros_initializer())
            self.logits_test = tf.matmul(self.logits, self.w) + self.b
            self.entropy_test = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_test, labels=self.target_sents_tag, name="entropy_test")
            self.loss_test = tf.reduce_min(self.entropy_test, name="loss_test")
            self.optimizer_test = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss_test)

        with tf.name_scope("dev_loss"):
            cosin_dis_dev = tf.divide(tf.reduce_sum(tf.multiply(self.context_sents_dev_embed, self.target_sents_dev_embed), 2), tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.context_sents_dev_embed), 2)), tf.sqrt(tf.reduce_sum(tf.square(self.target_sents_dev_embed), 2))))
            self.dev_logits = tf.get_variable(initializer=cosin_dis_dev, name="dev_logits")
            self.dev_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.dev_logits, labels=self.target_sents_tag_dev, name="dev_entropy")
            self.dev_loss = tf.reduce_sum(self.dev_entropy, name="dev_loss")


    def _create_optimizer(self):
        """ Step 5: 定义学习器 """
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(
            self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss_a', self.loss)
            # tf.summary.scalar('loss_b', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
            建立模型大概就这几个模块分布考虑一下
        """
        self._import_data()
        self._create_sent_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        compare_arr = None
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # dataset 这套代码需要初始化iterator迭代器
            sess.run([self.iterator.initializer, self.iterator_dev.initializer])
            # 查看数据状态
            sess.run(tf.global_variables_initializer())
            # print(self.context_sents.eval()[0])
            # input()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.check_p))
            # 如果存在check_point就从上次的参数加载继续训练
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # 学习和训练的过程
            total_loss = 0.0
            total_dev_loss = 0.0
            writer = tf.summary.FileWriter(self.lr_p + str(LEARNING_RATE), sess.graph)
            initial_step = self.global_step.eval()
            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_, _, summary = sess.run([self.loss_test, self.optimizer_test, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_
                    if (index + 1) % SKIP_STEP == 0:
                        print("查看embedding是否更新！")
                        if compare_arr is None:
                            compare_arr = self.voc_embed_matrix.eval()
                        else:
                            print((compare_arr == self.voc_embed_matrix.eval()).all())
                            compare_arr = self.voc_embed_matrix.eval()
                        print("======================================================")
                        print(self.voc_embed_matrix.eval())
                        print(self.w.eval())
                        input("view")
                        
                        # 一个批次的模型计算dev上面的平均loss
                        try:
                            while True:
                                loss_comp_ = sess.run(self.dev_loss)
                                self.dev_loss_comp_list.append(loss_comp_)

                        except tf.errors.OutOfRangeError:
                            tmp_ave_loss = np.average(np.array(self.dev_loss_comp_list))
                            print("一次dev_lossiu计算结束:", tmp_ave_loss)
                            self.dev_loss_list.append(tmp_ave_loss)
                            self.dev_loss_comp_list = []
                            sess.run(self.iterator_dev.initializer)

                        tmp_step_train_loss = total_loss / SKIP_STEP
                        # print('训练第' + str(index + 1) + '次计算的平均损失：')
                        # print(tmp_step_train_loss)
                        self.train_loss_list.append(tmp_step_train_loss)
                        total_loss = 0.0
                        saver.save(sess, self.sess_p, index)

                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)

            final_embed_matrix = self.voc_embed_matrix.eval()
            compare_matrix = self.compare_matrix.eval()
            save_cbos_data(final_embed_matrix, compare_matrix)
            writer.close()

            # 对训练过程中的loss数据进行刻画，对一次训练的次数增大 看变化情况
            self.draw_losses()
    
    """
        对self的两个列表中的数据
    """
    def draw_losses(self):
        print("train_loss: ", self.train_loss_list)
        print("dev_loss: ", self.dev_loss_list)


def save_cbos_data(final_embed_matrix, compare_matrix):
    print("最终比较!")
    print((final_embed_matrix == compare_matrix).all())

    cbos_embed = dict()
    # 加载word2ids数据
    word2ids = load_data(VOC_WORD2IDS_PATH)
    for word in word2ids.keys():
        cbos_embed[word] = final_embed_matrix[word2ids[word]]
    save_data(cbos_embed, CBOS_EMBED_PATH)

def gen():
    yield from batch_gen(TRAIN_SENTS_IDS_PATH)

def gen_dev():
    yield from batch_gen(DEV_SENTS_IDS_PATH)

def main_cbos():
    # 获取EDU级的数据集
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE, 2 * SKIP_WINDOW + 1, PAD_SIZE]),
                                              tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1, PAD_SIZE]),
                                              tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1])
                                              ))
    
    dataset_dev = tf.data.Dataset.from_generator(gen_dev, (tf.int32, tf.int32, tf.int32),
                                                 (tf.TensorShape([BATCH_SIZE, 2 * SKIP_WINDOW + 1, PAD_SIZE]),
                                                  tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1, PAD_SIZE]),
                                                  tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1])
                                                  ))

    # 创建配置对象加载数据
    model = cbos_model(dataset, dataset_dev)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    # model.visualize(VISUAL_FLD, NUM_VISUALIZE)

"""
    run tensorboard --logdir='visualization_e2v_pdtb'
    run tensorboard --logdir='graphs/e2v/lr0.5'
    run tensorboard --logdir='graphs/e2v/lr0.7'
    http://ArlenIAC:6006
"""
