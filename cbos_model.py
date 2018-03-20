"""
    运用CBOS思想，基于自己训练的词向量，用无监督的方法，用PDTB语料库训练EDU表示
    第一步：对PDTB整体的segmentation
    第二步：对EDU文件的装载，对word2ids的装载，对word_embedding的装载
    第三步：对无监督模型的设计

    理论上来说会建立一个edu_embedding, 个数等于edu个数，单个元素的长度等于edu最后要学习到的向量长度
    数据类型均采用 float32类型
    问题 ： 更新edu_embedding，这个embedding参数放在哪里

"""
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from code.utils.edu2vec_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class cbos_model:
    def __init__(self, dataset, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 这里的batch_size就是N个EDU，也就是n个EDU进来开始训练，论文中的N
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.skip_step = SKIP_STEP
        self.dataset = dataset
        # 获取初始化的eduids2vector 并将训练语料中的edu2ids进行存储
        self.edu_embedding = tf.convert_to_tensor(load_edu_embedding().astype(np.float32))  # (113813, 128)

    def _import_data(self):
        """
            加载数据, 这里加载EDU列表, 加载一批中心句和一批中心局的上下文, 均采用edu_ids即可
            负采样构成的target_sentences和之前训练好的词向量
        """
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.context_sents, self.target_sents, self.target_ids_tag = self.iterator.get_next()

    def _create_sent_embedding(self):
        """
            embedding lookup for center_sents compute the embedding for sentence embedding
        """
        with tf.name_scope('embed'):
            # 根据词向量计算上下文的平均表示从(?, 2*k+1)到(?, 2*k+1, edu_vector_size)
            # 将edu_embedding设置为placeholder
            self.edu_embedding_matrix = tf.Variable(
                initial_value=self.edu_embedding,
                name="_edu_embeddings",
                dtype=tf.float32,
                trainable=True  # 设置为可以被学习的模式，后期不断无监督学习
            )
            # shape is (100, 5, 128)
            context_sents_embed_a = tf.nn.embedding_lookup(self.edu_embedding_matrix, self.context_sents,
                                                                name='context_sents_embedding')
            # 对context进行求和
            context_sents_embed_b = tf.reduce_sum(context_sents_embed_a, 1)
            # 求平均,直接除以5即可  shape: (100, 128)
            self.context_sents_embed = tf.div(context_sents_embed_b, SKIP_WINDOW * 2 + 1)

            # 根据词向量计算target的向量值从(?, num_samples + 1) 到 (?, num_samples + 1, edu_vector_size)
            # shape is (100, 3, 128)
            self.target_sents_embed = tf.nn.embedding_lookup(self.edu_embedding_matrix, self.target_sents,
                                                             name='target_sents_embedding')
            # self.target_ids_tag 就是单纯的(100, num_samples + 1)

    # 对输入的一批batch_size的数据计算loss，最小化loss的过程得到最优的center_sents_embed
    def _create_loss(self):
        with tf.name_scope('loss'):
            loss_ = None
            # 外侧循环
            for i in range(0, BATCH_SIZE):
                # 内测循环
                inner_ = None
                for j in range(0, NUM_SAMPLED + 1):
                    temp = tf.multiply(self.target_ids_tag[i][j],
                                       tf.log(self.softmax_(self.target_sents_embed[i][j], i)))
                    if inner_ is None:
                        inner_ = temp
                    else:
                        inner_ = tf.add(inner_, temp)
                if loss_ is None:
                    loss_ = inner_
                else:
                    loss_ = tf.add(loss_, inner_)
            # 取负数
            self.loss = tf.multiply(-1.0, loss_)

    def softmax_(self, s_j, i):
        # 分子
        up = tf.exp(tf.multiply(s_j, self.context_sents_embed[i]))
        # 分母
        down = None
        for j in range(0, NUM_SAMPLED + 1):  # 迭代负采样+一个正例的结果
            s_k = self.target_sents_embed[i][j]
            temp_down = tf.exp(tf.multiply(s_k, self.context_sents_embed[i]))
            if down is None:
                down = temp_down
            else:
                down = tf.add(down, temp_down)
        s_ = tf.div(up, down)
        return s_

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                             global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
            Build the graph for our model
            建立模型大概就这几个模块分布考虑一下
        """
        self._import_data()
        self._create_sent_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()
        safe_mkdir('../../checkpoints')
        with tf.Session() as sess:
            # dataset 这套代码需要初始化iterator迭代器
            sess.run(self.iterator.initializer)
            # 查看数据状态
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../../checkpoints/e2v/checkpoint'))
            # 如果存在check_point就从上次的参数加载继续训练
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # 学习和训练的过程
            total_loss = 0.0
            writer = tf.summary.FileWriter('../../graphs/e2v/lr' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()
            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    if (index + 1) % 1000 == 0:
                        # 输出embedding
                        print("edu embeding 是否学习：")
                        print(self.edu_embedding[10].eval())
                    loss_batch, _ = sess.run([self.loss, self.optimizer])  # summary  , self.summary_op
                    # writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('训练第'+str(index+1)+'次计算的平均损失：')
                        print(total_loss / self.skip_step)
                        total_loss = 0.0
                        saver.save(sess, '../../checkpoints/e2v/cobs', index)
                        # print(len(self.embed_matrix.eval()))
                        # print(self.nce_weight.eval()[0])
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    # 生成视图, 给定视图文件名和要呈现的EDU个数
    def visualize(self, visual_fld, num_visualize):
        most_common_words(visual_fld, num_visualize)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../../checkpoints/e2v/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # sent_embedding
            # 就是最后要得到的
            final_embed_matrix = sess.run(self.edu_embedding)
            # update_embedding_matrix(final_embed_matrix)

            # 可视化
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


def update_embedding_matrix(final_m):
    with open("../../visualization_e2v/pdtb_word2ids.pkl", "rb") as f:
        dictionary = pkl.load(f)
    result_embedding = dict()
    for word in dictionary.keys():
        result_embedding[word] = final_m[dictionary[word]]
    # 写入
    with open("../../data/word2vec/pdtb_embedding.pkl", 'wb') as f:
        pkl.dump(result_embedding, f)


def gen():
    yield from batch_gen()


def main():
    # 获取EDU级的数据集
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32, tf.float32),
                                             (tf.TensorShape([BATCH_SIZE, 2 * SKIP_WINDOW + 1]),
                                              tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1]),
                                              tf.TensorShape([BATCH_SIZE, NUM_SAMPLED + 1])
                                              ))
    # 创建配置对象加载数据
    model = cbos_model(dataset, EDU_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize(VISUAL_FLD, NUM_VISUALIZE)


if __name__ == '__main__':
    main()

""" 
    run tensorboard --logdir='visualization_e2v'
    run tensorboard --logdir='graphs/e2v/lr0.5'
    run tensorboard --logdir='graphs/e2v/lr0.7'
    http://ArlenIAC:6006
"""
