"""
    生成对抗
    1. 搭建CNN完成隐式论元关系分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import matplotlib.pyplot as plt
from utils.file_util import *
import random
from utils.model_data_util import batch_gen
from gensim.models import KeyedVectors
from cnn_model.model_config import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """
            输入： (batch_size, 1, 100, 50) padding_size: 100, embed_size: 50
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,   # n_filters
                kernel_size=5,     # filter size
                stride=1,          # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding
                                            #      =(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, arg1, arg2):
        arg1 = self.conv1(arg1)
        arg1 = self.conv2(arg1)
        arg1 = arg1.view(arg1.size(0), -1)     # flatten the output of conv2 to (batch_size, 32 * 7 * 7)

        arg2 = self.conv1(arg2)
        arg2 = self.conv2(arg2)
        arg2 = arg2.view(arg2.size(0), -1)
        arg_pair = torch.cat(arg1, arg2, 1)  # 希望在第一维拼接
        output = self.out(arg_pair)
        return output    # return x for visualization

class Adversial(nn.Module):
    def __init__(self, vocab, word_embed):
        nn.Module.__init__(self)
        self.vocab = vocab
        self.word_embed = word_embed
        self.vocab = vocab
        self.wordemb = nn.Embedding(len(vocab), EMBED_SIZE)
        self.wordemb.weight.data.copy_(torch.from_numpy(word_embed))
        self.wordemb.requires_grad = False
        self.cnn = CNN()

    def embedding_lookup(self, arg1_batch, arg2_batch):
        ids1 = Var(torch.LongTensor(arg1_batch))
        ids2 = Var(torch.LongTensor(arg2_batch))
        emb1 = self.wordemb(ids1).view(-1)
        emb2 = self.wordemb(ids2).view(-1)
        return emb1, emb2

    def forward(self, arg1_batch, arg2_batch, connective_batch):
        output = self.cnn(arg1_batch, arg2_batch)
        return output


# 生成对抗网络
class Arg_Rel_Cls:
    def __init__(self):
        torch.manual_seed(SEED)
        # 加载数据集
        vocab = load_data(VOC_WORD2IDS_PATH)
        word_embed = load_data(VOC_EMBED_PATH)
        # 创建对抗网络对象
        self.model = Adversial(vocab, word_embed)
        # 数据
        self.a_batch_train = batch_gen(TRAIN_PAIR_IDS_PATH)
        self.a_batch_test = batch_gen(TEST_PAIR_IDS_PATH)
        self.arg1_batch = self.arg2_batch = self.connective_batch = self.rel_batch = None

    def get_next(self, type_):
        if type_ is "train":
            self.arg1_batch, self.arg2_batch, self.connective_batch, self.rel_batch = next(self.a_batch_train)
        else:
            self.arg1_batch, self.arg2_batch, self.connective_batch, self.rel_batch = next(self.a_batch_test)

    def session(self, tree):
        return self.model.new_session(tree)

    """
        描述： 计算分数
    """
    def score(self, session):
        return self.model.score(session).data[0]

    """
        描述： 训练主过程
    """
    def train(self):
        random.seed(SEED)
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        # 定义学习器，对SPINN模型参数的学习，初始梯度全部置0
        optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_TRAIN_STEPS):
            iter_count = 0
            loss_all = 0.
            while True:
                try:
                    self.get_next(type_="train")
                    arg1_batch, arg2_batch = self.model.embedding_lookup(self.arg1_batch, self.arg2_batch)
                    output = self.model(arg1_batch, arg2_batch, self.connective_batch)
                    loss = loss_func(output, self.rel_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_all += loss
                    iter_count += 1
                    # 评测
                    # if epoch % SKIP_STEP == 0:
                    #     test_output, last_layer = cnn(test_x)
                    #     pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    #     accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                    #     if HAS_SK:
                    #         # Visualization of trained flatten layer (T-SNE)
                    #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    #         plot_only = 500
                    #         low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    #         labels = test_y.numpy()[:plot_only]
                    #         plot_with_labels(low_dim_embs, labels)
                except Exception:
                    print("Epoch: ", epoch, "  Loss: ", loss_all/iter_count)

#
#         if step % 50 == 0:
#             test_output, last_layer = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.squeeze()
#             accuracy = sum(pred_y == test_y) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
#             if HAS_SK:
#                 # Visualization of trained flatten layer (T-SNE)
#                 tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                 plot_only = 500
#                 low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
#                 labels = test_y.numpy()[:plot_only]
#                 plot_with_labels(low_dim_embs, labels)
