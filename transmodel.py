# coding:utf-8
# Author: oisc <oisc@outlook.com>

from collections import deque
from interface import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as nnfunc
from torch.autograd import Variable as Var
from gensim.models import KeyedVectors
import numpy as np
import random
import os
import pickle
from structure import RelationNode
from structure.wrapper import BinarizeTreeWrapper
import logging


logger = logging.getLogger(__name__)


_UNK = '<UNK>'
_DUMB = '<DUMB>'


class Reducer(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size * 5)

    def forward(self, left, right):
        h1, c1 = left.chunk(2)
        h2, c2 = right.chunk(2)
        a, i, f1, f2, o = self.proj(torch.cat([h1, h2])).chunk(5)
        c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return torch.cat([h, c])


class Tracker(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(3 * self.hidden_size, hidden_size)

    def forward(self, stack, buffer, state):
        s2, s1 = stack[-2], stack[-1]
        b1 = buffer[0]
        s2h, s2c = s2.chunk(2)
        s1h, s1c = s1.chunk(2)
        b1h, b1c = b1.chunk(2)
        cell_input = torch.cat([s2h, s1h, b1h]).view(1, -1)
        tracking_h, tracking_c = self.rnn(cell_input, state)
        return tracking_h.view(-1), tracking_c.view(-1)


class SPINN(nn.Module):
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self, hidden_size, vocab, wordemb_weights):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.wordemb_size = wordemb_weights.shape[1]
        self.vocab = vocab
        self.wordemb = nn.Embedding(len(vocab), self.wordemb_size)
        self.wordemb.weight.data.copy_(torch.from_numpy(wordemb_weights))
        self.wordemb.requires_grad = False
        self.tracker = Tracker(self.hidden_size)
        self.reducer = Reducer(self.hidden_size)
        self.edu_proj = nn.Linear(self.wordemb_size * 3, self.hidden_size * 2)
        self.trans_logits = nn.Linear(self.hidden_size, 1)
	
	# 对新的会话初始化
    def new_session(self, tree):
		# 初始状态空栈中存在两个空数据
        stack = [Var(torch.zeros(self.hidden_size * 2)) for _ in range(2)]  # [dumb, dumb]
        # 初始化队列
		buffer = deque()
        for edu in tree.edus():
            buffer.append(self.edu_encode(edu, tree))  # 对edu进行编码
        buffer.append(Var(torch.zeros(self.hidden_size * 2)))  # [edu, edu, ..., dumb]
        tracker_init_state = (Var(torch.zeros(self.hidden_size)) for _ in range(2))
        tracking = self.tracker(stack, buffer, tracker_init_state)
        return stack, buffer, tracking
	
    def score(self, session):
        stack, buffer, tracking = session
        h, c = tracking
        return nnfunc.sigmoid(self.trans_logits(h))

    def copy_session(self, session):
        stack, buffer, tracking = session
        stack_clone = [s.clone() for s in stack]
        buffer_clone = [b.clone() for b in buffer]
        h, c = tracking
        tracking_clone = h.clone(), c.clone()
        return stack_clone, buffer_clone, tracking_clone

	# 只对edu的0 1 -1
    def edu_encode(self, edu, tree):
        words = tree.words(edu.span)
        if len(words) == 0:
            return torch.zeros(self.hidden_size * 2)
        elif len(words) == 1:
            w1 = words[0]
            w2 = _DUMB
            w_1 = _DUMB
        else:
            w1 = words[0]
            w2 = words[1]
            w_1 = words[-1]
        ids = [self.vocab[word] if word in self.vocab else self.vocab[_UNK] for word in [w1, w2, w_1]]
        # if not edu:
        #    pass
        ids = Var(torch.LongTensor(ids))
        emb = self.wordemb(ids).view(-1)
        return self.edu_proj(emb)

    def forward(self, session, trans):
        stack, buffer, tracking = session

        if trans == self.SHIFT:  # SHIFT
            stack.append(buffer.popleft())
        else:  # REDUCE
            s1 = stack.pop()
            s2 = stack.pop()
            compose = self.reducer(s2, s1)
            stack.append(compose)
        tracking = self.tracker(stack, buffer, tracking)
        return stack, buffer, tracking

class SPINNSRModel(Model):
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        
		# 创建词汇表，创建word_embedding层
        wordemb = KeyedVectors.load_word2vec_format(args.word_emb, binary=True)
        vocab = {_DUMB: 0, _UNK: 1}
        for word in wordemb.vocab:
            vocab[word] = len(vocab)
        wordemb_weights = np.zeros((len(vocab), wordemb.vector_size), dtype=np.float32)
        for word in vocab:
            if word == _DUMB:
                pass
            elif word == _UNK:
                wordemb_weights[vocab[word]] = np.random.rand(wordemb.vector_size)
            else:
                wordemb_weights[vocab[word]] = wordemb[word]
        logger.log(logging.INFO, "loaded word embedding of vocabulary size %d" % len(vocab))
        self.model = SPINN(args.spinn_hidden, vocab, wordemb_weights)

    def session(self, tree):
        return self.model.new_session(tree)

    def score(self, session):
        return self.model.score(session).data[0]

    def shift(self, session, copy=False):
        if copy:
            session = self.model.copy_session(session)
        return self.model(session, SPINN.SHIFT)

    def reduce(self, session, copy=False):
        if copy:
            session = self.model.copy_session(session)
        return self.model(session, SPINN.REDUCE)

    def train(self, trees, trees_eval=None):
        random.seed(self.args.seed)
        trees = trees[:]
        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.spinn_lr)

        niter = 0
        nbatch = 0
        loss = 0.
        optimizer.zero_grad()
        for epoch in range(self.args.spinn_epoch):
            random.shuffle(trees)
            for tree in trees:
                niter += 1
                session = self.session(tree)
                reduce_score = []
                reduce_ground = []
                for trans in self.oracle(tree):
                    reduce_score.append(self.model.score(session))
                    session = self.model(session, trans)
                    reduce_ground.append(1 if trans == SPINN.REDUCE else 0)
                pred = torch.cat(reduce_score)
                ground = Var(torch.FloatTensor(reduce_ground))
                loss += criterion(pred, ground)
                # batch
                if niter % self.args.spinn_batch == 0 and niter > 0:
                    nbatch += 1
                    # backward
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # log and evaluate
                    if nbatch % self.args.spinn_logevery == 0:
                        if trees_eval:
                            eval_loss = self.evaluate(trees_eval)
                        else:
                            eval_loss = None
                        logger.log(logging.INFO, "[iter %-5d epoch %-3d  batch %-3d] train loss:%.5f eval loss:%.5f"
                                   % (niter, epoch, nbatch, loss.data[0] / self.args.spinn_batch, eval_loss))
                        loss = 0.
                    # checkpoint
                    yield nbatch

    def evaluate(self, trees):
        loss = 0.
        loss_fn = nn.BCELoss()
        for tree in trees:
            session = self.session(tree)
            reduce_score = []
            reduce_ground = []
            for trans in self.oracle(tree):
                score = self.model.score(session)
                session = self.model(session, trans)
                reduce_score.append(score)
                reduce_ground.append(1 if trans == SPINN.REDUCE else 0)
            pred = torch.cat(reduce_score)
            ground = Var(torch.FloatTensor(reduce_ground))
            loss += loss_fn(pred, ground)
        return loss.data[0] / len(trees)

    def oracle(self, tree):
        bitree = BinarizeTreeWrapper(tree)
        for node in bitree.traverse(prior=False):
            if isinstance(node, RelationNode):
                yield SPINN.REDUCE
            else:
                yield SPINN.SHIFT

    def save(self, folder):
        with open(os.path.join(folder, "torch.bin"), "wb+") as torch_fd:
            torch.save(self.model, torch_fd)

        with open(os.path.join(folder, "model.pickle"), "wb+") as model_fd:
            pickle.dump(self, model_fd)

    @staticmethod
    def restore(folder):
        with open(os.path.join(folder, "model.pickle"), "rb") as model_fd:
            model = pickle.load(model_fd)
        with open(os.path.join(folder, "torch.bin"), "rb") as torch_fd:
            torch_model = torch.load(torch_fd)
        model.model = torch_model
        return model
