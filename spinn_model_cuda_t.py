# coding:utf-8
# Author: oisc <oisc@outlook.com>
#         arlenzhang<arlenzhang128128@gmail.com>

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as nnfunc
from torch.autograd import Variable as Var
from gensim.models import KeyedVectors
import numpy as np
import random
import pickle
import logging
from utils.file_util import *
from cbos_model.model_config import VOC_WORD2IDS_PATH, CBOS_VEC_PATH, EMBED_SIZE
from parser.parser_config import *

logger = logging.getLogger(__name__)

_UNK = '<UNK>'
_DUMB = '<DUMB>'
_DUMB_IDX = 0

"""
    Desc： The composition function for reduce option.
"""
class Reducer(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.proj = nn.Linear(self.hidden_size * 4, self.hidden_size * 5)

    """
         Desc: The forward of Reducer
        input: 
	       The rep of left node and right node, e is the tree lstm's output, it has a different D.
       output:
               The rep of temp node 
    """

    def forward(self, left, right, tracking):
        h1, c1 = left.chunk(2)
        h2, c2 = right.chunk(2)
        e_h, e_c = tracking
        g, i, f1, f2, o = self.proj(torch.cat([h1, h2, e_h, e_c])).chunk(5)
        c = g.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return torch.cat([h, c])


"""
    Desc: tracker for tree lstm
"""
class Tracker(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(3 * self.hidden_size, hidden_size)
    
    """
        Desc: tracking lstm
    """
    def forward(self, stack, buffer_, state):
        s2, s1 = stack[-2], stack[-1]
        b1 = buffer_[0]
        s2h, s2c = s2.chunk(2)
        s1h, s1c = s1.chunk(2)
        b1h, b1c = b1.chunk(2)
        cell_input = torch.cat([s2h, s1h, b1h]).view(1, -1)
        #print(type(cell_input))
        #input(type(state))
        tracking_h, tracking_c = self.rnn(cell_input, state)  # forward of the model rnn
        return tracking_h.view(-1), tracking_c.view(-1)


class SPINN(nn.Module):
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self, hidden_size, word2ids, wordemb_weights):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.wordemb_size = wordemb_weights.shape[1]
        self.word2ids = word2ids
        self.wordemb = nn.Embedding(len(word2ids), self.wordemb_size)
        self.wordemb.weight.data.copy_(torch.from_numpy(wordemb_weights))
        self.wordemb.requires_grad = False
        self.tracker = Tracker(self.hidden_size)
        self.reducer = Reducer(self.hidden_size)
        self.edu_proj = nn.Linear(self.wordemb_size * 3, self.hidden_size * 2)
        self.trans_logits = nn.Linear(self.hidden_size, 1)

    """
         Desc: Create a new session
        Input:
               the root of a new tree
       Output:
               stack, buffer, tracking
    """
    def new_session(self, tree):
        # 初始状态空栈中存在两个空数据
        stack = [Var(torch.zeros(self.hidden_size * 2)).cuda() for _ in range(2)]  # [dumb, dumb]
        # 初始化队列
        buffer_ = deque()
        for edu_ in tree.edus:
            buffer_.append(self.edu_encode(edu_))  # 对edu进行编码
        buffer_.append(Var(torch.zeros(self.hidden_size * 2)).cuda())  # [edu, edu, ..., dumb]
        tracker_init_state = (Var(torch.zeros(self.hidden_size)).cuda() for _ in range(2))
        tracking = self.tracker(stack, buffer_, tracker_init_state)  # forward of Tracker
        return stack, buffer_, tracking

    """
        Desc: return a copy of a session.
    """
    def copy_session(self, session):
        stack, buffer, tracking = session
        stack_clone = [s.clone() for s in stack]
        buffer_clone = [b.clone() for b in buffer]
        h, c = tracking
        tracking_clone = h.clone(), c.clone()
        return stack_clone, buffer_clone, tracking_clone

    """
        Desc: sigmoid(fullc(h->1))
    """
    def score(self, session):
        stack, buffer_, tracking = session
        h, c = tracking
        return nnfunc.sigmoid(self.trans_logits(h))

    """
         Desc: Use the 0 1 -1 word vector of a sentence to encode an EDU
        Input: An object of rst_tree, leaf node
       Output: An output of code with lower dimension.
    """
    def edu_encode(self, edu):
        edu_ids = edu.temp_edu_ids
        if len(edu_ids) == 0:
            return torch.zeros(self.hidden_size * 2)
        elif len(edu_ids) == 1:
            w1 = edu_ids[0]
            w2 = _DUMB_IDX
            w_1 = _DUMB_IDX
        else:
            w1 = edu_ids[0]
            w2 = edu_ids[1]
            w_1 = edu_ids[-1]
        ids = np.array([w1, w2, w_1])
        # if not edu:
        #    pass
        ids = Var(torch.from_numpy(ids)).cuda()
        emb = self.wordemb(ids).view(-1)
        return self.edu_proj(emb)

    """
         Desc: The forward of SPINN
        Input:
               session and (shift or reduce)
       output:
               newest stack and buffer, lstm output
    """
    def forward(self, session, transition):
        stack_, buffer_, tracking = session
        if transition == self.SHIFT:
            stack_.append(buffer_.popleft())
        else:
            s1 = stack_.pop()
            s2 = stack_.pop()
            compose = self.reducer(s2, s1, tracking)  # the forward of Reducer
            stack_.append(compose)
        tracking = self.tracker(stack_, buffer_, tracking)  # The forward of the Tracker
        return stack_, buffer_, tracking


"""
    Desc: Traing Container
"""
class SPINN_SR:
    def __init__(self):
        torch.manual_seed(SEED)
        # load rst vocabulary and cbos vector
        word2ids = load_data(VOC_WORD2IDS_PATH)
        word_embed = load_data(CBOS_VEC_PATH)
        # logger.log(logging.INFO, "loaded word embedding of vocabulary size %d" % len(vocab))
        # build the objective of SPINN
        self.model = SPINN(SPINN_HIDDEN, word2ids, word_embed).cuda()

    def session(self, tree):
        return self.model.new_session(tree)

    def score(self, session):
        return self.model.score(session).data[0]

    """
        Desc: Train procedure.
    """
    def train_(self, trees, trees_eval=None):
        print("training...")
        random.seed(SEED)
        trees = trees[:]
        loss_fn = nn.BCELoss()  # binary cross-entropy loss
        optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        iter_count = 0
        batch_count = 0
        loss = 0.
        for epoch in range(num_train_steps):
            random.shuffle(trees)  # shuffle the data is a good choice
            for tree in trees:
                iter_count += 1
                session = self.session(tree)  # build the session for temporary tree.
                trainer_score = []
                gold_label = []
                for transition in oracle(tree):
                    trainer_score.append(self.model.score(session))  # compute the output
                    session = self.model(session, transition)  # forward of SPINN
                    gold_label.append(1 if transition == SPINN.REDUCE else 0)  # shift0 or reduce1 tag
                # concat all the output and action tag
                pred = torch.cat(trainer_score)
                label = Var(torch.FloatTensor(gold_label)).cuda()
                # compute the BCELoss between preds and labels.
                loss += loss_fn(pred, label)
                if iter_count % BATCH_SIZE == 0 and iter_count > 0:
                    batch_count += 1
                    # compute the grads of the loss function according to this batch of data
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    # Do evaluation every SKIP_STEP batches
                    if batch_count % SKIP_STEP == 0:
                        if trees_eval is not None:
                            eval_loss = self.evaluate(trees_eval)
                        else:
                            eval_loss = None
                        print("epoch: ", epoch, "  iter_count: ", iter_count)
                        print("train_loss: ", loss.data[0]/BATCH_SIZE, "  eval_loss:", eval_loss)
                        loss = 0.

    """
         Desc: evaluation, you know!
        Input:
               evaluate trees
       Output:
               acc, bceLoss[0] means the number of correct prediction.
    """
    def evaluate(self, trees):
        loss = 0.
        loss_fn = nn.BCELoss()
        for tree in trees:
            session = self.session(tree)  # build a new session
            eval_score = []
            gold_label = []
            # traverse the tree
            for transition in oracle(tree):
                score = self.model.score(session)
                session = self.model(session, transition)  # 调用SPINN的前馈过程
                eval_score.append(score)
                gold_label.append(1 if transition == SPINN.REDUCE else 0)
            pred = torch.cat(eval_score)
            label = Var(torch.FloatTensor(gold_label)).cuda()
            loss += loss_fn(pred, label)
        return loss.data[0] / len(trees)

    """
        Desc: Save the model.
    """
    def save(self, folder):
        save_data(self.model, os.path.join(folder, "torch.bin"), append=True)
        save_data(self, os.path.join(folder, "model.pickle"), append=True)

    """
        Desc: Load the model.
    """
    @staticmethod
    def restore(folder):
        model = load_data(os.path.join(folder, "model.pickle"))
        model.model = load_data(os.path.join(folder, "torch.bin"))
        return model


"""
     Desc: Back_traverse a gold tree of rst-dt
    Input: 
           The tree object of rst_tree
   Output: 
           Temp transition.
"""
def oracle(tree):
    for node in tree.nodes:
        if node.left_child is not None and node.right_child is not None:
            yield SPINN.REDUCE
        else:
            yield SPINN.SHIFT
    # yield SPINN.SHIFT
