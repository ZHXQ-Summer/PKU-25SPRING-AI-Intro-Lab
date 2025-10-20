import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.00028  # 学习率
wd1 = 0.2  # L1正则化
wd2 = 0.2 # L2正则化
batchsize = 175
hidden_dim1=256
hidden_dim2=128
def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mnist.mean_X, mnist.std_X), Linear(mnist.num_feat, 576),BatchNorm(576),relu(),Dropout(), Linear(576, 256),relu(),Linear(256, 128),relu(),Linear(128,mnist.num_class),LogSoftmax(), NLLLoss(Y)]
    graph=Graph(nodes)
    return graph
