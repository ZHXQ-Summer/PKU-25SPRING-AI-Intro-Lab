import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *
import numpy as np
import pickle
from scipy.ndimage import rotate, shift, zoom
from autograd.utils import PermIterator
from util import setseed
def augment_image(img):
    img = img.reshape(28, 28,order='C').astype(np.float32)
    a=np.random.randint(1,101)
    if(a<=80):
        img = rotate(img, np.random.uniform(-8, 8), order=1,reshape=False, mode='reflect')
    a=np.random.randint(1,101)
    if(a<=80):
        offset = np.random.randint(-5, 5, size=2)
        img = shift(img, offset, mode='constant', cval=0)
    return img.reshape(784,order='C')
    # 余弦退火调度（替代固定学习率）
def cosine_annealing(epoch, total_epochs, max_lr, min_lr):
    return min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(epoch/total_epochs * np.pi))
setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/your.npy"

sample_indices = np.random.choice(69000,size=int(69000*0.5),replace=False)
X=np.vstack([mnist.trn_X,mnist.val_X[sample_indices]])
Y=np.concatenate([mnist.trn_Y,mnist.val_Y[sample_indices]])
tX = np.array([augment_image(x) for x in X])
X=tX
std_X, mean_X = np.std(X, axis=0, keepdims=True)+1e-4, np.mean(X, axis=0, keepdims=True)
# 超参数
# TODO: You can change the hyperparameters here
lr = 0.00009 # 学习率
wd1 = 0.58  # L1正则化
wd2 = 0.58 # L2正则化
batchsize = 700
dropout_rate = 0.68
def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mean_X, std_X), Linear(mnist.num_feat, 576),BatchNorm(576),relu(),Dropout(dropout_rate), Linear(576, 256),relu(),Linear(256,mnist.num_class),LogSoftmax(), NLLLoss(Y)]
    for node in nodes:
        if isinstance(node, Linear):
            fan_in = node.params[0].shape[1]
            node.params[0] = np.random.randn(*node.params[0].shape) * np.sqrt(2./fan_in)
    graph=Graph(nodes)
    return graph

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 80+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)
    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(mnist.val_X, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("valid acc", np.average(haty==mnist.val_Y))
