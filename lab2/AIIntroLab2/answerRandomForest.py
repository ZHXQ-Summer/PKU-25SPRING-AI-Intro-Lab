from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 15     # 树的数量
ratio_data = 0.65   # 采样的数据比例
ratio_feat = 0.65 # 采样的特征比例
hyperparams = {
    "depth": 6, 
    "purity_bound": 1, 
    "gainfunc": negginiDA
    } # 每颗树的超参数


def _remap_tree_features(node, feature_mapping):
    if node.isLeaf():
        return
    node.featidx = feature_mapping[node.featidx]
    for child in node.children.values():
        _remap_tree_features(child, feature_mapping)

def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees=[]
    n=X.shape[0]
    m=X.shape[1]
    for i in range(num_tree):
        sample_indices = np.random.choice(n, 
                                         size=int(n*ratio_data), 
                                         replace=True)  
        new_X = X[sample_indices]
        new_Y = Y[sample_indices]
        n_selected_feat = int(m * ratio_feat)
        feat_indices = np.random.choice(m, n_selected_feat, replace=False)
        now_X = new_X[:, feat_indices]
        tree=(buildTree(now_X, new_Y, unused=list(range(n_selected_feat)), **hyperparams))
        _remap_tree_features(tree, feat_indices)
        trees.append(tree)

    return trees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
