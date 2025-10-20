import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.2 # 学习率
wd = 0.1 # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    A=np.matmul(X,weight)
    ans=np.add(A,bias)
    return ans

def sigmoid(x):
    return np.where(x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    A=np.matmul(X,weight)
    haty=np.add(A,bias)
    n=X.shape[0]
    cur_loss=-np.log(sigmoid(np.multiply(haty,Y))+1e-8)
    loss=np.mean(cur_loss)
    loss+=0.5 * wd * np.sum(weight**2)
    loss_wb=sigmoid(Y*haty) - 1
    partial_b=np.matmul(loss_wb.T,Y)/n
    partial_w=np.matmul((loss_wb*Y).T,X).T/n+wd*weight
    weight=np.subtract(weight,np.multiply(partial_w,lr))
    bias=np.subtract(bias,np.multiply(partial_b,lr))
    return(haty,loss,weight,bias)
