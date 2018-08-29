# -*- coding:utf-8 -*-

"""
Created on Oct 27, 2010
Update  on 2018-08-28
Logistic Regression Working Module
@author: Guangzhan
"""

import numpy as np

def sigmoid(x):
    # 这里其实非常有必要解释一下，会出现的错误 RuntimeWarning: overflow encountered in exp
    # 这个错误在学习阶段虽然可以忽略，但是我们至少应该知道为什么
    # 这里是因为我们输入的有的 x 实在是太小了，比如 -6000之类的，那么计算一个数字 np.exp(6000)这个结果太大了，没法表示，所以就溢出了
    # 如果是计算 np.exp（-6000），这样虽然也会溢出，但是这是下溢，就是表示成零
    return 1.0 / (1 + np.exp(-x))


def grad_descent(data_arr, class_labels):
    """
    梯度下降法，其实就是因为使用了极大似然估计，这个大家有必要去看推导，只看代码感觉不太够
    :param data_arr: 传入的就是一个普通的数组，当然你传入一个二维的ndarray也行
    :param class_labels: class_labels 是类别标签，它是一个 1*100 的行向量。
                    为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给label_mat
    :return:
    """
    # 注意一下，我把原来 data_mat_in 改成data_arr,因为传进来的是一个数组，用这个比较不容易搞混
    # turn the data_arr to numpy matrix
    data_mat = np.mat(data_arr)
    # 变成矩阵之后进行转置
    label_mat = np.mat(class_labels).transpose()
    # m->数据量，样本数 n->特征数
    m, n = np.shape(data_mat)
    # 学习率，learning rate
    alpha = 0.001
    # 最大迭代次数，假装迭代这么多次就能收敛2333
    max_cycles = 500
    # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # 这里是点乘  m x 3 dot 3 x 1
        h = sigmoid(data_mat * weights)  # sigmoid
        error = h - label_mat
        # 这里比较建议看一下推导，这里已经是求导之后的
        weights = weights - alpha * data_mat.transpose() * error
    return weights

# -------从疝气病症预测病马的死亡率------


def classify_vector(in_x, weights):
    """
    最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    :param in_x: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return:
    """
    # print(np.sum(in_x * weights))
    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0


def colic_test():
    """
    打开测试集和训练集，并对数据进行格式化处理,其实最主要的的部分，比如缺失值的补充（真的需要学会的），人家已经做了
    :return:
    """
    f_train = open('../input/03.LR/HorseColicTraining.txt', 'r')
    f_test = open('../input/03.LR/HorseColicTest.txt', 'r')
    training_set = []
    training_labels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in f_train.readlines():
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 下降算法 求得在此数据集上的最佳回归系数 trainWeights
    train_weights = grad_descent(np.array(training_set), training_labels)
    error_count = 0
    num_test_vec = 0.0


    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('the error rate is {}'.format(error_rate))
    return error_rate


if __name__ == '__main__':
    # 请依次运行下面三个函数做代码测试
    colic_test()

