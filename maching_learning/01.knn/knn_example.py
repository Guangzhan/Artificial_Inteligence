#!/usr/bin/env python
# coding: utf-8

'''
Created on May 19, 2018
@author: Guangzhan
《机器学习实战》更新地址：https://github.com/Guangzhan/AiLearning
'''

import numpy as np   # 导入科学计算包numpy
import operator      # 导入运算符模块operator


"""
  创建数据集和标签
  group：训练集
  labels：标签
"""
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])        # 数据集矩阵
    labels = ['A', 'A', 'B', 'B']              # 定义列表标签
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    inX: 用于分类的输入向量 ，需要预测的数据[0.1,0.1]
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    """
        >>> import numpy as np
        >>> a = [1,2]
        >>> np.tile(a,(3,1))
        array([[1, 2],
               [1, 2],
               [1, 2]])
    """
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile生成和训练样本对应的矩阵，并与训练样本求差


    sqDiffMat = diffMat ** 2                 # 乘方
    sqDistances = sqDiffMat.sum(axis=1)      # 将矩阵的每一行相加
    distances = sqDistances ** 0.5           # 求距离的的开方
    """
        >>> x = np.array([3[0], 1[1], 2[0]])
        >>> np.argsort(x)
        array([1, 2, 0])
    """
    sortedDistIndicies = distances.argsort()   # sortedDistIndicies: [3 2 1 0] 位置最小的排在前面


    """
        选择距离最小的k个点
    """
    classCount = {}                # 定义字典
    for i in range(k):
        print(sortedDistIndicies[i])
        voteLabel = labels[sortedDistIndicies[i]]        #[3 2 1 0]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    print(classCount)
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)  # 对字典按值降序排序，返回类别最多的值在前面
    print(sortedClassCount)
    return sortedClassCount[0][0]

def main():
    """
    第一个例子演示
    """
    group, labels = createDataSet()                    # 调用
    print (classify0([0.1, 0.1], group, labels, 3))


if __name__ == '__main__':
    main()