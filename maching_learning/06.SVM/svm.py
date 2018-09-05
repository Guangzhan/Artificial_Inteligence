"""
==================================================
Plot different SVM classifiers in the iris dataset
不同的svm方法在鸢尾花数据集中的分类
==================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """创建一个网格用于画图

    Parameters
    ----------
    x: 网格的x轴
    y: 网格的y轴
    h: 划分网格的步长

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib 坐标轴类
    clf: 分类器
    xx: x轴网格 ndarray
    yy: y轴网格 ndarray
    params: contourf的字典
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # contour和contourf都是画三维等高线图的，不同点在于contourf会对等高线间的区域进行填充，
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 导入数据集
iris = datasets.load_iris()
# 使用数据集的前两个特征
X = iris.data[:, :2]
y = iris.target


C = 1.0  # SVM regularization parameter
models = (
          svm.LinearSVC(C=C),svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('LinearSVC (linear kernel)',         # 线性分类
          'SVC with linear kernel',            # 线性核函数
          'SVC with RBF kernel',               # 高斯核函数
          'SVC with polynomial (degree 3) kernel')   # 多项式核函数

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)               # 划分4个区域
plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 调整图像边缘及图像间的空白间隔

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)     # 划分网格


def main():
    # >>> a = [1, 2, 3]
    # >>> b = [4, 5, 6]
    # >>> c = [4, 5, 6, 7, 8]
    # >>> zipped = zip(a, b)  # 打包为元组的列表
    # [(1, 4), (2, 5), (3, 6)]

    for clf, title, ax in zip(models, titles, sub.flatten()):

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())   # 设定x坐标轴的上下限
        ax.set_ylim(yy.min(), yy.max())   # 设定y坐标轴的上下限
        ax.set_xlabel('Sepal length')     # 设定x轴的标题
        ax.set_ylabel('Sepal width')      # 设定y轴的标题
        ax.set_xticks(())    # 关闭x坐标轴刻度
        ax.set_yticks(())    # 关闭y坐标轴刻度
        ax.set_title(title)  # 设置标题

    plt.show()

if __name__ == '__main__':
    main()
