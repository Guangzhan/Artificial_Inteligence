#-*-coding:utf-8-*-


import numpy as np            # 导入numpy数组
import matplotlib.pyplot as plt   # 导入画图工具
from matplotlib.colors import ListedColormap    # 导入颜色列表
from sklearn.cross_validation import train_test_split        # 导入划分训练集和测试集函数
from sklearn.preprocessing import StandardScaler             # 导入标准差
from sklearn.datasets import make_moons, make_circles, make_classification   # 月亮形状数据、环形数据、分类数据
from sklearn.neighbors import KNeighborsClassifier         # KNN
from sklearn.svm import SVC                                # 支持向量机
from sklearn.tree import DecisionTreeClassifier            # 决策树
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier   # 随机森林、提升算法
from sklearn.naive_bayes import GaussianNB                                # 贝叶斯
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA     # 线性判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA   # 二次判别分析

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),                     # k近邻算法
    SVC(kernel="linear", C=0.025),               # 支持向量机线性核函数
    SVC(gamma=2, C=1),                           # 支持向量机高斯核函数
    DecisionTreeClassifier(max_depth=5),         # 决策树
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),   # 随机森林
    # max_depth: 随机森林决策树的最大深度
    # n_estimators: 决策树的个数
    # max_features：寻找最佳分割时需要考虑的特征数目
    AdaBoostClassifier(),                        # AdaBoost算法
    GaussianNB(),                                # 贝叶斯方法
    LDA(),                                       # 线性判别法
    QDA()]                                       # 二次判别法

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
                          # 总的特征数量
                          # 冗余信息特征
                          # 多信息特征的个数
                          # 随机数种子
                          # 某一个类别是由几个cluster构成的

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),            #  月亮形数据
            make_circles(noise=0.2, factor=0.5, random_state=1),   # 环形数据
            linearly_separable   # 线性可分割数据
            ]

figure = plt.figure(figsize=(27, 9))  # 创建一个27*9的点图
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)               # 标准化数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)   # 划分训练集和测试集

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))            # 划分网格

    # 先画出要分类的数据集
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])         # 加载颜色

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)   # 行，列，指定ax参数所在的区域
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)   # 训练样本点
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)  # 测试样本点
    ax.set_xlim(xx.min(), xx.max())       # x坐标轴限制
    ax.set_ylim(yy.min(), yy.max())       # y坐标轴限制
    ax.set_xticks(())                     # 去掉x轴刻度
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    # 迭代分类器
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)                      # 训练
        score = clf.score(X_test, y_test)              # 预测准确率

        #  划分决策边界，并为每一给边界给出不同的颜色
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])    # 计算样本点到分割超平面的函数距离。
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]  #  预测样本为某个标签的概率

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)    # 网格中填充区域

        # 训练集
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        #  测试机
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())   # x坐标轴限制
        ax.set_ylim(yy.min(), yy.max())   # # y坐标轴限制
        ax.set_xticks(())      # 去掉x坐标轴刻度
        ax.set_yticks(())      # 去掉y坐标轴刻度
        ax.set_title(name)     # 设置标题
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()