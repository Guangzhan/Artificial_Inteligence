# _*_ coding: utf-8 _*_

import numpy as np


#计算误差,均方误差
def compute_error_for_line_given_points(b, m, points):
    """
    :param b: 截距
    :param m: 斜率
    :param points: 样本点
    :return: 误差
    """
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

#梯度下降算法
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):        # 遍历样本
        x = points[i, 0]                   # 第一样本点
        y = points[i, 1]                   # 第二个样本点
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))        # 对参数b求导
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))    # 对参数m求导
    new_b = b_current - (learningRate * b_gradient)                       # 更新参数b
    new_m = m_current - (learningRate * m_gradient)                       # 更新参数m
    return [new_b, new_m]

# 执行梯度下降算法
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    """

    :param points:         样本点
    :param starting_b:     初始截距
    :param starting_m:     初始斜率
    :param learning_rate:  学习率
    :param num_iterations: 迭代次数
    :return:
    """
    b = starting_b                              # 初始化截距b
    m = starting_m                              # 初始化斜率m
    for i in range(num_iterations):             # 迭代1000次
        b, m = step_gradient(b, m, np.array(points), learning_rate)   # 调用梯度下降算法更新b, m
    return [b, m]

# 主函数
def main():
    points = np.genfromtxt("../input/02.LinearRegression/data.csv", delimiter=",")  # 读取数据集
    learning_rate = 0.0001
    initial_b = 0          # 初始化截距 guess
    initial_m = 0          # 初始化斜率 guess
    num_iterations = 1000  # 迭代次数
    print ("开始执行梯度下降算法 b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("输出系数和截距",[b,m])
    print ("迭代 {0} 后 b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    main()
