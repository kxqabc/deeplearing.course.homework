#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


def sigmoid(z):
    """
    sigmoid
    :param z: 任意维度数组
    :return: shape==z.shape, 元素在(0, 1)之间
    """
    assert isinstance(z, np.ndarray)
    return 1/(1 + np.exp(-z))


def init_params(dim):
    """
    初始化w, b参数
    :param dim: w的维度为（dim, 1）
    :return: w, b
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    """
    传播函数
    :param w: shape=(X.shape[0], 1)
    :param b: shape=( ,)
    :param X: shape=(横像素数*纵像素数*3, 图片个数)，输入数据集
    :param Y: shape=(1, 图片个数), 输入图片的标签(0, 1)
    :return: dw, db, cost
    """
    pix_num = float(X.shape[1])
    # 正向传播
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    cost = -np.sum(np.dot(Y, np.log(a).T) + np.dot(1.0 - Y, np.log(1.0 - a).T)) / pix_num

    # 反向传播
    dw = np.dot(X, (a - Y).T)/pix_num
    db = np.sum(a - Y, axis=1, keepdims=True)/pix_num
    return dw, db, cost


def regression(w, b, X, Y, iteration_num, learning_rate):
    """
    迭代求最优 w, b 参数
    :param w:
    :param b:
    :param X:
    :param Y:
    :param iteration_num: 迭代次数
    :param learning_rate: 每次更改的步长
    :return: w, b, cost_list
    """
    cost_list = []
    w = w.astype(np.float)
    for i in range(iteration_num):
        dw, db, cost = propagate(w, b, X, Y)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost_list.append(cost)
    print "after %s regression: w= %s, b= %s" % (iteration_num, w, b)
    print "cost: %s" % cost_list
    return w, b, cost_list


def predict(w, b, X):
    """
    根据w, b 参数预测X的输出
    :param w:
    :param b:
    :param X: 输入数据集, X.shape = (横像素数*纵像素数*3, 图片个数)
    :return: 预测结果，shape == (1, 图片个数)
    """
    predict_array = np.zeros(shape=(1, X.shape[1]))
    w = w.reshape(X.shape[0], 1)
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    for i in range(a.shape[1]):
        predict_array[0][i] = 1 if a[0][i] > 0.5 else 0
    return predict_array


def cal_accuracy(predicts, lables):
    """
    计算预测的准确率
    :param predicts: 预测结果，shape == (1, 图片个数)
    :param lables: 输入图片的正确标签，shape == (1, 图片个数)
    :return: float
    """
    assert predicts.shape == lables.shape
    accuracy = 100 - np.mean(np.abs(predicts - lables)) * 100
    print "正确率： %s%%" % accuracy
    return accuracy