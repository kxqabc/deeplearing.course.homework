#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np


def time_meter(func):
    """
    装饰器，作计时器使用
    :param func: 被装饰函数
    :return:
    """
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        cost = end - begin
        print "function %s cost: %d ms" % (str(func), int(round(cost * 1000)))
        return result
    return wrapper


def layer_size(x, y):
    x_layer_num = x.shape[0]
    y_layer_num = y.shape[0]
    hide_layer_num = 4
    return x_layer_num, y_layer_num, hide_layer_num


def sigmoid(z):
    """
    sigmoid
    :param z: 任意维度数组
    :return: shape==z.shape, 元素在(0, 1)之间
    """
    assert isinstance(z, np.ndarray)
    return 1/(1 + np.exp(-z))


def init_params(x_layer_num, y_layer_num, hide_layer_num):
    np.random.seed(5)
    w1 = np.random.randn(hide_layer_num, x_layer_num) * 0.01
    b1 = np.zeros(shape=(hide_layer_num, 1))
    w2 = np.random.randn(y_layer_num, hide_layer_num) * 0.01
    b2 = np.zeros(shape=(y_layer_num, 1))

    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }
    return params


def forward_propagation(x, params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    # 第一层
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)

    # 第二层
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    intermediate = {
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2
    }
    return intermediate


def backward_propagation(intermediate, w2, x, y):
    m = float(y.shape[1])
    a1 = intermediate['a1']
    a2 = intermediate['a2']
    dz2 = a2 - y 
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    da = np.dot(w2.T, dz2)
    dz1 = da * (1 - np.power(a1, 2))
    dw1 = np.dot(dz1, x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2
             }
    return grads


def update_params(params, grads, learning_rate):
    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新
    params['w1'] -= learning_rate * dw1
    params['b1'] -= learning_rate * db1
    params['w2'] -= learning_rate * dw2
    params['b2'] -= learning_rate * db2

    return params


def cal_cost(a, y):
    m = float(y.shape[1])
    cost = -np.sum(np.dot(y, np.log(a).T) + np.dot(1.0 - y, np.log(1.0 - a).T)) / m
    return cost


@time_meter
def regression(x, y, iteration_num, learning_rate=0.001):
    cost_list = []
    x_layer_num, y_layer_num, hide_layer_num = layer_size(x, y)
    params = init_params(x_layer_num, y_layer_num, hide_layer_num)

    for i in range(iteration_num):
        intermediate = forward_propagation(x, params)
        a2 = intermediate['a2']
        cost = cal_cost(a2, y)
        cost_list.append(cost)
        grads = backward_propagation(intermediate, params['w2'], x, y)
        update_params(params, grads, learning_rate)
    return params, cost_list


def predict(x, params):
    predict_dict = forward_propagation(x, params)
    a2 = predict_dict['a2']
    predict_array = np.round(a2)
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