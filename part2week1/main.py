#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

from utils.reg_utils import load_2D_dataset
from nerve_net.layer import NerveNetwork
from nerve_net.linear import LINEAR
from nerve_net.activation import ACTIVATION


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=False)
    print train_X.shape
    print train_Y.shape
    nerve_network = NerveNetwork(train_X, train_Y)
    nerve_network.add_layer(20, LINEAR['linear'], ACTIVATION['relu'], keep_prob=0.4)
    nerve_network.add_layer(10, LINEAR['linear'], ACTIVATION['relu'], keep_prob=0.5)
    iteration_num = 10000
    learning_rate = 0.01
    # 使用正则化
    # print "####################使用正则化###########################"
    # lambd = 3.0
    # cost_list = nerve_network.regression(iteration_num, learning_rate, regular=True, lambd=lambd)
    # print "cost: %s" % str(cost_list)
    # predicts = nerve_network.predict(test_X)
    # nerve_network.cal_accuracy(predicts, test_Y)
    print "####################不使用正则化########################"
    nerve_network.format_params()   # 格式化神经网络
    cost_list_without = nerve_network.regression(iteration_num, learning_rate)
    print "cost: %s" % str(cost_list_without)
    predicts = nerve_network.predict(test_X)
    nerve_network.cal_accuracy(predicts, test_Y)
