#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from train.logistic import init_params, regression, predict, cal_accuracy
from utils.lr_utils import load_dataset


def get_data_info():
    data_info_dict = dict()
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    print '训练集维度： %s' % str(train_set_x_orig.shape)
    print '测试集维度： %s' % str(test_set_x_orig.shape)
    # 原始数据
    data_info_dict['train_set_x_orig'] = train_set_x_orig
    data_info_dict['train_set_y_orig'] = train_set_y_orig
    data_info_dict['test_set_x_orig'] = test_set_x_orig
    data_info_dict['test_set_y_orig'] = test_set_y_orig
    # 一维化数据
    data_info_dict['train_set_x'] = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255.0
    data_info_dict['test_set_x'] = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255.0
    # 信息
    data_info_dict['classes'] = classes
    data_info_dict['n_train'] = train_set_x_orig.shape[0]
    data_info_dict['n_test'] = test_set_x_orig.shape[0]
    data_info_dict['pic_size'] = (train_set_x_orig[1:])
    return data_info_dict


if __name__ == '__main__':
    data_info_dict = get_data_info()
    X = data_info_dict['train_set_x']
    Y = data_info_dict['train_set_y_orig']
    w, b = init_params(X.shape[0])
    iteration_num = 3000
    learning_rate = 0.001
    optim_w, optim_b, cost = regression(w, b, X, Y, iteration_num, learning_rate)
    predicts = predict(optim_w, optim_b, data_info_dict['test_set_x'])
    cal_accuracy(predicts, data_info_dict['test_set_y_orig'])