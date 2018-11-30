#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from utils.lr_utils import load_dataset
from nerve_net.layer import NerveNetwork
from nerve_net.linear import LINEAR
from nerve_net.activation import ACTIVATION

def get_data_info():
    """
    加载上周课程中的图片数据集
    :return: 数据集字典
    """
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
    data_info_dict = get_data_info()  # 加载上周课程中的图片数据集
    train_x_set = data_info_dict['train_set_x']
    lables = data_info_dict['train_set_y_orig']
    nerve_network = NerveNetwork(train_x_set, lables)
    nerve_network.add_layer(4, LINEAR['linear'], ACTIVATION['relu'])
    # nerve_network.add_layer(2, LINEAR['linear'], ACTIVATION['sigmoid'])
    print "layer_num: %d" % nerve_network.layer_num
    iteration_num = 2000
    learning_rate = 0.005
    cost_list = nerve_network.regression(iteration_num, learning_rate)
    # print str(nerve_network)
    print "cost: %s" % str(cost_list)
    nerve_network.cal_params_avg()
    #    x_axes = np.arange(0, len(cost_list), 1, dtype=int)
    #    plt.plot(x_axes, cost_list)
    #    plt.show()
    predicts = nerve_network.predict(data_info_dict['test_set_x'])
    nerve_network.cal_accuracy(predicts, data_info_dict['test_set_y_orig'])
