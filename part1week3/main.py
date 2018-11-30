#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

from nerve_layer.planar import layer_size, init_params, regression, predict, cal_accuracy
from utils.planar_utils import load_planar_dataset
from utils.lr_utils import load_dataset


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


def test_cat_pic():
    data_info_dict = get_data_info()  # 加载上周课程中的图片数据集
    x = data_info_dict['train_set_x']
    y = data_info_dict['train_set_y_orig']
    print "x shape: " + str(x.shape)
    print "y shape: " + str(y.shape)
    iteration_num = 2000
    learning_rate = 0.005
    params, cost_list = regression(x, y, iteration_num, learning_rate)
    # print "params: %s" % str(params)
    print "cost_list: %s" % str(cost_list)
    x_test = data_info_dict['test_set_x']
    y_test = data_info_dict['test_set_y_orig']
    predicts = predict(x_test, params)
    cal_accuracy(predicts, y_test)


def test_planar_data():
    x, y = load_planar_dataset()  # 加载平面点数据集
    print "x shape: " + str(x.shape)
    print "y shape: " + str(y.shape)
    iteration_num = 3000
    learning_rate = 0.6
    params, cost_list = regression(x, y, iteration_num, learning_rate)
    print "params: %s" % str(params)
    print "cost_list: %s" % str(cost_list)
    predicts = predict(x, params)
    cal_accuracy(predicts, y)


if __name__ == '__main__':
    # print "================ test planar data ==================="
    # test_planar_data()
    print "================== test cat pic ====================="
    test_cat_pic()