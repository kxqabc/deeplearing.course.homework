#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np

from nerve_net.linear import *
from nerve_net.activation import *


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


class NerveLayer(object):
    def __init__(self, node_num, pre_layer, linear_dict, activation_dict, input_layer=False, output_layer=False, output=None, keep_prob=1.0):
        self.node_num = node_num        # 神经元个数
        self.pre_layer = pre_layer      # 上一层
        self.next_layer = None          # 下一层
        self.index = (pre_layer.index + 1) if pre_layer else 0  # 此层的索引，input层为0
        self.is_input = input_layer     # 是否为输入层
        self.is_output = output_layer   # 是否为输入层
        self.w = None
        self.b = None
        self.dw = None
        self.db = None
        self.init_params()              # 初始化"w, b, dw, db"
        self.liner_forward, self.liner_backward = self.init_linear_func(linear_dict)    # 线性函数
        self.activation_forward, self.activation_backward = self.init_activation_func(activation_dict)  # 激活函数
        self.output = output            # 经神经元计算后的输出数据
        self.intermediate = dict()      # 缓存中间值
        self.keep_prob = keep_prob      # dropout参数

    def init_linear_func(self, linear_dict):
        if self.is_input:
            return None, None
        return linear_dict['forward'], linear_dict['backward']

    def init_activation_func(self, activation_dict):
        if self.is_input:
            return None, None
        return activation_dict['forward'], activation_dict['backward']

    def init_params(self):
        if self.is_input:
            return
        pre_layer_node_num = self.pre_layer.node_num
        np.random.seed(5)
        self.w = np.random.randn(self.node_num, pre_layer_node_num) * np.sqrt(2.0/pre_layer_node_num)
        # self.w = np.random.randn(self.node_num, pre_layer_node_num) * 0.01
        # self.w = np.zeros((self.node_num, pre_layer_node_num))
        self.b = np.zeros((self.node_num, 1))
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)

    def update_params(self, learning_rate=0.01):
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db

    def __str__(self):
        info = "index: %d\n" % self.index
        info += "w:\n %s\n" % str(self.w)
        info += "b:\n %s\n" % str(self.b)
        info += 'dw:\n %s\n' % str(self.dw)
        info += 'db:\n %s\n' % str(self.db)
        info += 'z:\n %s\n' % str(self.intermediate)
        info += 'a:\n %s\n' % str(self.output)
        return info


class InputLayer(NerveLayer):
    """
    zero layer(input layer)
    """

    def __init__(self, input_data):
        super(InputLayer, self).__init__(input_data.shape[0], None, None, None, input_layer=True, output=input_data)


class OutputLayer(NerveLayer):
    """
    output layer
    """

    def __init__(self, pre_layer):
        """
        1. node num must be 1 because of output need to be dim(1);
        2. has default linear function and activation function;

        :param pre_layer: previous layer, in the beginning pre_layer is input_layer;
        """
        super(OutputLayer, self).__init__(1, pre_layer, LINEAR['linear'], ACTIVATION['sigmoid'], input_layer=False,output_layer=True)

    def update_after_add(self):
        """
        update 'w,b and index' after network change
        :return: output_layer
        """
        self.init_params()
        self.index = self.pre_layer.index + 1
        return self


class NerveNetwork(object):
    def __init__(self, train_x_set, train_y_set):
        self.__labels = train_y_set
        self.__header = InputLayer(train_x_set)
        self.__tail = OutputLayer(self.__header)
        self.__header.next_layer = self.__tail
        self.__layer_num = 1

    def add_layer(self, node_num, liner_dict, activation_dict, keep_prob=1.0):
        pre_layer = self.__tail.pre_layer
        new_layer = NerveLayer(node_num, pre_layer, liner_dict, activation_dict, keep_prob=keep_prob)
        new_layer.next_layer = self.__tail
        pre_layer.next_layer = new_layer
        self.__tail.pre_layer = new_layer
        # update output_layer's params like: w, b, index
        self.__tail.update_after_add()
        self.__layer_num += 1
        return self.__tail

    def format_params(self, input_data=None):
        """
        格式化神经网络的参数，恢复为初始状态
        :param input_data: 格式化的输入数据，格式化参数时会参考输入数据的shape，
        如果不为None，说明重新设置训练数据，不仅参数会发生变化，参数的shape也可能会变化；
        :return:
        """
        if input_data:
            self.__header.output = input_data
            self.__header.node_num = input_data.shape[0]
        curr_layer = self.__header.next_layer
        while curr_layer is not None:
            curr_layer.init_params()
            self.output = None
            self.intermediate = dict()
            curr_layer = curr_layer.next_layer

    def forward(self):
        curr_layer = self.__header.next_layer
        while curr_layer is not None:
            z = curr_layer.liner_forward(curr_layer.pre_layer.output, curr_layer.w, curr_layer.b)
            a = curr_layer.activation_forward(z)
            if not curr_layer.is_output and curr_layer.keep_prob != 1.0:
                # dropout处理
                d = np.random.rand(a.shape[0], a.shape[1])
                d = d < curr_layer.keep_prob
                a *= d
                a /= curr_layer.keep_prob
            curr_layer.intermediate['z'] = z
            curr_layer.output = a
            curr_layer = curr_layer.next_layer

    def cal_cost(self, a, regular=False, lambd=1.0):
        y = self.__labels
        m = float(y.shape[1])
        cost = -np.sum(np.dot(y, np.log(a).T) + np.dot(1.0 - y, np.log(1.0 - a).T)) / m
        if regular:
            # 计算正则化误差
            curr_layer = self.__header.next_layer
            regular_cost = 0.0
            while curr_layer is not None:
                regular_cost += np.sum(np.square(curr_layer.w))
                curr_layer = curr_layer.next_layer
            regular_cost = lambd * regular_cost / (2 * m)
            cost += regular_cost
        return cost

    def update_params(self, learning_rate):
        curr_layer = self.__header.next_layer
        while curr_layer is not None:
            curr_layer.update_params(learning_rate=learning_rate)
            curr_layer = curr_layer.next_layer

    def backward(self, regular=False, lambd=1.0):
        # calculate output_layer's dz and da
        y = self.__labels
        a = self.__tail.output
        m = float(y.shape[1])
        # backward from pre output_layer
        curr_layer = self.__tail
        da = None
        while not curr_layer.is_input:
            d = curr_layer.intermediate.get('d')    # dropout参数矩阵
            # 1. 计算dz
            if curr_layer.is_output:
                dz = (a - y)/m
            else:
                if d:
                    # 反向dropout
                    da *= d
                    da /= curr_layer.keep_prob
                dz = curr_layer.activation_backward(da, curr_layer.output)
            # 2. 计算dw, db
            x = curr_layer.pre_layer.output
            dw, db = curr_layer.liner_backward(dz, x)
            # 3. 是否正则化防止过拟合
            if regular:
                regular_effect = (lambd / m) * curr_layer.w
                dw += regular_effect
            # 4. 更新参数dw, db
            curr_layer.dw = dw
            curr_layer.db = db

            # this judge is very important, if not, it will task a lot of time;
            # When close to input layer, the params matrix are often vary large(means take a lot of time),
            # and update this param "da" is useless;
            if not curr_layer.pre_layer.is_input:
                da = np.dot(curr_layer.w.T, dz)     # update da
            curr_layer = curr_layer.pre_layer

    @time_meter
    def regression(self, iteration_num, learning_rate, regular=False, lambd=1.0):
        cost_list = []
        for i in range(iteration_num):
            self.forward()
            if i % 100 == 0:
                cost_list.append(self.cal_cost(self.__tail.output, regular=regular, lambd=lambd))
            self.backward(regular=regular, lambd=lambd)
            self.update_params(learning_rate)
        return cost_list

    def cal_params_avg(this):
        curr_layer = this.header.next_layer
        while curr_layer is not None:
            w_avg = np.mean(curr_layer.w)
            b_avg = np.mean(curr_layer.b)
            print "the %d layer's w_avg: %s, b_avg: %s" % (curr_layer.index, str(w_avg), str(b_avg))
            curr_layer = curr_layer.next_layer

    def predict(self, x_set):
        """
        与forward不同的是，predict最好不要影响神经网络的参数：output, cache等
        :param x_set: 输入数据集
        :return: 预测结果列表, shape = (1, item_num)
        """
        curr_layer = self.__header.next_layer
        a = x_set
        while curr_layer is not None:
            z = curr_layer.liner_forward(a, curr_layer.w, curr_layer.b)
            a = curr_layer.activation_forward(z)
            curr_layer = curr_layer.next_layer
        return np.round(a)

    @staticmethod
    def cal_accuracy(predicts, labels):
        assert predicts.shape == labels.shape
        accuracy = 100 - np.mean(np.abs(predicts - labels)) * 100
        print "正确率： %s%%" % accuracy
        return accuracy

    def __str__(self):
        curr_layer = self.__header.next_layer
        info = "There are %d layers in this network.\n" % self.__layer_num
        while curr_layer is not None:
            info += "############# Layer ################\n"
            info += str(curr_layer)
            info += "\n"
            curr_layer = curr_layer.next_layer
        return info
