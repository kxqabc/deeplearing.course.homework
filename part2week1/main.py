#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

from utils.init_utils import load_dataset

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset(is_plot=False)
    print train_X.shape
    print train_Y.shape
    plt.plot(train_X[0, :], train_X[1, :])
    plt.show()