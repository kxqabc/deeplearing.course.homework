#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


def sigmoid_backward(da, a):
    dz = da * a * (1.0-a)
    return dz


def relu(z):
    a = np.maximum(0, z)
    return a


def relu_backward(da, a):
    dz = np.array(da, copy=True)
    dz[a <= 0] = 0
    return dz


def tanh_backward(da, a):
    dz = da * (1.0 - np.power(a, 2))
    return dz


ACTIVATION = {
    'sigmoid': {
        'forward': sigmoid,
        'backward': sigmoid_backward
    },
    'relu': {
        'forward': relu,
        'backward': relu_backward
    },
    'tanh': {
        'forward': np.tanh,
        'backward': tanh_backward
    }
}