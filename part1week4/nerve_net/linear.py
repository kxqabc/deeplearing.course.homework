#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np


def linear_forward(x, w, b):
    z = np.dot(w, x) + b
    return z


def linear_backward(dz, x):
    dw = np.dot(dz, x.T)
    db = np.sum(dz, axis=1, keepdims=True)
    return dw, db

LINEAR = {
    'linear': {
        'forward': linear_forward,
        'backward': linear_backward
    }
}