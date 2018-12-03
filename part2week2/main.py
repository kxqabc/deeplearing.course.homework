#!/usr/bin/python
# -*- coding: UTF-8 -*-

from utils.opt_utils import load_dataset, random_mini_batches

if __name__ == '__main__':
    train_X, train_Y = load_dataset(is_plot=True)
    print "shape of X: %s" % str(train_X.shape)
    print "shape of Y: %s" % str(train_Y.shape)
    mini_batches = random_mini_batches(train_X, train_Y, mini_batch_size=64, seed=5)
    for batch in mini_batches:
        print "shape of batch: %s, %s" % (str(batch[0].shape), str(batch[1].shape))