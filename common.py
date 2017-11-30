import torch
import torch.nn as nn
import numpy as np
import scipy.signal

s_size = 7 * 6
a_size = 7
NUM_PROCESSES = 16
NUM_STACK = 4
CUDA = True
LR = 7e-4
EPS = 1e-5
ALPHA = .99
NUM_STEPS = 5
NUM_FRAMES = 10e6
VALUE_LOSS_COEF = .5
ENTROPY_COEF = .01
MAX_GRAD_NORM = .5


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1],
                                [1, -gamma],
                                x[::-1],
                                axis=0)[::-1]


def one_hot_encode(index, max):
    a = np.zeros([len(index), max])
    for idx, oh in enumerate(index):
        a[idx, oh] = 1
    return a
