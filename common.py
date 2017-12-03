import numpy as np
import scipy.signal
import torch.nn as nn

s_size = 7 * 6
a_size = 7
NUM_PROCESSES = 16
NUM_STACK = 1
CUDA = False
LR = 7e-4
EPS = 1e-5
ALPHA = .99
TAU = .95
GAMMA = .99
USE_GAE = False
NUM_STEPS = 2
NUM_FRAMES = 10e6
VALUE_LOSS_COEF = .5
ENTROPY_COEF = .01
MAX_GRAD_NORM = .5
LOG_INTERVAL = 100
SAVE_INTERVAL = 100
SAVE_DIR = './trained_models/'


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def one_hot_encode(index, max):
    a = np.zeros([len(index), max])
    for idx, oh in enumerate(index):
        a[idx, oh] = 1
    return a


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
