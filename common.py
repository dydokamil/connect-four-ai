import numpy as np
import scipy.signal

s_size = 7 * 6
a_size = 7


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
