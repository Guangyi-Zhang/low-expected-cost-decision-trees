import numpy as np

from util import weight, group, make_pairs, make_nonexc


def impurepairs(Y, idx, w):
    '''#impure pair'''
    return make_pairs(Y, idx)


def nonexcluded(Y, idx, w):
    '''sum of #objects in other classes over each object'''
    return make_nonexc(Y[idx], sorted=False, aggregate=True)


def entropy(Y, idx, w):
    if len(idx) == 0:
        return 0
    Y = Y[idx]
    iY = np.argsort(Y)
    Y, idx = Y[iY], idx[iY]
    E = 0
    ptot = weight(idx, w)
    for gidx in [idx[b:b+l] for b,l in group(Y)]:
        p = weight(gidx, w) / ptot
        E = E - p * np.log2(p)
    return E


def gini(Y, idx, w):
    if len(idx) == 0:
        return 0
    Y = Y[idx]
    iY = np.argsort(Y)
    Y, idx = Y[iY], idx[iY]
    sum = 0
    ptot = weight(idx, w)
    for gidx in [idx[b:b+l] for b,l in group(Y)]:
        p = weight(gidx, w) / ptot
        sum = sum + p**2
    return 1 - sum
