import numpy as np
from itertools import groupby, accumulate
from collections import Counter

########## Incense ##########
from incense import ExperimentLoader

# Try to locate config file for Mongo DB
import importlib
spec = importlib.util.find_spec('mongodburi')
if spec is not None:
    from mongodburi import mongo_uri, db_name
else:
    mongo_uri, db_name = None, None


def get_loader(uri=mongo_uri, db=db_name):
    loader = ExperimentLoader(
        mongo_uri=uri,
        db_name=db
    )
    return loader


########## Util ##########

def group(l):
    '''Given a sorted list, group by value and return a list of (beg, len)'''
    lens = [len(list(g)) for _, g in groupby(l)]
    begs = [0] + list(accumulate(lens))[:-1]
    return zip(begs, lens)


def weight(idx, w):
    return sum([w[i] for i in idx])


def find_biggest_branch(Y, branches):
    '''Biggest cardinality'''
    h = np.argmax([len(i) for v,i in branches])
    return h


def split_by_test(X, idx, test):
    '''
    :return: a list of (val of test, idx of branch)
    '''
    t = test
    x = X[idx][:,t]
    ix = np.argsort(x)
    x, idx = x[ix], idx[ix]
    gidx = [(x[b], idx[b:b+l]) for b,l in group(x)]
    return gidx


def make_nonexc(Y, sorted=False, aggregate=False):
    # return dict of (label, #objects in Y not with the label)
    # if aggregate=True, return total sum on num of excluded objects in other classes
    if not sorted:
        Y = np.sort(Y)
    if aggregate:
        return sum([(len(Y)-l) * l for b, l in group(Y)])
    else:
        return dict([(Y[b], len(Y)-l) for b, l in group(Y)])


def make_pairs(Y, idx=None, pairs=False):
    '''
    :param pairs: return a list of real pairs; otherwise return a length to save time
    '''
    if idx is None:
        idx = np.arange(Y.shape[0])
    if len(idx) == 0:
        return []
    Y = Y[idx]
    iY = np.argsort(Y)
    Y, idx = Y[iY], idx[iY]
    if pairs:
        pairs = []
        for b,l in group(Y):
            for i1 in idx[b:b+l]:
                for i2 in idx[b+l:]:
                    pairs.append((i1,i2))
        return pairs
    else:
        sum = 0
        for b, l in group(Y):
            sum = sum + l * (len(idx)-b-l)
        return sum


########## Data related ##########

def gen_costs(ntest, scale=1, rn=None):
    if rn is not None:
        r = np.random.RandomState(rn)
        c = r.rand(ntest)
    else:
        c = np.random.rand(ntest)
    return c * scale


most_common = lambda l: np.bincount(l).argmax()  # labels are non-neg ints

def majority_label(X, Y):
    '''use majority class for each object'''
    lbl = dict()
    for x,y in zip(X,Y):
        key = tuple(x)
        if key not in lbl:
            lbl[key] = [y]
        else:
            lbl[key].append(y)

    # Assign the majority label to each y
    for k in lbl.keys():
        lbl[k] = most_common(lbl[k])
    Y = np.array([lbl[tuple(x)] for x in X])

    cnt = Counter([tuple(x) for x in X])
    newX, newY = zip(*[(list(k), lbl[k]) for k in cnt.keys()]) # list(k) gives back x
    newX, newY = np.array(newX), np.array(newY)
    idx = np.arange(newX.shape[0])
    tot = sum([v for v in cnt.values()])
    w = dict([(i, cnt[tuple(newX[i])] / tot) for i in idx])

    X, Y = newX, newY

    return X, Y, w


########## TO BE DELETED ##########
