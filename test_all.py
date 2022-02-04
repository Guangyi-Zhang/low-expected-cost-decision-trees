import numpy as np
import pickle
import pytest
from os import path

from sklearn.model_selection import train_test_split

from util import *


@pytest.fixture()
def iris_data():
    with open(path.join('dataset', 'iris-bin5.pkl'), 'rb') as f:
        X, Y = pickle.load(f)
        #X, Y = majority_label(X.values, Y.values)
        Xtr, Xt, Ytr, Yt = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)
        #return Xtr, Xt, Ytr, Yt
        return Xtr.values, Xt.values, Ytr.values, Yt.values


def test_group():
    arr = [2,1,2,1,3]
    gs = group(np.sort(arr))
    assert list(gs) == [(0, 2), (2, 2), (4, 1)]


def test_make_pairs():
    l = np.array([1,2,3,1,2])
    pairs1 = make_pairs(l, np.arange(len(l)))
    pairs = make_pairs(l, np.arange(len(l)), pairs=True)
    assert pairs1 == len(pairs)
    assert len(pairs) == 2*3 + 2*1
    print(pairs)
    assert pairs[1] == (0,4)
    assert pairs[-1] == (4,2)


def test_split_by_test():
    X = np.array([1, 2, 3, 1, 2]).reshape(-1, 1)
    gidx = split_by_test(X, np.arange(X.shape[0]), 0)
    print(gidx)
    assert len(gidx) == 3
    assert gidx[0][0].tolist() == 1
    assert gidx[1][0].tolist() == 2
    assert gidx[2][0].tolist() == 3
    assert gidx[0][1].tolist() == [0,3]
    assert gidx[1][1].tolist() == [1,4]
    assert gidx[2][1].tolist() == [2]


def test_preproc():
    X = np.array([[0,0],
                  [1,1],
                  [1,1],
                  [1,1]])
    Y = np.array([0,0,1,1])
    X_, Y_, _ = majority_label(X, Y)
    assert len(Y_) == 4
    assert sum(Y_) == 3
    i = np.argmin(Y_)
    assert X_[i].tolist() == [0,0]
