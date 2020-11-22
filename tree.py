import numpy as np

from util import majority_label, group, \
    weight, make_pairs, split_by_test, find_biggest_branch
from impurity import entropy, gini


class Node(object):

    def __init__(self, test=None, idx=None, Y=None, w=None):
        self.children = dict()
        self.test = test
        self.leaf = idx
        self.isleaf = True if test is None else False

        # make leaf for every node.
        #+see details in _predict()
        self.make_leaf(idx, Y, w)

    def make_leaf(self, idx, Y, w):
        labels = Y[idx]
        il = np.argsort(labels)
        counter = dict([(labels[il][b], weight(idx[il[b:b + l]], w)) # pairs of (label, weight)
                        for b, l in group(labels[il])])
        self.label = max([(k, v) for k, v in counter.items()], key=lambda x: x[1])[0] # majority label
        tot = sum([v for v in counter.values()])
        self.proba = [counter.get(ll, 0) / tot for ll in np.unique(Y)] # class distr

    def add_child(self, val, node):
        self.children[val] = node


class Tree(object):

    def __init__(self, min_samples_split, node_class=Node, criterion='entropy|gini', costaware=False,
                 asr=False, tradeoff=0):
        self.thr = min_samples_split
        self.Node = node_class
        self.asr = asr # adaptive submodular ranking
        self.tradeoff = tradeoff
        self.criterion = criterion
        self.costaware = costaware

        if self.criterion == 'entropy':
            self.measure = entropy
        if self.criterion == 'gini':
            self.measure = gini

    def is_stop(self, Y, idx):
        if weight(idx, self.w) <= self.thr or \
                np.abs(weight(idx, self.w) - self.thr) <= 1e-7:  # avoid floating error
            return True
        if make_pairs(Y, idx) == 0:
            return True
        return False

    def preproc(self, X, Y, cost, make_weight=False):
        X, Y, w = majority_label(X, Y, make_weight)
        self.tests = np.arange(X.shape[1]) # features
        self.c = dict([(t,c) for t,c in zip(self.tests, cost)]) # cost
        self.Y = Y
        self.X = X
        self.idx = np.arange(self.X.shape[0])
        self.w = w if make_weight else \
            dict([(i, 1 / len(self.idx)) for i in self.idx]) # uniform weight for each obj

        if self.asr:
            self.goal_pair = make_pairs(Y, idx=None)

    def fit(self, X, Y, cost):
        '''Notes:
        Don't change X,Y,self.idx, and only generate new idx to reflect partition.
        '''
        make_weight = True if self.asr else False
        self.preproc(X, Y, cost, make_weight)
        self.tree = self._fit(self.idx, self.tests, self.w, self.c)

    def _fit(self, idx, tests, w, c):
        '''_fit() recursively calls itself.'''
        if self.is_stop(self.Y, idx) or len(tests) == 0:
            return self.Node(idx=idx, Y=self.Y, w=w)

        if self.asr:
            t = self.greedy_asr(idx, tests, w, c)
        else:
            t = self.greedy(idx, tests, w, c)
        cur = self.Node(test=t, idx=idx, Y=self.Y, w=w)
        branches = split_by_test(self.X, idx, t)

        # Recursively expand the current node
        tests = tests[tests != t]
        for v,br in branches:
            child = self._fit(br, tests, w, c)
            cur.add_child(v, child)

        return cur

    def greedy_asr(self, idx, tests, w, c):
        zs = []
        for t in tests:
            branches = split_by_test(self.X, idx, t)
            h = find_biggest_branch(self.Y, branches)

            # 1st term
            z1 = weight(idx, w) - weight(branches[h][1], w)

            # 2nd term
            # sum up each object by branches.
            # two subm funcs are the same within each branch.
            # tho the goal values may be diff.
            gi = lambda i: min(1-w[i], 1-self.thr)
            trun = lambda x: min(x, 1-self.thr)
            fprob_bf = trun(1 - weight(idx, w))
            fprob_af = dict([(v, trun(1-weight(br, w))) for v,br in branches])

            gpair = self.goal_pair
            fpair_bf = gpair - make_pairs(self.Y, idx)
            fpair_af = dict([(v, gpair-make_pairs(self.Y,br)) for v,br in branches])

            # fOR = Q1Q2 - (Q1-f1)(Q2-f2)
            z2 = 0
            for v, br in branches:
                fOR_bf = [gi(i) * gpair - (gi(i) - fprob_bf) * (gpair - fpair_bf)
                          for i in br]
                fOR_af = [gi(i) * gpair - (gi(i) - fprob_af[v]) * (gpair - fpair_af[v])
                          for i in br]
                z2 = z2 + sum([w[i] * (fOR_af[j] - fOR_bf[j]) / (gpair*gi(i) - fOR_bf[j])
                               for j,i in enumerate(br)])

            # 3rd term
            tot = self.measure(self.Y, idx, w)
            ig = tot - sum([weight(br, w) * self.measure(self.Y, br, w) for v,br in branches])
            if np.abs(ig) < 1e-7: # probs could have small floating-point error
                ig = 0
            if ig < 0:
                print('ig',ig)
            assert ig >= 0 # impurity reduction is nonneg
            z3 = self.tradeoff * weight(idx, w) * ig

            zs.append((z1+z2+z3) / c[t])

        t = tests[np.argmax(zs)]
        return t

    def greedy(self, idx, tests, w, c):
        # \max (tot - score) / c
        tot = 0 # tot does not matter here
        scores = [(tot - sum([weight(br, w) * self.measure(self.Y, br, w)
                              for v,br in split_by_test(self.X, idx, t)])) /
                  (c[t] if self.costaware else 1)
                  for t in tests]
        t = tests[np.argmax(scores)]
        return t

    def _predict(self, root, x, proba=False, trace=None):
        '''Recursively trace down a path.'''
        if root.isleaf:
            return root.proba if proba else root.label if trace is None else (root.label, trace)
        if trace is not None:
            trace.append(root.test)

        v = x[root.test]
        # return label of parent when empty branch.
        #+this could happen when given example unseen in training
        if v not in root.children:
            if proba:
                return root.proba
            else:
                return root.label if trace is None else (root.label, trace)

        # recurse
        next = root.children[v]
        return self._predict(next, x, proba=proba, trace=trace)

    def predict_with_tests(self, x):
        '''
        :return: (label, tests)
        '''
        return self._predict(self.tree, x, trace=[])

    def coste(self):
        '''Return the expected cost of the whole tree'''
        return sum([self.w[i] * sum([self.c[t] for t in self.predict_with_tests(self.X[i])[1]]) for i in self.idx])

    def predict(self, X):
        return np.array([self._predict(self.tree, x) for x in X])

    def predict_proba(self, X):
        proba = np.array([self._predict(self.tree, x, proba=True) for x in X])
        return proba[:,1] if proba.shape[1] == 2 else proba
