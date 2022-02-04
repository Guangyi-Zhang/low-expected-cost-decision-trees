import numpy as np

from util import majority_label, group, \
    weight, split_by_test, find_biggest_branch, \
    make_pairs, make_nonexc
from impurity import entropy, gini, impurepairs, nonexcluded


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

    def isrealleaf(self):
        '''Attribute `isleaf` is a soft mark for a leaf used in prediction'''
        return len(self.children) == 0

    def add_child(self, val, node):
        self.children[val] = node


class Tree(object):

    def __init__(self, min_samples_split_rate, min_samples_split=2,
                 node_class=Node, criterion='entropy', costaware=False,
                 binary=False,
                 asr=False, subm='pair', tradeoff=0):
        self.thr = min_samples_split_rate
        self.min_samples_split = min_samples_split
        self.Node = node_class
        self.asr = asr # adaptive submodular ranking
        self.subm = subm # submodular function chosen, impure "pair" or "excluded" objects in other classes
        self.tradeoff = tradeoff
        self.binary = binary # binary balance split
        self.criterion = criterion
        self.costaware = costaware

        if self.criterion == 'entropy':
            self.measure = entropy
        if self.criterion == 'gini':
            self.measure = gini
        if self.criterion == 'subm' and self.subm == 'pair':
            self.measure = impurepairs
        if self.criterion == 'subm' and self.subm == 'excluded':
            self.measure = nonexcluded

    def is_stop(self, Y, idx):
        w = weight(idx, self.w)
        if w <= self.thr or np.abs(w - self.thr) <= 1e-7: # avoid floating error
            return True
        if len(idx) <= self.min_samples_split:  # TODO: min_samples_split > 1 changes the subm func
            return True
        if make_pairs(Y, idx) == 0:
            return True
        return False

    def preproc(self, X, Y, cost):
        X, Y, w = majority_label(X, Y)
        self.tests = np.arange(X.shape[1]) # features
        if cost is None:
            cost = np.ones(X.shape[1])
        self.c = dict([(t,c) for t,c in zip(self.tests, cost)]) # cost
        self.Y = Y
        self.X = X
        self.idx = np.arange(self.X.shape[0])
        self.w = w

        self.goal_pair = None
        self.goal_exc = None
        if self.subm == 'pair':
            self.goal_pair = make_pairs(Y, idx=None)
        if self.subm == 'excluded':
            self.goal_exc = make_nonexc(self.Y)

    def fit(self, X, Y, cost=None):
        '''Notes:
        Don't change X,Y,self.idx, and only generate new idx to reflect partition.
        '''
        self.preproc(X, Y, cost)
        self.tree = self._fit(self.idx, self.tests, self.w, self.c)

    def _fit(self, idx, tests, w, c):
        '''_fit() recursively calls itself.'''
        if self.is_stop(self.Y, idx) or len(tests) == 0:
            return self.Node(idx=idx, Y=self.Y, w=w)

        if self.asr:
            t = self.greedy_asr(idx, tests, w, c)
        elif self.binary:
            t = self.greedy_balance(idx, tests, w, c)
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

    def asr2(self, idx, w, branches):
        '''
        The 2nd term of ASR greedy rule
        '''
        RKEY = 'root'
        # 1st subm func
        trun = lambda x: min(x, 1)  # truncate function
        wt = dict()
        wt[RKEY] = weight(idx, w)
        for v, br in branches:
            wt[v] = weight(br, w)
        f1 = lambda i,v,idx: trun((1 - wt[v]) / (1 - max(w[i], self.thr)))

        # 2nd subm func and OR operation
        if self.goal_pair is not None:
            pairs = dict()
            pairs[RKEY] = make_pairs(self.Y, idx)
            for v, br in branches:
                pairs[v] = make_pairs(self.Y, br)
            f2 = lambda i,v,idx: (self.goal_pair - pairs[v]) / self.goal_pair
        if self.goal_exc is not None:
            l = lambda i: self.Y[i]  # label of i
            g2 = lambda i: self.goal_exc[l(i)]
            exc = dict()
            exc[RKEY] = make_nonexc(self.Y[idx])
            for v, br in branches:
                exc[v] = make_nonexc(self.Y[br])
            f2 = lambda i,v,idx: (g2(i) - exc[v][l(i)]) / g2(i)

        # fOR = 1 - (1-f1)(1-f2)
        fOR_bf = lambda i,idx: 1 - (1 - f1(i,RKEY,idx)) * (1 - f2(i,RKEY,idx))
        fOR_af = lambda i,v,br: 1 - (1 - f1(i,v,br)) * (1 - f2(i,v,br))

        z2 = 0
        for v, br in branches:
            z2 = z2 + sum([w[i] * (fOR_af(i,v,br) - fOR_bf(i,idx)) / (1 - fOR_bf(i,idx))
                           for _, i in enumerate(br)])
        return z2

    def greedy_asr(self, idx, tests, w, c):
        zs = []
        for t in tests:
            branches = split_by_test(self.X, idx, t)
            h = find_biggest_branch(self.Y, branches)

            # 1st term
            z1 = weight(idx, w) - weight(branches[h][1], w)

            # 2nd term
            z2 = self.asr2(idx, w, branches)

            # 3rd term
            if self.tradeoff == 0:
                z3 = 0
            else:
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
        tot = weight(idx, w) * self.measure(self.Y, idx, w) # tot matters when divided by cost
        scores = [(tot - sum([weight(br, w) * self.measure(self.Y, br, w)
                              for v,br in split_by_test(self.X, idx, t)])) /
                  (c[t] if self.costaware else 1)
                  for t in tests]
        t = tests[np.argmax(scores)]
        return t

    def greedy_balance(self, idx, tests, w, c):
        # Choose the test that minimizes the largest branch-wise weight difference.
        scores = []
        for t in tests:
            ws = [weight(br, w) for v,br in split_by_test(self.X, idx, t)]
            scores.append(max(ws) - min(ws))
        t = tests[np.argmin(scores)]
        return t

    def ccp(self, alpha):
        '''Cost-complexity pruning'''
        self._ccp_reset(self.tree)
        self._ccp(self.tree, alpha)

    def _ccp_leaves(self, root):
        if root.isrealleaf():
            imp = weight(root.leaf, self.w) * self.measure(self.Y, root.leaf, self.w)
            return (imp, 1)

        imp_tot, nleaf_tot = 0, 0
        for v,child in root.children.items():
            imp, nleaf = self._ccp_leaves(child)
            imp_tot = imp_tot + imp
            nleaf_tot = nleaf_tot + nleaf
        return (imp_tot, nleaf_tot)

    def _ccp(self, root, alpha):
        imp = weight(root.leaf, self.w) * self.measure(self.Y, root.leaf, self.w)
        imp_leaf, nleaf = self._ccp_leaves(root)
        if imp + alpha < imp_leaf + alpha * nleaf:
            root.isleaf = True
        else:
            for v,child in root.children.items():
                self._ccp(child, alpha)

    def _ccp_reset(self, root):
        if len(root.children) == 0:
            root.isleaf = True
        else:
            root.isleaf = False
            for v,child in root.children.items():
                self._ccp_reset(child)

    def _predict(self, root, x, proba=False, trace=None):
        '''Recursively trace down a path.'''
        if root.isleaf:
            if proba:
                return root.proba if trace is None else (root.proba, trace)
            else:
                return root.label if trace is None else (root.label, trace)
        if trace is not None:
            trace.append(root.test)

        v = x[root.test]
        # return label of parent when empty branch.
        #+this could happen when given example unseen in training
        if v not in root.children:
            if proba:
                return root.proba if trace is None else (root.proba, trace)
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
        path_lens = [self.w[i] * sum([self.c[t] for t in self.predict_with_tests(self.X[i])[1]]) for i in self.idx]
        ret = sum(path_lens)
        return ret if type(ret) == float else ret.item()

    def predict(self, X):
        return np.array([self._predict(self.tree, x) for x in X])

    def predict_proba(self, X):
        proba = np.array([self._predict(self.tree, x, proba=True) for x in X])
        return proba[:,1] if proba.shape[1] == 2 else proba
