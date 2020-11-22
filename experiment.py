import numpy as np
import pandas as pd
from functools import partial
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, balanced_accuracy_score, confusion_matrix
roc_auc_score = partial(roc_auc_score, multi_class='ovo', average='macro')
from sklearn.preprocessing import LabelBinarizer

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from util import mongo_uri, db_name
from tree import Tree


def load_data(fn, rn, dup_pert, dup_times, log):
    with open(fn, 'rb') as f:
        X, Y = pickle.load(f)
        n = X.shape[0]

        # Categorical into ordinal
        for col in X.select_dtypes('category').columns:
            ncat = len(X[col].cat.categories.tolist())
            X[col] = X[col].cat.codes.replace(-1, ncat) # NaN has a code of -1

        # Create duplicated rows
        if dup_times > 1:
            Xdups, Ydups = [], []
            rand = np.random.randint(1, dup_times, size=n)
            irand = np.random.choice(np.arange(n), int(n*dup_pert), replace=False)
            for i in irand:
                Xdups = Xdups + [X.loc[i, :]] * rand[i]
                Ydups = Ydups + [Y.loc[i]] * rand[i]
            X = X.append(Xdups, ignore_index=True)
            Y = Y.append([pd.Series(Ydups)], ignore_index=True)

        # Fit LabelBinarizer
        lb = LabelBinarizer()
        lb.fit(Y.values)

        # Split
        Xtr, Xt, Ytr, Yt = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=rn)

        log('n', X.shape[0])
        log('nvar', X.shape[1])
        log('ncls', len(Y.unique()))
    return Xtr.values, Xt.values, Ytr.values, Yt.values, lb


def new_exp(interactive=True):
    ex = Experiment('jupyter_ex', interactive=interactive)
    # ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
    if mongo_uri is not None and db_name is not None:
        ex.observers.append(MongoObserver(url=mongo_uri, db_name=db_name))

    @ex.config
    def my_config():
        dup_pert, dup_times = 0.1, 1

    @ex.main
    def my_main(_run, ver, iter, dataset, rn,
                costtype, min_samples_split, criterion, prefix, tradeoff,
                dup_pert, dup_times):
        def log_run(key, val):
            _run.info[key] = val
        from logger import log
        if mongo_uri is not None and db_name is not None:
            # log = _run.log_scalar # for numerical series
            log = log_run

        dup_pert = 1 / dup_times # to not inc data size on average
        np.random.seed(rn)

        fn = 'dataset/{}-bin5.pkl'.format(dataset)
        X, Xt, Y, Yt, lb = load_data(fn, np.random.randint(42), dup_pert, dup_times, log)

        if costtype == 'unit':
            cost = np.ones(Xt.shape[1])
        if costtype == 'random':
            cost = np.random.randint(1, 10 + 1, size=Xt.shape[1])

        costaware = prefix == 'c'
        asr = prefix.startswith('e') # 'e' and 'e0'
        tradeoff = 0 if prefix=='e0' else tradeoff
        tr = Tree(min_samples_split=min_samples_split, criterion=criterion, costaware=costaware, asr=asr, tradeoff=tradeoff)
        tr.fit(X, Y, cost)
        Yp = tr.predict_proba(Xt)
        Yt = lb.transform(Yt)
        log('auc', roc_auc_score(Yt, Yp).item())
        log('cost', tr.coste().item())

    return ex
