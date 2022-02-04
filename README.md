# Non-overfitting Class Identification

Project structure:

* ``Tree`` in ``tree.py`` is a the main class.
* Parameters are set in ``run.py``.
* Data are collected and preprocessed in ``gen-data.ipynb``.
* Evaluation result is shown in ``result.ipynb``.

Examples

```
from tree import Tree

# Our algorithm
tr = Tree(min_samples_split=0.001, criterion='entropy', asr=True, tradeoff=50)

# ASR
tr = Tree(min_samples_split=0.001, criterion='entropy', asr=True, tradeoff=0)

# C4.5
tr = Tree(min_samples_split=0.001, criterion='entropy', asr=False, tradeoff=0)
# Cost-benefit C4.5
tr = Tree(min_samples_split=0.001, criterion='entropy', costaware=True, asr=False, tradeoff=0)

# X, Y as numpy array
cost = np.ones(X.shape[1])
tr.fit(X, Y, cost) # or just tr.fit(X, Y)
Yp = tr.predict_proba(X)
```


To run the Iris dataset:

```
>>> from sklearn import datasets
>>> import numpy as np
>>> import pandas as pd
>>> from tree import Tree

>>> name = 'iris'
>>> data = datasets.fetch_openml(name=name)
/Users/janfan/.pyenv/versions/miniconda3-latest/lib/python3.8/site-packages/sklearn/datasets/_openml.py:404: UserWarning: Multiple active versions of the dataset matching the name iris exist. Versions may be fundamentally different, returning version 1.
  warn("Multiple active versions of the dataset matching the name"
>>> data.data = pd.DataFrame(data.data, columns=data.feature_names)
>>> data.target = pd.Series(data.target)
>>> for c in data.data.columns:
...     data.data[c] = pd.qcut(data.data[c], q=5)
>>> X = pd.get_dummies(data.data).to_numpy()
>>> X[0:3]
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2]])
>>> Y = data.target.cat.codes.to_numpy()
>>> Y[0:3]
array([0, 0, 0], dtype=int8)

>>> tr = Tree(min_samples_split_rate=0.02,
...           criterion='entropy', # or 'gini'
...           asr=True,
...           tradeoff=10)
>>> tr.fit(X, Y)
>>> Yp = tr.predict_proba(X)
>>> Yp[0:3]
array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.]])
```

## License

This project is licensed under the terms of the MIT license.
