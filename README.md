# Non-overfitting Class Identification


To run the example dataset, Iris:

```
$ bash run.sh
```

Project structure:

* ``Tree`` in ``tree.py`` is a the main class.
* Parameters are set in ``run.py``.
* Data are preprocessed in ``gen-data.ipynb``.
* Evaluation result is shown in ``result.ipynb``.

Examples

```
from tree import Tree

# Our algorithm
tr = Tree(min_samples_split=0.001, criterion='entropy', costaware=False, asr=True, tradeoff=50)

# ASR
tr = Tree(min_samples_split=0.001, criterion='entropy', costaware=False, asr=True, tradeoff=0)

# C4.5
tr = Tree(min_samples_split=0.001, criterion='entropy', costaware=False, asr=False, tradeoff=0)
# Cost-benefit C4.5
tr = Tree(min_samples_split=0.001, criterion='entropy', costaware=True, asr=False, tradeoff=0)

cost = np.ones(X.shape[1])
tr.fit(X, Y, cost)
Yp = tr.predict_proba(X)
```


## Database

In case you wonder, this repo adopts [sacred](https://github.com/IDSIA/sacred) and [incense](https://github.com/JarnoRFB/incense) to manage experiment results in MongoDB.
An example to set up the environments is given below.
However, you don't need them to run the code except for evaluation.
Without Sacred, results will be redirected to stdout.

```
$ pip install dnspython incense sacred 

# a private file: mongodburi.py
mongo_uri = 'mongodb+srv://xxx'
db_name = 'yyy'
```


## Testing
First install unit testing framework ``pip install pytest``, and then run the following to test functionalities of the code.

```
py.test
```


## Datasets

Datasets can be downloaded from the following list.

* http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
* http://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
* http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
* http://archive.ics.uci.edu/ml/datasets/Poker+Hand
* *http://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
* http://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data
* http://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+
* http://archive.ics.uci.edu/ml/datasets/Facebook+Large+Page-Page+Network

