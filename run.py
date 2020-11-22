import numpy as np
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=4)

from experiment import new_exp


def run(ver, it, rn, dataset):
    ex = new_exp(interactive=False)
    nsplit = {
        'iris': 0.02,
    }

    kv = [
        ('ver', [ver]),
        ('iter', [it]),
        ('dataset', [dataset]),
        #('criterion', ['gini', 'entropy']),
        ('criterion', ['entropy']),
        ('costtype', ['unit']),
        ('prefix', ['e', 'e0', '']),
        ('tradeoff', [50]),
        ('rn', [rn])
    ]

    ks, vs = zip(*kv)
    for alls in itertools.product(*list(vs)):
        conf = dict([(k, v) for k,v in zip(list(ks),alls)])
        if 'min_samples_split' not in conf:
            conf['min_samples_split'] = nsplit[conf['dataset']]
        pp.pprint(conf)
        r = ex.run(config_updates=conf)

if __name__ == '__main__':
    import sys
    ver = int(sys.argv[1])
    it = int(sys.argv[2])
    rn = int(sys.argv[3])
    dataset = sys.argv[4]
    run(ver, it, rn, dataset)
