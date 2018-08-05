# import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from sklearn.metrics import roc_auc_score
from scipy.io import mmread

import argparse
import os

import BiMent
import comparison

parser = argparse.ArgumentParser(description='Get run params')
parser.add_argument('--trials', type=int, nargs='?', default=20,
                    help='num trials per config')
parser.add_argument('--dataset', nargs='?', default='dem110',
                    help='dataset name')
args = parser.parse_args()

##########################################################
# Run experiment

aic_base_list = []
aic_ergm_list = []

n_converge = 0

for t in range(args.trials):
    ROOT = '/Users/tradergllc/rsi/datasets/congress/csv/test/'
    adj_mx = mmread(os.path.join(ROOT, args.dataset, 'data%d.mtx' % (t+ 1))).toarray()
    with open(os.path.join(ROOT, args.dataset, 'numPositivePairs%d.txt' % (t + 1)), 'r') as f:
        N_PAIRS = int(f.read())

    adj_mx_nonzero, affil_nonzero, phi = comparison.eval_auc_base(adj_mx)
    adj_mx_ergm, phi_ia, converge = comparison.eval_auc_ERGM(adj_mx)

    print phi.shape

    L_base = adj_mx_nonzero * phi + (1-adj_mx_nonzero)*(1-phi)
    L_ergm = adj_mx * phi_ia + (1-adj_mx)*(1-phi_ia)

    aic_base_list.append(-np.sum(np.log(L_base)))
    aic_ergm_list.append( -np.sum(np.log(L_ergm)))

    if converge:
        n_converge += 1


print 'DATASET' , args.dataset

print 'base' , np.mean(aic_base_list), np.std(aic_base_list)
print 'ergm' , np.mean(aic_ergm_list) , np.std(aic_ergm_list)

print '#####' * 10
