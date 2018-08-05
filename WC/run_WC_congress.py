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

def eval_auc_base(adj_mx, GT_POS_PAIRS, phi=None):
    if phi is None:
        print 'Estimating phi from test data...'
        phi = adj_mx.sum(axis=0) / adj_mx.shape[0]

        # Ensure all p_i's are non-zero
        adj_mx_nonzero = adj_mx.copy()
        adj_mx_nonzero = adj_mx_nonzero[:, np.where(np.logical_and(phi > 0, phi < 1))[0]]

        # Recalculate phi
        phi = adj_mx_nonzero.sum(axis=0) / adj_mx_nonzero.shape[0]
    else:
        assert phi.shape[0] == adj_mx.shape[1], \
            'phi must has same len as number of affiliations'
        assert len(np.where(phi == 0)[0]) == 0 , 'phi must be all nonzero'
        adj_mx_nonzero = adj_mx.copy()

    # %%timeit
    start = time()

    WC_arr = []
    WC_idx = []
    for i in np.arange(adj_mx_nonzero.shape[0]):
        for j in np.arange(i + 1, adj_mx_nonzero.shape[0]):
            X = adj_mx_nonzero[i]
            Y = adj_mx_nonzero[j]

            WC = (1.0/ phi.shape[0])* (X- phi)* (Y- phi)/ (phi*(1-phi))
            WC_arr.append(WC.sum())

            if (i, j) in GT_POS_PAIRS:
                WC_idx.append(True)
            else:
                WC_idx.append(False)

    WC_arr = np.array(WC_arr)
    WC_idx = np.array(WC_idx)

    auc = roc_auc_score(y_true=WC_idx, y_score=WC_arr)
    return auc, time() - start

def eval_auc_ERGM(adj_mx, GT_POS_PAIRS, first_order=False, X=None, Y=None,
                    idx_choice=None):
    if X is None or Y is None or idx_list is None:
        fs = adj_mx.sum(axis=1)
        gs = adj_mx.sum(axis=0)
        X, Y, X_bak, Y_bak = BiMent.solver(fs, gs, tolerance=1e-5, max_iter=5000,
                                            first_order=first_order)
        phi_ia = X[:, None] * Y / (1 + X[:, None] * Y)
    else:
        assert Y.shape[0] == adj_mx.shape[1], \
            'Y and adj_mx must have same dim2 shape but have shapes %d and %d' \
            % (X.shape[0], adj_mx.shape[1])

        # Return that ERGM converged
        X_bak = 1
        Y_bak = 1

        phi_all = X[:, None] * Y / (1 + X[:, None] * Y)
        phi_ia = phi_all[idx_choice]

    start = time()

    WC_arr = []
    WC_idx = []
    for i in np.arange(adj_mx.shape[0]):
        for j in np.arange(i + 1, adj_mx.shape[0]):
            X = adj_mx[i]
            Y = adj_mx[j]
            E_X = phi_ia[i]
            E_Y = phi_ia[j]

            WC = (1.0/ phi_ia.shape[1])* (X- E_X)* (Y- E_Y)/ (np.sqrt(E_X*E_Y*(1-E_X)*(1-E_Y)) + 1e-6)
            WC_arr.append(WC.sum())

            if (i, j) in GT_POS_PAIRS:
                WC_idx.append(True)
            else:
                WC_idx.append(False)

    WC_arr = np.array(WC_arr)
    WC_idx = np.array(WC_idx)

    auc = roc_auc_score(y_true=WC_idx, y_score=WC_arr)

    if X_bak is None and Y_bak is None:
        return auc, time() - start, False

    return auc, time() - start, True

##########################################################
# Run experiment

auc_base_list = []
auc_ergm_list = []
auc_ergm_first_order_list = []

auc_jaccard_list = []
auc_hamming_list = []
auc_adamic_list = []
auc_cosidf_list = []

n_converged = 0

for t in range(args.trials):
    ROOT = '/Users/tradergllc/rsi/datasets/congress/csv/test/'
    adj_mx = mmread(os.path.join(ROOT, args.dataset, 'data%d.mtx' % (t+ 1))).toarray()
    with open(os.path.join(ROOT, args.dataset, 'numPositivePairs%d.txt' % (t + 1)), 'r') as f:
        N_PAIRS = int(f.read())

    GT_POS_PAIRS = [(2*i, 2*i + 1) for i in range(N_PAIRS)]

    auc_base, time_elapsed = eval_auc_base(adj_mx, GT_POS_PAIRS)
    print 'Trial=%d, Base WC: time=%fs, auc=%f' % (t, time_elapsed, auc_base)

    auc_ergm, time_elapsed, converge = eval_auc_ERGM(adj_mx, GT_POS_PAIRS)
    print 'Trial=%d, ERGM WC: time=%fs, auc=%f' % (t, time_elapsed, auc_ergm)

    auc_first_order, time_elapsed, _ = eval_auc_ERGM(adj_mx, GT_POS_PAIRS, True)
    print 'Trial=%d, ERGM FIRST ORDER WC: time=%fs, auc=%f' \
        % (t, time_elapsed, auc_first_order)

    # if converge:
    auc_base_list.append(auc_base)
    auc_ergm_list.append(auc_ergm)
    auc_ergm_first_order_list.append(auc_first_order)

    if converge:
        n_converged += 1


    ###################
    jaccard, _ = comparison.eval_method(adj_mx, GT_POS_PAIRS, 'jaccard')
    print 'Trial=%d, JACCARD: auc=%f' % (t, jaccard)
    hamming, _ = comparison.eval_method(adj_mx, GT_POS_PAIRS, 'hamming')
    print 'Trial=%d, HAMMING: auc=%f' % (t, hamming)
    adamic, _ = comparison.eval_method(adj_mx, GT_POS_PAIRS, 'adamic_adar')
    print 'Trial=%d, ADAMIC: auc=%f' % (t, adamic)
    cosidf, _ = comparison.eval_method(adj_mx, GT_POS_PAIRS, 'cosine_idf')
    print 'Trial=%d, COSINE IDF: auc=%f' % (t, cosidf)

    auc_jaccard_list.append(jaccard)
    auc_hamming_list.append(hamming)
    auc_adamic_list.append(adamic)
    auc_cosidf_list.append(cosidf)

# print 'base:' , auc_base_list
# print 'ergm:', auc_ergm_list
#
# print 'AVERAGE AUC BASE=%f, AVERAGE AUC ERGM=%f, AVERAGE AUC ERGM 1st order=%f' % \
#     (np.mean(auc_base_list), np.mean(auc_ergm_list), np.mean(auc_ergm_first_order_list))
#
# print 'n_converged=%d' % n_converged
#
# print 'avg auc base std: ', np.std(auc_base_list)
# print 'avg auc ergm std: ', np.std(auc_ergm_list)
# print 'avg auc ergm 1st order std: ', np.std(auc_ergm_first_order_list)


print 'DATASET' , args.dataset
print 'n_converged=%d' % n_converged

print 'base: ', np.mean(auc_base_list), np.std(auc_base_list)
print 'ergm: ', np.mean(auc_ergm_list), np.std(auc_ergm_list)
print 'ergm_1st_order: ', np.mean(auc_ergm_first_order_list), np.std(auc_ergm_first_order_list)
# print 'base all: ', np.mean(auc_base_all_list), np.std(auc_base_all_list)
# print 'ergm all: ', np.mean(auc_ergm_all_list) , np.std(auc_ergm_all_list)

# print 'AVERAGE AUC BASE=%f' % np.mean(auc_base_list)

print 'jaccard: ' , np.mean(auc_jaccard_list), np.std(auc_jaccard_list)
print 'hamming: ' , np.mean(auc_hamming_list), np.std(auc_hamming_list)
print 'adamic: ' , np.mean(auc_adamic_list), np.std(auc_adamic_list)
print 'cosidf: ' , np.mean(auc_cosidf_list), np.std(auc_cosidf_list)
