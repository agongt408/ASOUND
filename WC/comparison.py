import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine

import BiMent

def jaccard(X,Y):
    n11 = (X * Y).sum()
    n10 = np.abs(X - Y).sum()
    # print n11, n10

    return float(n11) / (n11 + n10 + 1e-4)

def hamming(X,Y):
    return -np.abs(X - Y).sum()

def adamic_adar(X,Y, adj_mx):
    affil_freq = adj_mx.sum(axis=0)
    where_11 = np.where(X * Y == 1)[0]
    score = 1.0 / np.log(affil_freq[where_11])
    return score.sum()

def cosine_idf(X,Y, adj_mx):
    phi = adj_mx.sum(axis=0) / float(adj_mx.shape[0])
    where_nonzero = np.where(np.logical_and(phi > 0, phi < 1))[0]
    adj_mx = adj_mx[:, where_nonzero]
    affil_prob = adj_mx.sum(axis=0) / float(adj_mx.shape[0])

    idf_weights = np.log(1 / affil_prob)
    X_weight = X[where_nonzero] * idf_weights + 1e-4
    Y_weight = Y[where_nonzero] * idf_weights + 1e-4

    # print np.abs(X_weight).max(), np.abs(Y_weight).max()
    # if np.isnan(X_weight).any():
    #     print 'X'
    # if np.isnan(Y_weight).any():
    #     print 'Y'
    score = 1 - cosine(X_weight, Y_weight)
    return score

def eval_method(adj_mx, GT_POS_PAIRS, method='jaccard'):
    start = time()

    score_arr = []
    score_idx = []
    for i in np.arange(adj_mx.shape[0]):
        for j in np.arange(i + 1, adj_mx.shape[0]):
            X = adj_mx[i]
            Y = adj_mx[j]

            if method == 'jaccard':
                score = jaccard(X, Y)
            elif method == 'hamming':
                score = hamming(X, Y)
            elif method == 'adamic_adar':
                score = adamic_adar(X, Y, adj_mx)
            elif method == 'cosine_idf':
                score = cosine_idf(X, Y, adj_mx)
            else:
                raise ValueError , 'method %s not supported' % method

            score_arr.append(score)

            if (i, j) in GT_POS_PAIRS:
                score_idx.append(True)
            else:
                score_idx.append(False)

    # print np.max(score_arr)
    # print np.isnan(score_arr).any()
    auc = roc_auc_score(y_true=score_idx, y_score=score_arr)
    return auc, time() - start

def eval_auc_base(adj_mx):
    print 'Estimating phi from test data...'
    phi = adj_mx.sum(axis=0) / adj_mx.shape[0]

    # Ensure all p_i's are non-zero
    adj_mx_nonzero = adj_mx.copy()
    affil_nonzero = np.where(np.logical_and(phi > 0, phi < 1))[0]
    adj_mx_nonzero = adj_mx_nonzero[:, affil_nonzero]

    # Recalculate phi
    phi = adj_mx_nonzero.sum(axis=0) / adj_mx_nonzero.shape[0]

    return adj_mx_nonzero, affil_nonzero, phi

def eval_auc_ERGM(adj_mx):
    fs = adj_mx.sum(axis=1)
    gs = adj_mx.sum(axis=0)
    X, Y, X_bak, Y_bak = BiMent.solver(fs, gs, tolerance=1e-5, max_iter=5000)
    phi_ia = X[:, None] * Y / (1 + X[:, None] * Y)

    if X_bak is None and Y_bak is None:
        converge = False
    else:
        converge = True

    return adj_mx, phi_ia, converge
