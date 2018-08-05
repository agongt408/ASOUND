import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser(description='Get run params')
parser.add_argument('--n_total', type=int, nargs='*', default=[75],
                    help='num total items')
parser.add_argument('--n_pos_pairs', type=int, nargs='*', default=[5],
                    help='num positive pairs')
parser.add_argument('--n_affil', type=int, nargs='*', default=[100],
                    help='num affiliations')
parser.add_argument('--s_gen', type=float, nargs='*', default=[0.2],
                    help='similarity of generated entities 0-1')
parser.add_argument('--s_test', type=float, nargs='*', default=[0.2],
                    help='similarity during testing 0-1')
parser.add_argument('--n_trials', type=int, nargs='?', default=5,
                    help='num trials per config')
parser.add_argument('--p', type=float, nargs='?', default=None,
                    help='phi, if None then p sampled from normal dist, ' + \
                        'else p is constant')
args = parser.parse_args()

# NUM_TOTAL_ITEMS = 75
# NUM_POSITIVE_PAIRS = 5
# NUM_AFFIL = 100
# S_PARAM = 0.2

print args.n_total, args.n_pos_pairs , args.n_affil, args.s_gen, \
    args.s_test, args.n_trials, args.p

def get_phi(shape, constant=None):
    if constant:
        assert constant >= 0 and constant <= 1
        return np.ones(shape)* constant
    return np.random.random(shape)

def bernoulli(phi):
    return (np.random.random(phi.shape) < phi).astype(np.int32)

def run_auc(n_total, n_pos_pairs, s_gen, s_test, n_affil, p=None, time_it=True):
    phi = get_phi(n_affil, p)

    GT_POS_PAIRS = []

    adj_mx = []
    it = 0

    assert n_total - 2*n_pos_pairs >= 0, 'n_total - 2*n_pos_pairs < 0'

    for _ in range(n_total - 2*n_pos_pairs):
        X1 = bernoulli(phi)
        adj_mx.append(X1)
        it += 1
    for _ in range(n_pos_pairs):
        X1 = bernoulli(phi)
        mask_similar = bernoulli(s_gen * np.ones(phi.shape))
        X2 = mask_similar* X1 + (1 - mask_similar)*bernoulli(phi)
        adj_mx.append(X1)
        adj_mx.append(X2)

        GT_POS_PAIRS.append((it, it + 1))
        it += 2

    adj_mx = np.stack(adj_mx)

    start = time()

    LLR_arr = []
    LLR_idx = []
    for i in np.arange(adj_mx.shape[0]):
        for j in np.arange(i + 1, adj_mx.shape[0]):
            n_11 = (adj_mx[i] * adj_mx[j] == 1).astype(np.int32)
            n_10 = (adj_mx[i] != adj_mx[j]).astype(np.int32)
            n_00 = (adj_mx[i] + adj_mx[j] == 0).astype(np.int32)

            assert (n_11 + n_10 + n_00 == 1).all()

            LLR = (n_11* np.log((s_test + (1 - s_test) * phi) / phi)).sum()
            LLR += (n_10* np.log(1 - s_test)).sum()
            LLR += (n_00* np.log(((1- phi + phi * s_test) / (1-phi)))).sum()
            LLR_arr.append(LLR)

            if (i, j) in GT_POS_PAIRS:
                LLR_idx.append(True)
            else:
                LLR_idx.append(False)

    if time:
        print 'time elapsed: %f s' % (time() - start)

    LLR_arr = np.array(LLR_arr)
    LLR_idx = np.array(LLR_idx)
    auc = roc_auc_score(y_true=LLR_idx, y_score=LLR_arr)

    return auc

for n_total in args.n_total:
    for n_pos_pairs in args.n_pos_pairs:
        for n_affil in args.n_affil:
            for s_gen in args.s_gen:
                for s_test in args.s_test:
                    auc_list = []
                    for t in range(args.n_trials):
                        auc = run_auc(n_total, n_pos_pairs, s_gen, s_test,
                            n_affil, args.p)
                        auc_list.append(auc)
                        print ('n_total=%d, n_pos_pairs=%d, s_gen=%.2f, ' + \
                            's_test=%.2f, n_affil=%d, auc=%f') % \
                            (n_total, n_pos_pairs, s_gen, s_test, n_affil, auc)
                    print ('n_total=%d, n_pos_pairs=%d, s_gen=%.2f, ' + \
                            's_test=%.2f, n_affil=%d, AVERAGE_AUC=%f') % \
                        (n_total, n_pos_pairs, s_gen, s_test, n_affil,
                            np.mean(auc_list))


# bins = np.linspace(LLR_arr.min(), LLR_arr.max(), 100)
# plt.hist(LLR_arr[LLR_idx], bins, alpha=0.5, label='pos')
# plt.hist(LLR_arr[LLR_idx == False], bins, alpha=0.5, label='neg')
# plt.legend(loc='upper right')
# plt.show()
