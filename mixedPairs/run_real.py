import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser(description='Get run params')
# parser.add_argument('--n_total', type=int, nargs='*', default=[75],
#                     help='num total items')
# parser.add_argument('--n_pos_pairs', type=int, nargs='*', default=[5],
#                     help='num positive pairs')
# parser.add_argument('--n_affil', type=int, nargs='*', default=[100],
#                     help='num affiliations')
# parser.add_argument('--s_gen', type=float, nargs='*', default=[0.2],
#                     help='similarity of generated entities 0-1')
parser.add_argument('--s_test', type=float, nargs='*', default=[0.2],
                    help='similarity during testing 0-1')
# parser.add_argument('--n_trials', type=int, nargs='?', default=5,
#                     help='num trials per config')
# parser.add_argument('--p', type=float, nargs='?', default=None,
#                     help='phi, if None then p sampled from normal dist, ' + \
#                         'else p is constant')
args = parser.parse_args()

NUM_TOTAL_ITEMS = 75
NUM_POSITIVE_PAIRS = 5
# NUM_AFFIL = 100
# S_PARAM = 0.2

# print args.n_total, args.n_pos_pairs , args.n_affil, args.s_gen, \
#     args.s_test, args.n_trials, args.p


def run_auc(adj_mx, phi, s_test, GT_POS_PAIRS, time_it=True):
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


GT_POS_PAIRS = np.arange(NUM_POSITIVE_PAIRS * 2).reshape((-1,2))
GT_POS_PAIRS = [tuple(x) for x in GT_POS_PAIRS]
print GT_POS_PAIRS

for s in args.s_test:
    auc_list = []
    for i in range(1, 11):
        dir = 'allPairs-appsByDay'
        df_adj_mx = pd.read_csv('~/rsi/datasets/py/%s/%s%d.csv' % (dir, dir, i))
        df_adj_mx = df_adj_mx.drop(columns=[df_adj_mx.columns[0]])
        adj_mx = np.array(df_adj_mx, dtype=np.int32)
        phi = np.array(pd.read_csv(
            '~/rsi/datasets/py/%s/%s%d.phi.csv' % (dir, dir, i)).iloc[:, 1])

        auc = run_auc(adj_mx, phi, s, GT_POS_PAIRS)
        print 's=%.2f , auc=%f' % (s, auc)
        auc_list.append(auc)

    print 's=%.2f , AVERAGE AUC=%f' % ( s, np.mean(auc_list))


# bins = np.linspace(LLR_arr.min(), LLR_arr.max(), 100)
# plt.hist(LLR_arr[LLR_idx], bins, alpha=0.5, label='pos')
# plt.hist(LLR_arr[LLR_idx == False], bins, alpha=0.5, label='neg')
# plt.legend(loc='upper right')
# plt.show()
