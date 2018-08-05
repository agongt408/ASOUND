# import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
from sklearn.metrics import roc_auc_score

import argparse
import os

import BiMent
import comparison

parser = argparse.ArgumentParser(description='Get run params')
parser.add_argument('--trials', type=int, nargs='?', default=20,
                    help='num trials per config')
parser.add_argument('--dataset', nargs='?', default='appsByWeek',
                    choices=['appByDay', 'appByWeek', 'bluetoothByDay',
                    'bluetoothByWeek', 'cellByDay', 'cellByWeek'],
                    help='dataset name')
parser.add_argument('--train_all', action='store_true',
                    help='if true, estimate phi values using all items')
args = parser.parse_args()

N_SINGLETONS = 65
N_PAIRS = 5

def convert_to_dict(df):
    data_dict = {}

    unique = list(df.iloc[:, 2].unique())

    date_list = [df.iloc[0, 1]]
    pid_date_list = [(df.iloc[0,0], df.iloc[0, 1])]

    data_dict[int(df.iloc[0,0])] = {}
    data_dict[int(df.iloc[0,0])][df.iloc[0, 1]] = np.zeros((len(unique),))
    data_dict[int(df.iloc[0,0])][df.iloc[0, 1]][unique.index(df.iloc[0, 2])] = 1

    for r in range(1, len(df.index)):
        pid = int(df.iloc[r, 0])
        if pid not in data_dict.keys():
            data_dict[pid] = {}

        if (df.iloc[r, 0], df.iloc[r, 1]) in pid_date_list:
            data_dict[pid][df.iloc[r, 1]][unique.index(df.iloc[r, 2])] = 1 #df.iloc[r, 3]
        else:
            date_list.append(df.iloc[r, 1])
            pid_date_list.append((df.iloc[r, 0], df.iloc[r, 1]))

            data_dict[pid][df.iloc[r, 1]] = np.zeros((len(unique),))
            data_dict[pid][df.iloc[r, 1]][unique.index(df.iloc[r, 2])] = 1

    return data_dict, unique

def dict_to_np_array(d):
    vector_list = []
    gt_pid = []
    idx_list = []
    for pid in d.keys():
        for date in d[pid].keys():
            vector_list.append(d[pid][date])
            gt_pid.append(d.keys().index(pid))
            idx_list.append((pid, date))
    return np.stack(vector_list), np.array(gt_pid), idx_list

def gen_data(x_train, y_train, N_SINGLETONS, N_PAIRS, idx_list):
    singletons = np.random.choice(range(y_train.max() + 1), N_SINGLETONS, replace=False)

    remaining_idx = []
    for i in np.arange(y_train.max() + 1):
        if i not in singletons:
            remaining_idx.append(i)

    pairs = np.random.choice(remaining_idx, N_PAIRS, replace=False)

    vec_list = []
    labels = []
    GT_POS_PAIRS = []
    idx_choice = []

    pid_list = data_dict.keys()

    for p in range(len(pairs)):
        pid = pid_list[pairs[p]]
        w1, w2 = np.random.choice(data_dict[pid].keys(), 2)
        app_vec1, app_vec2 = data_dict[pid][w1], data_dict[pid][w2]

        vec_list.append(app_vec1)
        vec_list.append(app_vec2)

        labels.append(p + 1)
        labels.append(p + 1)
        GT_POS_PAIRS.append((2* p, 2* p + 1))

        idx_choice.append(idx_list.index((pid, w1)))
        idx_choice.append(idx_list.index((pid, w2)))

    for s in singletons:
        pid = pid_list[s]
        w = np.random.choice(data_dict[pid].keys(), 1)
        app_vec = data_dict[pid][w[0]]
        vec_list.append(app_vec)
        labels.append(0)
        idx_choice.append(idx_list.index((pid, w)))

    adj_mx = np.array(vec_list)

    return adj_mx, GT_POS_PAIRS, idx_choice

# def eval_auc_base(adj_mx):
#     print 'Estimating phi from test data...'
#     phi = adj_mx.sum(axis=0) / adj_mx.shape[0]
#
#     # Ensure all p_i's are non-zero
#     adj_mx_nonzero = adj_mx.copy()
#     affil_nonzero = np.where(np.logical_and(phi > 0, phi < 1))[0]
#     adj_mx_nonzero = adj_mx_nonzero[:, affil_nonzero]
#
#     # Recalculate phi
#     phi = adj_mx_nonzero.sum(axis=0) / adj_mx_nonzero.shape[0]
#
#     return adj_mx_nonzero, affil_nonzero, phi
#
# def eval_auc_ERGM(adj_mx):
#     fs = adj_mx.sum(axis=1)
#     gs = adj_mx.sum(axis=0)
#     X, Y, X_bak, Y_bak = BiMent.solver(fs, gs, tolerance=1e-5, max_iter=5000)
#     phi_ia = X[:, None] * Y / (1 + X[:, None] * Y)
#
#     if X_bak is None and Y_bak is None:
#         converge = False
#     else:
#         converge = True
#
#     return adj_mx, phi_ia, converge

# Apps By Week
data_root = '/Users/tradergllc/rsi/datasets/reality-mining/txt/interim/'

if args.dataset.find('cell') > -1:
    if args.dataset == 'cellByDay':
        path = os.path.join(data_root, 'cellTowersByDay.txt')
    elif args.dataset == 'cellByWeek':
        path = os.path.join(data_root, 'cellTowersByWeek.txt')
    else:
        raise ValueError, 'Dataset not selected.'

    df = pd.read_csv(path, sep=', ', header=None, engine='python', dtype=str,
        names=['id', 'date', 'tower', 'tower2', 'freq'])
    df['tower'] = df['tower'].map(str) + df['tower2']
    df = df.drop(columns=['tower2'])

else:
    if args.dataset == 'appByDay':
        path = os.path.join(data_root, 'appsByDay.txt')
    elif args.dataset == 'appByWeek':
        path = os.path.join(data_root, 'appsByWeek-clean.txt')
    elif args.dataset == 'bluetoothByDay':
        path = os.path.join(data_root, 'bluetoothDevicesByDay.txt')
    elif args.dataset == 'bluetoothByWeek':
        path = os.path.join(data_root, 'bluetoothDevicesByWeek.txt')
    else:
        raise ValueError, 'Dataset not selected.'

    df = pd.read_csv(path, sep=', ', header=None, engine='python', dtype=str,
        names=['id', 'date', 'app', 'freq'])

data_dict, unique = convert_to_dict(df)
x_train, y_train, idx_list = dict_to_np_array(data_dict)

print 'DATASET' , args.dataset
print x_train.shape, y_train.shape

aic_base_list = []
aic_ergm_list = []

n_converge = 0

for t in range(args.trials):
    adj_mx, GT_POS_PAIRS, idx_choice = gen_data(
        x_train, y_train, N_SINGLETONS, N_PAIRS, idx_list)

    adj_mx_nonzero, affil_nonzero, phi = comparison.eval_auc_base(adj_mx)
    adj_mx_ergm, phi_ia, converge = comparison.eval_auc_ERGM(adj_mx)

    print phi.shape

    # choice = np.random.choice(range(len(phi)), int(1.0*(len(phi))), replace=False)
    # choice_ergm = affil_nonzero[choice]

    # print np.any(adj_mx[:, choice_ergm] == adj_mx_nonzero[:, choice])

    L_base = adj_mx_nonzero * phi + (1-adj_mx_nonzero)*(1-phi)
    L_ergm = adj_mx * phi_ia + (1-adj_mx)*(1-phi_ia)

    aic_base_list.append(-np.sum(np.log(L_base)))
    aic_ergm_list.append( -np.sum(np.log(L_ergm)))

    if converge:
        n_converge += 1

print 'base' , np.mean(aic_base_list), np.std(aic_base_list)
print 'ergm' , np.mean(aic_ergm_list), np.std(aic_ergm_list)
