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

    # WC_arr = np.array(WC_arr)
    # WC_idx = np.array(WC_idx)

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
        # assert X.shape[0] == adj_mx.shape[0], \
        #     'X and adj_mx must have same dim1 shape but have shapes %d and %d' \
        #     % (X.shape[0], adj_mx.shape[0])
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

    # WC_arr = np.array(WC_arr)
    # WC_idx = np.array(WC_idx)

    auc = roc_auc_score(y_true=WC_idx, y_score=WC_arr)

    if X_bak is None and Y_bak is None:
        return auc, time() - start, False

    return auc, time() - start, True

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

print x_train.shape, y_train.shape

if args.train_all:
    print 'Estimating phi and X, Y values from entire data...'

    # Estimate phi values for baseline WC
    phi = x_train.sum(axis=0) / x_train.shape[0]

    # Estimate X, Y values for ERGM WC
    fs = x_train.sum(axis=1)
    gs = x_train.sum(axis=0)
    X, Y, _, _ = BiMent.solver(fs, gs, tolerance=1e-5, max_iter=10000)
else:
    phi = None
    X = None
    Y = None

auc_base_list = []
auc_ergm_list = []
auc_ergm_first_order_list = []
# auc_base_all_list = []
# auc_ergm_all_list = []

auc_jaccard_list = []
auc_hamming_list = []
auc_adamic_list = []
auc_cosidf_list = []

n_converged = 0

for t in range(args.trials):
    adj_mx, GT_POS_PAIRS, idx_choice = gen_data(
        x_train, y_train, N_SINGLETONS, N_PAIRS, idx_list)

    auc_base, time_elapsed = eval_auc_base(adj_mx, GT_POS_PAIRS)
    print 'Trial=%d, Base WC: time=%fs, auc=%f' % (t, time_elapsed, auc_base)

    auc_ergm, time_elapsed, converge = eval_auc_ERGM(adj_mx, GT_POS_PAIRS)
    print 'Trial=%d, ERGM WC: time=%fs, auc=%f' % (t, time_elapsed, auc_ergm)

    auc_first_order, time_elapsed, _ = eval_auc_ERGM(adj_mx, GT_POS_PAIRS, True)
    print 'Trial=%d, ERGM FIRST ORDER WC: time=%fs, auc=%f' \
        % (t, time_elapsed, auc_first_order)

    # auc_base_all, _ = eval_auc_base(adj_mx, GT_POS_PAIRS, phi=phi)
    # print 'Trial=%d, Base WC ALL: auc=%f' % (t, auc_base_all)
    #
    # auc_ergm_all, _, _ = eval_auc_ERGM(
    #     adj_mx, GT_POS_PAIRS, X=X, Y=Y, idx_choice=idx_choice)
    # print 'Trial=%d, ERGM WC ALL: auc=%f' % (t, auc_ergm_all)

    # if converge:
    auc_base_list.append(auc_base)
    auc_ergm_list.append(auc_ergm)
    auc_ergm_first_order_list.append(auc_first_order)
    # auc_base_all_list.append(auc_base_all)
    # auc_ergm_all_list.append(auc_ergm_all)

    if converge:
        n_converged += 1


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

# print 'AVERAGE AUC BASE=%f, AVERAGE AUC ERGM=%f, AVERAGE AUC ERGM 1st order=%f' % \
#     (np.mean(auc_base_list), np.mean(auc_ergm_list), np.mean(auc_ergm_first_order_list))

# print 'AVERAGE AUC BASE ALL=%f, AVERAGE AUC ERGM ALL=%f' % \
#     (np.mean(auc_base_all_list), np.mean(auc_ergm_all_list))

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
