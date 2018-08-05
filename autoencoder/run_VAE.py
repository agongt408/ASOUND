from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from time import time
from scipy.spatial.distance import euclidean
from sklearn.metrics import average_precision_score
from scipy.stats import multivariate_normal as multi_norm
from sklearn.decomposition import PCA

import os
import argparse

parser = argparse.ArgumentParser(description='Get run params')
parser.add_argument('--nu', type=float, nargs='*', default=[0.1],
                    help='std of displacement')
parser.add_argument('--directory', nargs='?', default=None,
                    help='data directory')
parser.add_argument('--data', type=int, nargs='?', default=10,
                    help='data file number')
parser.add_argument('--load_weights', action='store_true',
                    help='load model weights (npy) from current directory')
parser.add_argument('--latent_dim', default=2, type=int,
                    help='number of dim in latent representation')
parser.add_argument('--data_frac', default=1.0, type=float,
                    help='fraction of total items for training')
args = parser.parse_args()

print args.load_weights

def convert_to_dict(df, col='app'):
    data_dict = {}

    unique = list(df[col].unique())

    pid = int(df.iloc[0,0])
    date = df.iloc[0, 1]
    data_dict[pid] = {}
    data_dict[pid][date] = np.zeros((len(unique),))

    for r in range(1, len(df.index)):
        if int(df.iloc[r, 0]) != pid:
            pid = int(df.iloc[r, 0])
            data_dict[pid] = {}
            date = None

        if df.iloc[r, 1] == date:
            try:
                data_dict[pid][date][unique.index(df.iloc[r, 2])] = 1 #df.iloc[r, 3]
            except ValueError:
                pass
        else:
            date = df.iloc[r, 1]
            data_dict[pid][date] = np.zeros((len(unique),))
            try:
                data_dict[pid][date][unique.index(df.iloc[r, 2])] = 1 #int(df.iloc[r, 3])
            except ValueError:
                pass
    return data_dict, unique

def dict_to_np_array(d):
    vector_list = []
    gt_pid = []
    for pid in d.keys():
        for date in d[pid].keys():
            vector_list.append(d[pid][date])
            gt_pid.append(d.keys().index(pid))
    return np.stack(vector_list), np.array(gt_pid)

def get_train_data(path='~/rsi/datasets/proc/reality-mining/txt/interim/appsByWeek-clean.txt',
    remove_near_duplicates=False):
    # Apps By Week
    df = pd.read_csv(path, sep=', ', header=None, engine='python', dtype=str,
                     names=['id', 'date', 'app', 'freq'])

    # Clean app names
    if remove_near_duplicates:
        for r in range(len(df.index)):
            if df.iloc[r, 2].find(':') > -1:
                df.iloc[r, 2] = 'Phone'

    data_dict, unique = convert_to_dict(df)

    x_train, y_train = dict_to_np_array(data_dict)
    print x_train.shape, y_train.shape

    return x_train, y_train, data_dict, unique

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def VAE(original_dim=200, intermediate_dim=100, latent_dim=2):
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(original_dim,), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = categorical_crossentropy(inputs,outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(lr=0.001, decay=1.0e-5))
    # vae.summary()

    # start = time()
    # vae.fit(x_train, epochs=1000, batch_size=batch_size)
    # print 'Total training time: %f s' , time() - start

    return vae, encoder, decoder

def gen_data(data_dict, N_SINGLETONS=75, N_PAIRS=10):
    singletons = np.random.choice(range(y_train.max() + 1), N_SINGLETONS, replace=False)

    remaining_idx = []
    for i in np.arange(y_train.max() + 1):
        if i not in singletons:
            remaining_idx.append(i)

    pairs = np.random.choice(remaining_idx, N_PAIRS, replace=False)

    vec_list = []
    labels = []
    GT_POS_PAIRS = []

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

    for s in singletons:
        pid = pid_list[s]
        w = np.random.choice(data_dict[pid].keys(), 1)
        app_vec = data_dict[pid][w[0]]
        vec_list.append(app_vec)
        labels.append(0)

    return np.array(vec_list), GT_POS_PAIRS, labels

def read_csv(path, unique, num_pairs=5):
    """
    Assumes first '2*num_pairs' rows correspond to GT pairs
    """
    df = pd.read_csv(path, sep=',', engine='python', dtype=str)
    affil_matrix = np.array(df.iloc[:, 1:]).astype(int)

    labels=[]
    GT_POS_PAIRS = []
    for i in range(num_pairs):
        GT_POS_PAIRS.append((2*i, 2*i+1))
        labels.append(i + 1)
        labels.append(i + 1)

    for i in range(affil_matrix.shape[0] - 2*num_pairs):
        labels.append(0)

    test_app_list = list(np.array(df.columns)[1:])

    mapping = []
    for app in unique:
        mapping.append(test_app_list.index(app))

    affil_mtx_aligned = affil_matrix.copy()
    for i in range(len(mapping)):
        affil_mtx_aligned[:, i] = affil_matrix[:, mapping[i]]

    return affil_mtx_aligned, GT_POS_PAIRS, labels

def eval_auc(points, GT_POS_PAIRS, nu, labels=None, const_var=False):
    # if labels is not None:
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='jet')
    #     plt.colorbar()
    #     plt.show()

    N_DIM = len(points.shape)
    test_mean = np.mean(points, axis=0)
    # print 'test_mean:', test_mean
    sigma = np.std(points)
    # print 'sigma axis=0:' , np.std(points, axis=0)
    # print 'sigma:' , sigma

    std = np.std(points, axis=0)
    t = (np.ones(std.shape) * nu) / std
    dis_var = np.diag(t)
    cov = np.cov(points.transpose())

    start = time()

    LR_arr = []
    LR_idx = []

    # dist_matrix = np.zeros((points.shape[0], points.shape[0]))
    adj_matrix = np.zeros((points.shape[0], points.shape[0]))

    for i in np.arange(points.shape[0]):
        for j in np.arange(i + 1, points.shape[0]):
            midpoint = 0.5 * (points[i] + points[j])

            if const_var:
                # Assumes each dim has equal variance
                m = euclidean(midpoint, test_mean)
                d = euclidean(0.5* (points[i] - points[j]), np.zeros(midpoint.shape))

                m_prime = m / sigma
                d_prime = d / sigma
                t = nu / sigma

                LR = np.power(0.5/t, N_DIM)* np.exp(0.5*(np.power(m_prime,2)+ \
                        np.power(d_prime,2)*(2- np.power(t,-2))))
            else:
                # ~6x computation time
                dis = 0.5* (points[i] - points[j])
                pdf_mid = multi_norm.pdf(midpoint, mean=test_mean, cov=cov, allow_singular=True)
                pdf_dis = multi_norm.pdf(dis, mean=np.zeros(dis.shape), cov=dis_var, allow_singular=True)
                pdf_x_i = multi_norm.pdf(points[i], mean=test_mean, cov=cov, allow_singular=True)
                pdf_x_j = multi_norm.pdf(points[j], mean=test_mean, cov=cov, allow_singular=True)
                LR = pdf_mid * pdf_dis / (pdf_x_i*pdf_x_j)

            LR_arr.append(LR)
            # dist_matrix[i, j] = LR

            if (i, j) in GT_POS_PAIRS:
                LR_idx.append(True)
                adj_matrix[i, j] = 1
            else:
                LR_idx.append(False)
                adj_matrix[i, j] = 0

    time_elapsed = time() - start

    LR_arr = np.array(LR_arr)
    LR_idx = np.array(LR_idx)

    auc = roc_auc_score(y_true=LR_idx, y_score=LR_arr)

    # if auc < 0.5:
    #     auc = 1 - auc

    return auc, time_elapsed


def run_auc_base(affil_mtx, GT_POS_PAIRS, s_test=0.1, time_it=True):
    print 'Estimating phi...'

    adj_mx = affil_mtx.copy()
    phi_init = adj_mx.sum(axis=0) / adj_mx.shape[0]
    for i in range(len(phi_init) - 1, -1, -1):
        if phi_init[i] == 0:
            adj_mx = np.delete(adj_mx, i, axis=1)
    phi = adj_mx.sum(axis=0) / adj_mx.shape[0]

    # print adj_mx, phi

    start = time()

    LLR_arr = []
    LLR_idx = []
    for i in np.arange(adj_mx.shape[0]):
        for j in np.arange(i + 1, adj_mx.shape[0]):
            n_11 = (adj_mx[i] * adj_mx[j] == 1).astype(np.int32)
            n_10 = (adj_mx[i] != adj_mx[j]).astype(np.int32)
            n_00 = (adj_mx[i] + adj_mx[j] == 0).astype(np.int32)
            # print n_11 + n_10 + n_00
            assert np.all(n_11 + n_10 + n_00 == 1)

            LLR = (n_11* np.log((s_test + (1 - s_test) * phi) / (phi + 1e-5))).sum()
            LLR += (n_10* np.log(1 - s_test)).sum()
            LLR += (n_00* np.log(((1- phi + phi * s_test) / (1-phi + 1e-5)))).sum()
            LLR_arr.append(LLR)

            if (i, j) in GT_POS_PAIRS:
                LLR_idx.append(True)
            else:
                LLR_idx.append(False)

    if time_it:
        print 'time elapsed: %f s' % (time() - start)

    LLR_arr = np.array(LLR_arr)
    LLR_idx = np.array(LLR_idx)
    auc = roc_auc_score(y_true=LLR_idx, y_score=LLR_arr)

    # if auc < 0.5:
    #     return 1 - auc

    return auc


x_train, y_train, data_dict, unique = get_train_data()
vae, encoder, decoder = VAE(original_dim=len(unique), latent_dim=args.latent_dim)
print encoder.summary()
print decoder.summary()

if args.load_weights:
    print 'Loading model weights...'
    encoder.set_weights(np.load('encoder_weights.npy'))
    decoder.set_weights(np.load('decoder_weights.npy'))
    print 'Model loaded successfully!'
else:
    start = time()
    n_train = x_train.shape[0]
    subset = np.random.choice(range(n_train), int(args.data_frac*n_train), replace=False)
    x_train_subset = x_train[subset]
    print x_train_subset.shape
    vae.fit(x_train_subset, epochs=1000, batch_size=128)
    print 'Total training time: %f s' , time() - start

    np.save('encoder_weights.npy', encoder.get_weights())
    np.save('decoder_weights.npy', decoder.get_weights())

    encoded_imgs = encoder.predict(x_train_subset)
    decoded_imgs = decoder.predict(encoded_imgs[2])

    n = 10  # how many digits we will display
    choice = np.random.choice(range(x_train_subset.shape[0]), n, replace=False)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_train_subset[choice[i], :200].reshape(20, 10))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[choice[i], :200].reshape(20, 10))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    x_test_encoded = encoder.predict(x_train)
    x_test_encoded = np.array(x_test_encoded)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test_encoded[2, :, 0], x_test_encoded[2, :, 1], c=y_train)
    plt.colorbar()
    plt.show()

    mean = np.mean(x_test_encoded[0], axis=0)
    print 'training mean:' , mean
    std = np.std(x_test_encoded[0], axis=0)
    print 'training std:' , std

    for i in range(len(mean)):
        plt.hist(x_test_encoded[0, :, i], bins=100, alpha=0.5)
        # plt.axvline(mean[i], color=i, linestyle='dashed', linewidth=1)

    plt.legend()
    plt.show()

    pca = PCA(n_components=args.latent_dim)
    pca = pca.fit(x_train_subset)


if args.directory is not None and args.data is not None:
    print 'Reading data from file...'

    auc_list = {}
    auc_base_list = {}
    for nu in args.nu:
        auc_list[nu] = []
        auc_base_list[nu] = []

    for n in range(np.max(args.data)):
        path = os.path.join(args.directory, 'data%d.csv' % n)
        print path
        affil_matrix, GT_POS_PAIRS, labels = read_csv(path, unique)
        points = np.array(encoder.predict(affil_matrix)[2])

        for nu in args.nu:
            auc, time_elapsed = eval_auc(points, GT_POS_PAIRS, nu, labels)
            auc_base = run_auc_base(affil_matrix, GT_POS_PAIRS)
            print 'time=%f, nu=%f, AUC=%f, AUC BASE=%f' % (time_elapsed, nu, auc, auc_base)
            auc_list[nu].append(auc)
            auc_base_list[nu].append(auc_base)

    for nu in args.nu:
        print 'nu=%f, AVERAGE AUC=%f' % (nu, np.mean(auc_list[nu]))
        print 'nu=%f, AVERAGE AUC BASE=%f' % (nu, np.mean(auc_base_list[nu]))
else:
    print 'Generating data...'

    auc_list = {}
    auc_base_list = {}
    auc_pca_list = {}
    for nu in args.nu:
        auc_list[nu] = []
        auc_base_list[nu] = []
        auc_pca_list[nu] = []

    for i in range(np.max(args.data)):
        vec_arr, GT_POS_PAIRS, _ = gen_data(data_dict)
        points = np.array(encoder.predict(vec_arr)[2])
        # print GT_POS_PAIRS
        for nu in args.nu:
            auc, time_elapsed = eval_auc(points, GT_POS_PAIRS, nu, const_var=True)
            auc_pca, _ = eval_auc(pca.fit_transform(vec_arr), GT_POS_PAIRS, nu, const_var=True)
            auc_base = run_auc_base(vec_arr, GT_POS_PAIRS)

            print 'time=%f, nu=%f, AUC=%f, AUC BASE=%f, AUC PCA=%f' % \
                (time_elapsed, nu, auc, auc_base, auc_pca)
            auc_list[nu].append(auc)
            auc_base_list[nu].append(auc_base)
            auc_pca_list[nu].append(auc_pca)

    for nu in args.nu:
        print 'nu=%f, AVERAGE AUC=%f, std=%f' % \
            (nu, np.mean(auc_list[nu]), np.std(auc_list[nu]))
        print 'nu=%f, AVERAGE AUC BASE=%f, std=%f' % \
            (nu, np.mean(auc_base_list[nu]), np.std(auc_list[nu]))
        print 'nu=%f, AVERAGE AUC PCA=%f, std=%f' % \
            (nu, np.mean(auc_pca_list[nu]), np.std(auc_pca_list[nu]))
