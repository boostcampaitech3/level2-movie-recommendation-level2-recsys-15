from __future__ import print_function #파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸 수 있게 해줌 (미래에 ~~)

import torch
import torch.utils.data as data_utils

import numpy as np
import pandas as pd

from scipy.io import loadmat #mat 파일을 읽기 위함
from scipy import sparse #sparse matrix(대부분 0), dense matrix(대부분 1) 둘 중 어느 하나가 많을 때 적은 쪽의 인덱스만 별도로 저장하여
                         #메모리 공간을 효율적으로 사용하기 위함
import os

import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def load_ml1m(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("datasets", "ML_1m", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip()) # [4643, ... , 6702]

    n_items = len(unique_sid) # 6807

    # set args
    args.input_size = [1, 1, n_items] # [1, 1, 6807]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    def load_all_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items)).toarray()
        return data

    # start processing
    def load_train_data(csv_file): # x_train = load_train_data(os.path.join("datasets", "ML_1m", 'train.csv'))
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1 # 25360

        rows, cols = tp['uid'], tp['sid']
        # 희소 행렬(sparse matrix)의 경우 대부분의 값이 0이르모 이를 그대로 사용할 경우 메모리 낭비가 심하고 연산 시간도 오래 걸림
        # 이런 단점을 피하기 위해 희소 행렬을 다른 형태의 자료구조로 변환해서 저장하고 사용함
        # Compressed sparse row(CSR): 가로 순서대로 재정렬하는 방법으로 행에 관여하여 정리 압축.
        data = sparse.csr_matrix((np.ones_like(rows),(rows, cols)), dtype='float32',shape=(n_users, n_items)).toarray()
        # print(data.shape) #(25360, 6807)
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        # x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "ML_1m", 'validation_tr.csv'),
        #                                      os.path.join("datasets", "ML_1m", 'validation_te.csv'))
        # x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "ML_1m", 'test_tr.csv'),
        #                                        os.path.join("datasets", "ML_1m", 'test_te.csv'))

        tp_tr = pd.read_csv(csv_file_tr) # val: (397924, 2)    test: (393157, 2)
        tp_te = pd.read_csv(csv_file_te) # val: (98001, 2)     test: (96791, 2)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min()) # val_start_idx: 25360   test_start_idx: 28360
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())   # val_end_idx: 28359     test_end_idx: 31359

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        # print(data_tr.shape) # (3000, 6807) (3000, 6807)
        # print(data_te.shape) # (3000, 6807) (3000, 6807)
        return data_tr, data_te


    all = load_all_data(os.path.join("datasets", "ML_1m", 'all.csv'))

    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "ML_1m", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "ML_1m", 'validation_tr.csv'),
                                         os.path.join("datasets", "ML_1m", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "ML_1m", 'test_tr.csv'),
                                           os.path.join("datasets", "ML_1m", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1)) # 25360 -> (25360, 1)

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args, all

# ======================================================================================================================
# experiment.py -> train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
def load_dataset(args, **kwargs):
    if args.dataset_name == 'ml1m':
        train_loader, val_loader, test_loader, args, all = load_ml1m(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args, all
