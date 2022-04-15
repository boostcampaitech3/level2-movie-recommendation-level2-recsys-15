from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.optimizer import AdamNormGrad

import os
import numpy as np
import datetime
import pandas as pd
from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#----------------------------------------------------preprocessing-----------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

data = pd.read_csv("/opt/ml/input/data/train/train_ratings_4col.csv")

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상)
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

# 훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
# 100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
# 확인하기 위함입니다.
def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

print("Load and Preprocess Movielens dataset")
# Load Data
DATA_DIR = "/opt/ml/input/EVCF/datasets/ML_1m"
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings_4col.csv'), header=0)
print("원본 데이터\n", raw_data)

# Filter Data
raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)
#제공된 훈련데이터의 유저는 모두 5개 이상의 리뷰가 있습니다.
print("5번 이상의 리뷰가 있는 유저들로만 구성된 데이터\n",raw_data)

print("유저별 리뷰수\n",user_activity)
print("아이템별 리뷰수\n",item_popularity)

# Shuffle User Indices
unique_uid = user_activity.index
print("(BEFORE) unique_uid:",unique_uid)

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]
print("(AFTER) unique_uid:",unique_uid)

n_users = unique_uid.size #31360
n_heldout_users = 3000

# Split Train/Validation/Test User Indices
tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

#주의: 데이터의 수가 아닌 사용자의 수입니다!
print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
print("검증 데이터에 사용될 사용자 수:", len(vd_users))
print("테스트 데이터에 사용될 사용자 수:", len(te_users))

##훈련 데이터에 해당하는 아이템들
#Train에는 전체 데이터를 사용합니다.
train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

##아이템 ID
unique_sid = pd.unique(train_plays['movieId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
# print(show2id)
# print(profile2id)

pro_dir = os.path.join(DATA_DIR)

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

#Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

print("Done!")

#데이터 셋 확인
print(train_data)
print(vad_data_tr)
print(vad_data_te)
print(test_data_tr)
print(test_data_te)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#----------------------------------------------------Start experiment--------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')

# arguments for optimization
parser.add_argument('--batch_size', type=int, default=200, metavar='BStrain',
                    help='input batch size for training (default: 200)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='BStest',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='E',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warm-up')
parser.add_argument('--max_beta', type=float, default=1., metavar='B',
                    help='maximum value of beta for training')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: latent size, input_size, so on
parser.add_argument('--num_layers', type=int, default=1, metavar='NL',
                    help='number of layers')

parser.add_argument('--z1_size', type=int, default=200, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=200, metavar='M2',
                    help='latent size')
parser.add_argument('--hidden_size', type=int, default=600 , metavar="H",
                    help='the width of hidden layers')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=1000, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='baseline', metavar='MN',
                    help='model name: baseline, vamp, hvamp, hvamp1')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous, multinomial')

parser.add_argument('--gated', action='store_true', default=False,
                    help='use gating mechanism')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='ml1m', metavar='DN',
                    help='name of the dataset:  ml20m, netflix, pinterest')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# note
parser.add_argument('--note', type=str, default="none", metavar='NT',
                    help='additional note on the experiment')
parser.add_argument('--no_log', action='store_true', default=False,
                    help='print log to log_dir')

# save
parser.add_argument('--save', type=str, default='model1.pt',
                    help='path to save the final model')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  #! Changed num_workers: 1->0 because of error

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#----------------------------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:10]

    model_name = args.dataset_name + '_' + args.model_name + '_' + \
                 '(K_' + str(args.number_components) + ')' + \
                 '_' + args.input_type + '_beta(' + str(args.max_beta) + ')' + \
                 '_layers(' + str(args.num_layers) + ')' + '_hidden(' + str(args.hidden_size) + ')' + \
                 '_z1(' + str(args.z1_size) + ')' + '_z2(' + str(args.z2_size) + ')'
    # print(model_name) # ml1m_baseline_(K_1000)_binary_beta(1.0)_layers(1)_hidden(600)_z1(200)_z2(200)

    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + args.model_signature + '_' + model_name + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    # utils -> load_data.py(def load_ml1m -> def load_dataset  => return train_loader, val_loader, test_loader, args)
    train_loader, val_loader, test_loader, args, all = load_dataset(args, **kwargs)
    # print("~~~~~~~~", all)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'baseline':
        from models.Baseline import VAE
    elif args.model_name == 'vamp':
        from models.Vamp import VAE
    elif args.model_name == 'hvamp':
        from models.HVamp import VAE
    elif args.model_name == 'hvamp1':
        from models.HVamp_1layer import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    # print(VAE)
    # Namespace(MB=100, S=5000, activation=None, batch_size=200, cuda=True, dataset_name='ml1m',
    #           dynamic_binarization=False, early_stopping_epochs=50, epochs=1, gated=False, hidden_size=600,
    #           input_size=[1, 1, 6807], input_type='binary', lr=0.0005, max_beta=1.0, model_name='baseline',
    #           model_signature='2022-04-12', no_cuda=False, no_log=False, note='none', num_layers=1,
    #           number_components=1000, pseudoinputs_mean=0.05, pseudoinputs_std=0.01, seed=14, test_batch_size=1000,
    #           use_training_data_init=False, warmup=100, z1_size=200, z2_size=200)
    # print(model)
    # (q_z_layers): Sequential(
    #     (0): Dropout(p=0.5, inplace=False)
    # (1): NonLinear(
    #     (activation): Tanh()
    # ... 모델 구조

    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)
    # print(optimizer)
    # AdamNormGrad(Parameter Group 0
    #                 betas: (0.9, 0.999)
    #                 eps: 1e-08
    #                 lr: 0.0005
    #                 weight_decay: 0     )

    # ======================================================================================================================
    # /opt/ml/input/EVCF/vae_experiment_log_None.txt 파일에 진행 결과 저장되어 있음
    print(args)
    log_dir = "vae_experiment_log_" + str(os.getenv("COMPUTERNAME")) +".txt"

    open(log_dir, 'a').close()

    # ======================================================================================================================
    print('perform experiment')
    from utils.perform_experiment import experiment_vae
    best_model = experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, log_dir, model_name = args.model_name)

    return best_model, all
    # ======================================================================================================================

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    best_model, all = run(args, kwargs)
    # print(best_model(torch.FloatTensor(all)))
    with open("/opt/ml/input/EVCF/snapshots/2022-04-12_ml1m_baseline_(K_1000)_binary_beta(1.0)_layers(1)_hidden(600)_z1(200)_z2(200)/model1.pt", 'rb') as f:
        model = torch.load(f)
        print('ok')
    print(all)
    # all_predictions = model(torch.FloatTensor(all.toarray()))
    # print(all_predictions)
# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

# best_model = torch.load(dir + args.save) #baseline.model + '.model.pt'
# all_predictions, mu_pred, logvar_pred = best_model((torch.FloatTensor))
# print(all_predictions, mu_pred, logvar_pred)
# print(best_model)