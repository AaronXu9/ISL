"""
This file contains code for running ISL on supervised learning dataset
The goal is to discover the target related DAG and obtain optimal target predictor.
"""

#!/usr/bin/env python3
from pyexpat.errors import XML_ERROR_TEXT_DECL
import utils
import metrics
import isl_module as isl
import torch
import numpy as np
import argparse
import os
from sklearn import preprocessing
import datetime

# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def Swap(arr, start_index, last_index):
    '''
    Switch column start_index and last_index
    arr: numpy array
    return numpy array
    '''
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]] # swap collum

def Swap_row_column(arr, start_index, last_index):
    '''
    Switch column and row of start_index and last_index
    arr: numpy array
    return numpy array
    '''
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]] # swap collum
    arr[[start_index, last_index], :] = arr[[last_index, start_index], :]  # swap row
def swap_vertex_label(vertex_label, index):
    # swap label
    default_y = vertex_label[0]
    swap_y = vertex_label[index]
    vertex_label[0] = swap_y
    vertex_label[index] = default_y
    # swap X value

def split_sachs(folder_path: str):
    """
    Splits sachs dataset
    Args:
            folder_path:
    returns:
            [number of all sample, num of features] ex: (11672, 11)
    """
    import glob
    import pandas
    split_dict = {'1': [1,2,3,4,5], '2': [6,7,8,9,10], '3': [11,12,13,14]}
    sachs_data = list()
    for file in sorted(os.listdir('../data/sachs')):
        if '.xls' in file:
            sachs_df = pandas.read_excel(os.path.join(folder_path, file))
            sachs_data.append(sachs_df.to_numpy())

    return np.vstack(sachs_data)
def read_sachs(folder_path: str):
    """
    Read all sachs source dataset
    Args:
            folder_path:
    returns:
            [number of all sample, num of features] ex: (11672, 11)
    """
    import glob
    import pandas
    sachs_data = list()
    for file in sorted(os.listdir('../data/sachs')):
        if '.xls' in file:
            sachs_df = pandas.read_excel(os.path.join(folder_path, file))
            sachs_data.append(sachs_df.to_numpy())

    return np.vstack(sachs_data)
def squared_loss(output, target):
        n = target.shape[0]
        loss = 0.5 / n * np.sum((output - target) ** 2)
        return loss
def main(args):
    starttime = datetime.datetime.now()
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    # load data
    X_envs, X_test= utils.load_data(dataset_name=args.dataset_name, preprocess=args.preprocess, num_envs=args.num_envs)

    # set target variable Y
    y_index = args.y_index
    vertex_label = utils.load_vertex_label(dataset_name=args.dataset_name)
    # adjust dataset X according to y_index
    X_swap = []
    for X_env in X_envs:
        Swap(X_env, 0, y_index)
        X_swap.append(X_env)
    X = np.stack(X_swap)
    # swap X_test
    Swap(X_test, 0, y_index)
    #  we do not need swap label because when drawing, we will keep the original order.
    # Build model
    n_envs, n, d = X.shape
    Y_pred_hidden = [args.Y_hidden] * (args.Y_hidden_layer-1)
    NOTEAR_hidden = [args.hidden] * args.NOTEAR_hidden_layer
    model = isl.isl_module(
        n_envs=n_envs, Y_dims=[d, args.hidden] + Y_pred_hidden + [1],
        dims=[d] + NOTEAR_hidden + [1], bias=True)
    if args.continue_path:
        pretrained_path = args.continue_path
        model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    # conduct Invariant Structure Learning (ISL)
    y_pred_loss, W_est_origin_envs = isl.notears_nonlinear(model,
                              X, y_index = 0, lambda1=args.lambda1, lambda2=args.lambda2, lambda1_Y=args.lambda1_Y,
                            lambda2_Y_fc1=args.lambda2_Y_fc1, lambda2_Y_fc2=args.lambda2_Y_fc2, w_threshold=0, beta=args.beta) # y_index always be 0, w-thresh always be zero

    #tune W for the best shd score
    g, W_true = utils.load_W_true(args.dataset_name)

    # save results
    os.makedirs(args.Output_path, exist_ok=True)
    for i in range(len(W_est_origin_envs)):# for each environment
        Swap_row_column(W_est_origin_envs[i], 0, y_index) # recover to original order
        if args.Only_Y_DAG: # for boston housing, we only care Y related DAG
            W_est_origin_envs[i][:, 1:] = np.zeros(W_orig[:, 1:].shape)
        # rank W
        wthresh_W_shd_dict = utils.rank_W(W_est_origin_envs[i], 10, 90, W_true, W_threshold_low=0, W_threshold_high=10,
                                          step_size=0.02)
        # select W
        W_est = None
        w_threshold = None
        for threshold, W_element in wthresh_W_shd_dict.items():
            if utils.is_dag(W_element[0]):
                W_est = W_element[0]
                w_threshold = threshold
                break
        exp = args.Output_path + 'pretrain-IRM_BH_norm={}_wthreshold={}_beta={}_y_index={}_lam1={}_lam2fc1={}_lam2fc2_laterY{}NO{}_Yhid={}'.format(
        args.normalization, w_threshold, args.beta, args.y_index, args.lambda1_Y, args.lambda2_Y_fc1, args.lambda2_Y_fc2, args.Y_hidden_layer, args.NOTEAR_hidden_layer, args.Y_hidden)
        # save W_est and W_origin
        utils.save_dag(W_est, f'{exp}_{i}', vertex_label=vertex_label)
        # save the leared W matrix
        np.savetxt(f'{exp}_West_env_{i}.csv', W_est, fmt='%.2f', delimiter=',')
        np.savetxt(f'{exp}_West_origin_env_{i}.csv', W_est_origin_envs[i], fmt='%.2f', delimiter=',')

    # save accuracy
    acc = metrics.count_accuracy(W_true, W_est != 0)
    # acc = {}
    y = model.test(X_test)
    mse = squared_loss(y[:,0], X_test[:,0])
    acc['mse'] = mse
    print(acc)
    with open(f'{exp}_metrics.json', 'w+') as f:
        import json
        json.dump(acc, f)

    # save and load model weights
    MODEL_save_path = exp + '_save_model.pt'
    torch.save(model.state_dict(), MODEL_save_path)
    # model2.load_state_dict(torch.load(MODEL_save_path)) # load model will use the following format

    # print time
    endtime = datetime.datetime.now()
    print('total time is ', (endtime - starttime).seconds)

def parse_args():
    parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
    parser.add_argument('--dataset_name', type=str, default='BH', help='BH (for boston housing), Insurance, Sachs')
    parser.add_argument('--X_path', type=str, help='n by p data matrix in csv format')
    parser.add_argument('--y_index', type = int, default=0, help='use feature y-index to set variable as y for prediction')
    parser.add_argument('--hidden', type=int, default=10, help='Number of hidden units for Notear module')
    parser.add_argument('--Y_hidden', type=int, default=10, help='Number of hidden units of h() function to reconstruct Y')
    parser.add_argument('--Y_hidden_layer', type=int, default=2, help='Number of hidden layers of h() function to reconstruct Y')
    parser.add_argument('--NOTEAR_hidden_layer', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--lambda1', type=float, default=0.01, help='L1 regularization parameter for Notear module')
    parser.add_argument('--lambda2', type=float, default=0.01, help='L2 regularization parameter for Notear module')
    parser.add_argument('--lambda1_Y', type=float, default=0.001, help='L1 regularization parameter on first layer (theta_{1}^{Y}) to reconstruct Y')
    parser.add_argument('--lambda2_Y_fc1', type=float, default=0.001, help='L2 regularization parameter on first layer (theta_{1}^{Y}) to reconstruct Y')
    parser.add_argument('--lambda2_Y_fc2', type=float, default=0.001, help='L2 regularization parameter on rest layers (theta_{r}^{Y}) to reconstruct Y')
    parser.add_argument('--W_path', type=str, default='W_est.csv', help='p by p weighted adjacency matrix of estimated DAG in csv format')
    parser.add_argument('--Output_path', type=str, default='./output/test', help='output path')
    parser.add_argument('--W_threshold', type=float, default=0.5, help='i < threshold no edge')
    parser.add_argument('--Notear_activation', type=str, default='relu', help='relu, sigmoid')
    parser.add_argument('--preprocess', type=str, default='standard', help='removeOutlier, minmax, standard')
    parser.add_argument('--num_envs', type=int, default=2, help='num_envs')
    parser.add_argument('--beta', type=float, default=1, help='y_prediction loss weight, Notear weight')
    parser.add_argument('--continue_path', type=str, default=False, help='None or pretrained model path')
    parser.add_argument('--Only_Y_DAG', type=str, default=False, help='is True, only compare the Y related DAG')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
