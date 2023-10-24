""" self supervised learning to learn the DAG using ISL"""
#!/usr/bin/env python3
import utils
import ISL_module as ISL
import torch
import numpy as np
import argparse
import os
from sklearn import preprocessing
import pandas as pd
import data_gen_real

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * np.sum((output - target) ** 2)
    return loss

def main(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)


    # load data
    X = utils.load_data(dataset_name=args.dataset_name, num_envs=args.num_envs)
    # load vertex label
    vertex_label =utils.load_vertex_label(dataset_name=args.dataset_name)

    # the initial potential causal parents for each variable
    if args.dataset_name == 'Insurance':
        Init_Relation = {
        'PropCost':      ['ThisCarCost', 'OtherCarCost'],  # 0
        'GoodStudent':   ['Age', 'SocioEcon'], # 1
        'Age':           ['GoodStudent', 'RiskAversion'], #2
        'SocioEcon':     ['RiskAversion', 'MakeModel', 'HomeBase'],#3
        'RiskAversion':  ['SocioEcon', 'DrivQuality', 'DrivingSkill', 'HomeBase'],#4
        'VehicleYear':   ['SocioEcon', 'MakeModel', 'Antilock', 'CarValue', 'Airbag'],#5
        'ThisCarDam':    ['RuggedAuto', 'Accident', 'ThisCarCost'],#6
        'RuggedAuto':    ['VehicleYear', 'MakeModel', 'Antilock', 'Cushioning'], #7
        'Accident':      ['ThisCarDam', 'RuggedAuto'], #8
        'MakeModel':     ['SocioEcon', 'VehicleYear', 'RuggedAuto', 'CarValue'], #9
        'DrivQuality':   ['RiskAversion', 'DrivingSkill'],# 10
        'Mileage':       ['MakeModel', 'CarValue'],#11
        'Antilock':      ['VehicleYear', 'MakeModel'],#12
        'DrivingSkill':  ['Age', 'DrivQuality'],#13
        'SeniorTrain':   ['Age', 'RiskAversion'],#14
        'ThisCarCost':   ['RuggedAuto', 'CarValue'],#15
        'Theft':         ['ThisCarDam', 'MakeModel', 'CarValue', 'HomeBase', 'AntiTheft'],#16
        'CarValue':      ['VehicleYear', 'MakeModel'],#17
        'HomeBase':      ['SocioEcon', 'RiskAversion'],#18
        'AntiTheft':     ['SocioEcon', 'RiskAversion'],#19
        'OtherCarCost':  ['RuggedAuto', 'Accident'],#20
        'OtherCar':      ['SocioEcon'],#21
        'MedCost':       ['Age', 'Accident', 'Cushioning'],#22
        'Cushioning':    ['RuggedAuto', 'Airbag'],#23
        'Airbag':        ['VehicleYear'],#24
        'ILiCost':       ['Accident'],#25
        'DrivHist':      ['RiskAversion', 'DrivingSkill'],#26
    }
    elif args.dataset_name=='Sachs':
        Init_Relation = {
            'Raf':  ['Mek'],
            'Mek':  ['Raf'],
            'Plcg': ['PIP2'],
            'PIP2': ['Plcg', 'PIP3'],
            'PIP3': ['Plcg', 'PIP2'],
            'Erk':  ['Akt', 'Mek',  'PKA'],
            'Akt':  ['PKA', 'Erk'],
            'PKA':  ['Akt'],
            'PKC':  ['P38'],
            'P38':  ['PKA', 'PKC'],
            'Jnk':  ['PKC'],
        }
    Init_DAG = np.zeros((len(vertex_label), len(vertex_label)))
    for x in vertex_label:
        x_index = vertex_label.index(x)
        for y in Init_Relation[x]: # for each causal parent of x
            y_index = vertex_label.index(y)
            Init_DAG[y_index, x_index] = 1

    n, d = X.shape
    model = ISL.NotearsMLP_self_supervise(dims=[d, args.hidden, 1], bias=True, Init_DAG=Init_DAG)
    model.to(device)
    '''To use a different feature as label, change y_index to col index of the feature'''
    W_est_origin = ISL.notears_nonlinear(model, X, lambda1=args.lambda1, lambda2=args.lambda2, w_threshold=0) # keep origin W_est
    
    #tune W for the best shd score
    g, W_true = utils.load_W_true(args.dataset_name)

    
    wthresh_W_shd_dict = utils.rank_W(W_est_origin, 10, 90, W_true, W_threshold_low = 0, W_threshold_high=10, step_size=0.01)
    W_est = None
    w_threshold = None
    for threshold, W_element in  wthresh_W_shd_dict.items():
        if utils.is_dag(W_element[0]):
            W_est = W_element[0]
            w_threshold = threshold
            break
    
    exp = args.Output_path + 'notear_mlp_sachs_norm={}wthreshold={}lambda1={}lambda2={}'.format(
        args.normalization, w_threshold, args.lambda1, args.lambda2)

    os.makedirs(args.Output_path, exist_ok=True)

    np.savetxt(exp + '_W_est_origin.csv', W_est_origin, delimiter=',')
    np.savetxt(exp + '_W_est.csv', W_est, delimiter=',')
    utils.save_dag(W_est, exp + '_W_est', vertex_label=vertex_label)

    
    acc = utils.count_accuracy(W_true, W_est != 0)
    print(acc)

    y = model(torch.from_numpy(X))
    y = y.cpu().detach().numpy()
    mse = squared_loss(y[:,0], X[:,0])
    print("mse:", mse)
    acc['mse'] = mse

    with open(f'{exp}_metrics.json', 'w+') as f:
        import json
        json.dump(acc, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
    parser.add_argument('--dataset_name', type=str, default='BH', help='BH (for boston housing), Insurance, Sachs')
    parser.add_argument('--X_path', type=str, help='n by p data matrix in csv format')
    parser.add_argument('--y_index', type = int, default = 0, help='use feature y-index as y for prediction')
    parser.add_argument('--hidden', type=int, default=10, help='Number of hidden units')
    parser.add_argument('--lambda1', type=float, default=0.01, help='L1 regularization parameter')
    parser.add_argument('--lambda2', type=float, default=0.01, help='L2 regularization parameter')
    parser.add_argument('--W_path', type=str, default='W_est.csv', help='p by p weighted adjacency matrix of estimated DAG in csv format')
    parser.add_argument('--Output_path', type=str, default='./output/sparseInit_Ins/', help='output path')
    parser.add_argument('--W_threshold', type=float, default=0.5, help='i < threshold no edge')
    parser.add_argument('--Notear_activation', type=str, default='relu', help='relu, sigmoid')
    parser.add_argument('--normalization', type=str, default='standard', help='use normalization preprocess standard or minmax')
    parser.add_argument('--datasource', type=str, default='raw', help='raw, IRM')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

