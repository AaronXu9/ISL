"""
The file contains the functions used for loading, preprocessing and saving 
the synthetic datasets
"""
import argparse
from matplotlib.pyplot import axis, sca
import numpy as np
from pyparsing import nums
# from paddle import rand, scale
from scipy.special import expit as sigmoid
import igraph as ig
import random
import pandas as pd
import os 
from sklearn import semi_supervised
from sklearn import preprocessing
import utils

def customized_environments(dims:list):
    """
    Simulates 4 different environments

    Args: 
        dim: number of training examples to generate 
    
    Return: 
        a list of 4 DAGs of the four generated environments 
        a list of data for the 4 environmnents 
    """
    vertex_attrs = {'name': ['y', 'x1', 'x2', 's1']}
    g_e1 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e2 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e3 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [2,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )


    g_e4 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )
    
    d = 4

    simu_environs = list()
    for i, G in enumerate([g_e1, g_e2, g_e3, g_e4]):
        # G = ig.Graph.Weighted_Adjacency(W.tolist())
        # G = W
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        X = np.zeros([dims[i], d])
        
        if i == 0:
            '''sigma1 = sigma2 = sigma3 =1'''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 1:
            '''sigma5 = sigma2 =5,  sigma3 =1'''
            X[:,1] = np.random.normal(scale=5.0, size = dims[i])
            X[:,2] = np.random.normal(scale=5.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 2:
            '''sigma5 = sigma2 =1,  sigma3 =1, x2->s1'''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,2] + np.random.normal(scale = 1.0, size = dims[i])
    
        else:
            '''sigma5 = sigma2 =5,  sigma3 =1, '''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
    
        simu_environs.append(X)
        
    return [g_e1, g_e2, g_e3, g_e4] ,simu_environs


def customized_environments_s2(dims:list):
    """
    Simulate 4 different environments of different setting 

    Args: 
        dim: number of training examples to generate 
    
    Return: 
        a list of 4 DAGs of the four generated environments 
        a list of data for the 4 environmnents 
    """
    vertex_attrs = {'name': ['y', 'x1', 'x2', 's1', 's2']}
    g_e1 = ig.Graph(
            n = 5, edges = [[1, 0], [2,0], [0,3], [0,4]],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e2 = ig.Graph(
            n = 5, edges = [[1, 0], [2,0], [0,3], [0,4]],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e3 = ig.Graph(
            n = 5, edges = [[1, 0], [2,0], [0,3], [0,4]],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e4 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3], [0,4]],
            vertex_attrs = vertex_attrs,
            directed = True
    )
    
    d = 5

    simu_environs = list()
    for i, G in enumerate([g_e1, g_e2, g_e3, g_e4]):
        # G = ig.Graph.Weighted_Adjacency(W.tolist())
        # G = W
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        X = np.zeros([dims[i], d])
        
        if i == 0:
            '''sigma1 = sigma2 = sigma3 =1'''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,4] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 1:
            '''sigma5 = sigma2 =5,  sigma3 =1'''
            X[:,1] = np.random.normal(scale=5.0, size = dims[i])
            X[:,2] = np.random.normal(scale=5.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,4] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 2:
            '''sigma5 = sigma2 =1,  sigma3 =1, x2->s1'''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,2] + np.random.normal(scale = 5.0, size = dims[i])
            X[:,4] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
    
        else:
            '''sigma5 = sigma2 =5,  sigma3 =1, '''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,4] = X[:,0] + np.random.normal(scale = 5.0, size = dims[i])
    
        simu_environs.append(X)
        
    return [g_e1, g_e2, g_e3, g_e4] ,simu_environs


def customized_environments_modified_s1(dims:list):
    """Simulate 4 different environments 

    arguments: n: number of training examples 
    """
    vertex_attrs = {'name': ['y', 'x1', 'x2', 's1']}
    g_e1 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e2 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e3 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [2,3], [0,3]],
            vertex_attrs = vertex_attrs,
            directed = True
    )

    g_e4 = ig.Graph(
            n = 4, edges = [[1, 0], [2,0], [0,3],],
            vertex_attrs = vertex_attrs,
            directed = True
    )
    
    d = 4

    simu_environs = list()
    for i, G in enumerate([g_e1, g_e2, g_e3, g_e4]):
        # G = ig.Graph.Weighted_Adjacency(W.tolist())
        # G = W
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        X = np.zeros([dims[i], d])
        
        if i == 0:
            '''sigma1 = sigma2 = sigma3 =1'''
            X[:,1] = np.random.normal(scale=1.0, size = dims[i])
            X[:,2] = np.random.normal(scale=1.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2] + np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 1:
            '''sigma5 = sigma2 =5,  sigma3 =1'''
            X[:,1] = np.random.normal(scale=5.0, size = dims[i])
            X[:,2] = np.random.normal(scale=5.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
        
        elif i == 2:
            '''sigma5 = sigma2 =1,  sigma3 =1, x2->s1'''
            X[:,1] = np.random.normal(scale=5.0, size = dims[i])
            X[:,2] = np.random.normal(scale=5.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 5.0, size = dims[i])
            X[:,3] = X[:,0] + X[:,2] + np.random.normal(scale = 1.0, size = dims[i])
    
        else:
            '''sigma5 = sigma2 =5,  sigma3 =1, '''
            X[:,1] = np.random.normal(scale=5.0, size = dims[i])
            X[:,2] = np.random.normal(scale=5.0, size = dims[i])
            X[:,0] = X[:,1] + X[:,2]+ np.random.normal(scale = 1.0, size = dims[i])
            X[:,3] = X[:,0] + np.random.normal(scale = 1.0, size = dims[i])
    
        simu_environs.append(X)
        
    return [g_e1, g_e2, g_e3, g_e4] ,simu_environs


def bi_classify_env_1(dim):
    """
    Generates binary synthetic data for one enviroment
                    Y = sigmoid(X1 + 2X2)
                    P(S=1 | Y=1) = 0.9, P(S=0 | Y=0) = 0.9 
    Args:
        dim: the number of samples to generate 
    Returns:
        X: np.array, the generated data of shape [dim, 4] for one environment
    """
    d = 4
    X = np.zeros([dim, d])
    scale = 1.0

    X[:,1] = np.random.normal(scale = scale, size = dim)
    X[:,2] = np.random.normal(scale = scale, size = dim)
    X[:,0] = sigmoid(X[:, 1] + 2 * X[:, 2])
    
    Y = X[:,0] > 0.5
    # X[:,0] = X[:,0] > 0.5

    for i in range(dim):
        X[i,0] = int(Y[i])
        if Y[i]:
            X[i,3] = np.random.choice([0,1], p=[0.1, 0.9])
        else:
            X[i,3] = np.random.choice([0,1], p=[0.9, 0.1])
    
    return X


def bi_classify_env_2(dim, indep_sampled = True):
    """
    Generates binary synthetic data for one enviroment
                    Y = sigmoid(X1 + 2X2)
                    P(S=1 | Y=1) = 0.1, P(S=0 | Y=0) = 0.1
    Args:
        dim: the number of samples to generate 
    Returns:
        X: np.array, the generated data of shape [dim, 4] for one environment
    """
    d = 4
    X = np.zeros([dim, d])
    scale = 1.0
    
    X[:,1] = np.random.normal(scale = scale, size = dim)
    X[:,2] = np.random.normal(scale = scale, size = dim)
    X[:,0] = sigmoid(X[:, 1] + 2 * X[:, 2])
    
    Y = X[:,0] > 0.5
    # X[:,0] = X[:,0] > 0.5

    for i in range(dim):
        X[i,0] = int(Y[i])
        if not Y[i]:
            X[i,3] = np.random.choice([0,1], p=[0.1, 0.9])
        else:
            X[i,3] = np.random.choice([0,1], p=[0.9, 0.1])
    return X


def bi_classify_env_3(dim, indep_sampled = True):
    """
    Generates binary synthetic data for one enviroment
                    Y = sigmoid(X1 + 2X2)
                    P(S=1 | Y=1) = 0.5, P(S=0 | Y=0) = 0.5
    Args:
        dim: the number of samples to generate 
    Returns:
        X: np.array, the generated data of shape [dim, 4] for one environment
    """
    d = 4
    X = np.zeros([dim, d])
    scale = 1.0
    
    X[:,1] = np.random.normal(scale = scale, size = dim)
    X[:,2] = np.random.normal(scale = scale, size = dim)
    X[:,0] = sigmoid(X[:, 1] + 2 * X[:, 2])
    
    Y = X[:,0] > 0.5
    # X[:,0] = X[:,0] > 0.5

    for i in range(dim):
        X[i,0] = int(Y[i])
        if not Y[i]:
            X[i,3] = np.random.choice([0,1], p=[0.5, 0.5])
        else:
            X[i,3] = np.random.choice([0,1], p=[0.5, 0.5])
    
    return X


def bi_classify_env_jointSampled(dim, num_env = 3, scale = 1.0, rand = False, probs=[0.1, 0.9, 0.5]):
    """
    Generates synthetic combined from bi_env1, bi_env2, bi_env3
    args:   dim: the number of samples to generate
            num_env: the number of env the samples come from. 
    return: envs: a list of envs containing data belonging to that env
    """
    envs = []
    scale = scale
    X1 = np.random.normal(scale = scale, size = dim * num_env)
    X2 = np.random.normal(scale = scale, size = dim * num_env)

    for i in range(num_env):
        d = 4
        X = np.zeros([dim, d])
        scale = 1.0
        
        X[:,1] = X1[dim * i:dim * (i+1)]
        X[:,2] = X2[dim * i:dim * (i+1)]
        X[:,0] = sigmoid(X[:, 1] + 2 * X[:, 2])
        
        # Y = X[:,0] > 0.5
        X[:,0] = X[:,0] > 0.5
        if i == 0: # for different environment
            if rand:
                X[:, 3] = gen_distribution_S_Y(X[:, 0], random.uniform(0, 1))
            else:
                X[:,3] = gen_distribution_S_Y(X[:,0], probs[i])
        elif i == 1:
            if rand:
                X[:, 3] = gen_distribution_S_Y(X[:, 0], random.uniform(0, 1))
            else:
                X[:,3] = gen_distribution_S_Y(X[:,0], probs[i]) # origin 0.1
        elif i == 2:
            if rand:
                X[:, 3] = gen_distribution_S_Y(X[:, 0], random.uniform(0, 1))
            else:
                X[:,3] = gen_distribution_S_Y(X[:,0], probs[i])
        
        envs.append(X)
    
    return envs


def gen_distribution_S_Y(Y, prob):
    """
    generate probability of P(S|Y) 
    :parm: P(S = 1 | Y = 1) = prob
    :return: S 
    """
    S = np.zeros(Y.shape)
    for i in range(len(Y)):
        if Y[i]: # Y=1
            S[i] = np.random.choice([0,1], p=[1-prob, prob])
        else: 
            S[i] = np.random.choice([0,1], p=[prob, 1-prob])
    
    return S


def binary_ISL(args, dim, num_xy, num_s, num_env = 3, scale = 1.0, sem_type='uniform', rand = False, probs=[0.1, 0.9, 0.5]):
    """
    Generates binary data with with X, Y, and S nodes where X->Y and S are the spurious correlations of Y satisfying the
                condition P(S=1 | Y =1) = probs[i] for environment i 
    args:   args: containing argumetns 
            dim: the number of samples to generate
            num_env: the number of env the samples come from. 
            scale: the variation of the data 
            sem_type: the noise type
            rand:
            probs: the probability controling S's distributiont with regard to Y
    return: np.array of shape [num_env, dim, num_xy + num_s]
    """
    
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=dim * num_env)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=dim * num_env)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=dim * num_env)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=dim * num_env)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == 'none':
            x = X @ w
        else:
            raise ValueError('unknown sem type')
        
        return x
    
    X_envs = np.zeros([num_env, dim, num_xy+num_s])
    scale = scale
    X = np.zeros([dim * num_env, num_xy])
    S = np.zeros([dim * num_env, num_s])

    '''initialize X'''
    for i in range(1, num_xy, 1):
        X[:, i] = np.random.normal(scale = scale, size = dim * num_env)
    
    '''Y = sigmoid(sum(X))'''
    # X[:, 0] = sigmoid(np.sum(X[:, 1:], axis=1))
    W1 = np.random.uniform(low=0.5, high=2.0, size=num_xy-1)
    print(W1)
    os.makedirs(args.Output_path, exist_ok=True)
    if not os.path.exists(f'{args.Output_path}Wparam.csv'):
        np.savetxt(f'{args.Output_path}Wparam.csv', W1, fmt='%.3f', delimiter=',')
    X[:, 0] = _simulate_single_equation(X[:, 1:], W1, scale=scale)

    X[:,0] = X[:,0] > 0.5
    '''P(Y = 1 | X) = sigmoid(X)'''
    # for i in range(X.shape[0]):
    #     X[i][0] = np.random.choice([1,0], p=[X[i][0], 1 - X[i][0]])
    
    for i in range(num_env):
        '''modify the corresponding X_envs'''
        X_envs[i][:, :num_xy] = X[i * dim : (i + 1) * dim]
        
        for j in range(num_s):
            if not rand:
                X_envs[i][:, num_xy+j] = gen_distribution_S_Y(X_envs[i][:, 0], probs[i])
            else:
                X_envs[i][:, num_xy+j] = gen_distribution_S_Y(X_envs[i][:, 0], np.random.uniform(0.0, 1.0))

    return X_envs


def bi_classify_env(dim):
    """
    Wrapper function for bi_classify_env_jointSampled
    Args: 
        dim: number of sampel to generate
    """
    return bi_classify_env_jointSampled(dim)


def gen_castle_complex_env(dim, num_env = 3):
    """
    Generates samples following the castle graph but generated in different envs
    Args:   
            dim: the number of samples to generate
            num_env: the number of env the samples come from. 
    Returns: 
            a list of envs containing samples sampled in the env.
    """
    envs = list()
    # env_0 Y->X8, Y->X9, Y->X6
    g1 = ig.Graph(
        n = 10, edges =  [[1,2], [1,3], [1,4], [2,5], [2,0], [3,0],[3,6],[3,7], [0,8], [0,9], [0,6], [6,9]],
        vertex_attrs = {'vertex_label': ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9']},
        directed=True
    )

    X1 = utils.simulate_nonlinear_sem(np.array(g1.get_adjacency().data), n=200, sem_type='mlp')

    # env_1 Y->X5,
    g2 = ig.Graph(
        n = 10, edges =  [[1,2], [1,3], [1,4], [2,5], [2,0], [3,0], [3,6],[3,7], [0,5], [6,9]],
        vertex_attrs = {'vertex_label': ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9']},
        directed=True
    )
    X2 = utils.simulate_nonlinear_sem(np.array(g2.get_adjacency().data), n=200, sem_type='mlp')

    # env_2 Y->X7, 
    g3 = ig.Graph(
        n = 10, edges =  [[1,2], [1,3], [1,4], [2,5], [2,0], [3,0],[3,6],[3,7], [0,7], [6,9]],
        vertex_attrs = {'vertex_label': ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9']},
        directed=True
    )

    X3 = utils.simulate_nonlinear_sem(np.array(g3.get_adjacency().data), n=200, sem_type='mlp')
    
    return [X1, X2, X3]


def gen_castle_env(dim, probs, num_env=3):
    
    X = np.zeros([num_env, dim, 10])
    for i in range(num_env):
        X[i] = gen_castle_single_env(dim, [probs[0][i], probs[1][i]])
    
    return X


def gen_castle_single_env(dim, probs): 
    """
    Generates samples following the castle graph but generated different envs
    Args:   
            dim: the number of samples to generate
            probs: list containing probs for spurious correlation vars 
            num_env: the number of env the samples come from. 
    Returns: 
            samples from a single env
    """

    g = ig.Graph(
        n = 8, edges =  [[1,2], [1,3], [1,4], [2,5], [2,0], [3,0],[3,6], [3,7]],
        vertex_attrs = {'vertex_label': ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']},
        directed=True
    )
    
    X = utils.simulate_nonlinear_sem(np.array(g.get_adjacency().data), n=dim, sem_type='mlp')
    S = np.zeros([dim, 2])
    for i in range(2):
        S[:,i] = gen_S(X[:,0], dim, probs[i])

    return np.hstack([X, S])
    

def gen_xyz_complex_env(dim, num_env = 3):
    """
    Generates samples following the our xyz graph but generated in different envs
    Args:   
            dim: the number of samples to generate
            num_env: the number of env the samples come from. 
    Returns: 
            a list of envs and samples sampled in the env.
    """
    envs = list()
    # env_0 Z->X6 X-Y3
    g1 = ig.Graph(
            n=10, edges = [[1,7], [2,7], [3,7], [3,8], [4,8], [5,8], [7,0], [8,0], [0,9], [0,6]],
            vertex_attrs={'name': ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1','y2', 'y3']},
            directed=True
        )

    X1 = utils.simulate_nonlinear_sem(np.array(g1.get_adjacency().data), n=200, sem_type='mlp')

    # env_1 Z->X6
    g2 = ig.Graph(
            n=10, edges = [[1,7], [2,7], [3,7], [3,8], [4,8], [5,8], [7,0], [8,0], [0,6]],
            vertex_attrs={'name': ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1','y2', 'y3']},
            directed=True
        )
    X2 = utils.simulate_nonlinear_sem(np.array(g2.get_adjacency().data), n=200, sem_type='mlp')

    # env_2 Z->Y3
    g3 = ig.Graph(
            n=10, edges = [[1,7], [2,7], [3,7], [3,8], [4,8], [7,0], [8,0], [0,9]],
            vertex_attrs={'name': ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1','y2', 'y3']},
            directed=True
        )

    X3 = utils.simulate_nonlinear_sem(np.array(g3.get_adjacency().data), n=200, sem_type='mlp')
    
    return [X1, X2, X3]


def gen_synthetic_env(data_source: str, dim):
    """
    generate synthetic data (created in different envs) depending on data_source
    Args:
            data_source: 
            dim: sample size for each env
    return: a list of envs each containing data sampled in that env
    """
    if data_source == 'binary':
        vertex_label = ['Y', 'X1', 'X2', 'S1']
        return np.array(bi_classify_env_jointSampled(dim)), vertex_label, customized_graph(data_source)
    
    elif data_source == 'xyz_complex':
        vertex_label = ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1','y2', 'y3']

        return np.array(gen_xyz_complex_env(dim)), vertex_label, customized_graph(data_source)
    
    elif data_source == 'castle_complex':
        vertex_label = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9']
        return np.array(gen_castle_complex_env(dim)), vertex_label, customized_graph(data_source)
    
    elif data_source == 'random':
        return 
    
    else:
        ValueError('invalid data source')    


def gen_ISL_simple(num_env, n, ISL_param,  noise_scale=None):
    """Simulate samples from nonlinear SEM.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    d = ISL_param['num_xy']
    s = ISL_param['num_s']
    X_envs = np.zeros([num_env, n, d+s])
    for i in range(num_env):
        scale = 1
        X = np.zeros([n, d])
        '''the X's are random Gaussian noise'''
        for i in range(1, d, 1):
            X[:, i] = np.random.normal(scale=scale, size=n)

        np.random.seed(42)
        hidden = 100
        W1 = np.random.uniform(low=0.5, high=2.0, size=[d-1, hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
        W2[np.random.rand(hidden) < 0.5] *= -1
        '''Y = f(X) + z'''
        z = np.random.normal(scale=scale, size=n)
        X[:,0] = sigmoid(X[:, 1:] @ W1) @ W2 + z
        S = np.zeros([n, ISL_param['num_s']])
        for i in range(2):
            if ISL_param['train']:
                S[:,i] = gen_S(X[:,0], n, ISL_param['probs'][i])
            else:
                S[:,i] = gen_S(X[:,0], n, ISL_param['test_probs'][i])
        data = np.hstack([X, S])
        # append to the envs
        X_envs[i] = data

    return X_envs


def gen_counterfactural_envs(data, ISL_param, counter_var='X1', counter_scale=2, counter_shift = 5):
    """
    Generates the data after applying a perturbation on a variable X or S
    """
    X = np.zeros(data.shape)
    for i in range(data.shape[0]):
        X[i] = gen_counterfactural(data[i], ISL_param, counter_var='X1', counter_scale=2, counter_shift = 5)
    
    return X


def gen_counterfactural(data, ISL_param, counter_var='X1', counter_scale=2, counter_shift = 5):
    '''
    generates the counterfactures

    '''
    n = data.shape[0]
    d = ISL_param['num_xy']
    hidden =100
    np.random.seed(42)
    W1 = np.random.uniform(low=0.5, high=2.0, size=[d-1, hidden])
    W1[np.random.rand(*W1.shape) < 0.5] *= -1
    W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
    W2[np.random.rand(hidden) < 0.5] *= -1
    z = np.random.normal(scale=1, size=n)

    if 'X' in counter_var:
        data[:, int(counter_var[1])] = np.random.normal(scale=counter_scale, size=n) + counter_shift
        # X[:, ]
        data[:,0] = sigmoid(data[:, 1:ISL_param['num_xy']] @ W1) @ W2 + z
    elif 'S' in counter_var:
        data[:, ISL_param['num_xy'] - 1 + int(counter_var[1])] = gen_S(data[:,0], ISL_param['num_s'], prob=ISL_param['probs'])
        data[:,0] = sigmoid(data[:, 1:ISL_param['num_xy']] @ W1) @ W2 + z
    
    return data


def gen_ISL(num_xy, num_s, num_sample, prob = 0.3, scale=None, shift=None, sem_type='mlp'):
    """
    n: number of x-y nodes 
    dim: number of samples 
    """

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)
    
    g = ig.Graph(n=num_xy, directed=True)

    # generate y-realted edges 
    edges = [(i, 0) for i in range(1,num_xy,1)]
    g.add_edges(edges)
    B = _graph_to_adjmat(g)
    X = utils.simulate_nonlinear_sem(B, num_sample, sem_type='mlp', noise_scale=scale)
    
    # create S spurious correlation variable 
    Y = sigmoid(X[:,0]) > 0.5
    S = np.zeros([num_sample, num_s])

    X = np.hstack([X, S])
    for i in range(num_sample):
        for j in range(num_s):
            W1 = np.random.uniform(low=0.5, high=2.0, size=1)
            gs = sigmoid(X[i][0] * W1) + np.random.normal(scale=1.0, size = 1)
            random_s = np.random.normal(scale=1.0, size = 1)

            X[i][num_xy+j] = np.random.choice([np.squeeze(gs), np.squeeze(random_s)], p=[prob, 1-prob])
    
    return X


def gen_S(Y, dim, prob):
    """
    Generates spurious correlation variables using the conditional probability given prob Y
    Args:   
            Y: 
            dim: the number of samples to generate
            prob: the conditional probability given Y
    Returns: 
            [dim] array of the S var
    """
    S = np.zeros(dim)
    Y_boolean = sigmoid(Y) > 0.5

    for i in range(dim):
        W1 = np.random.uniform(low=0.5, high=2.0, size=1)
        gs = sigmoid(Y[i] * W1) + np.random.normal(scale=1.0, size = 1)
        random_s = np.random.normal(scale=1.0, size = 1)

        S[i] = np.random.choice([np.squeeze(gs), np.squeeze(random_s)], p=[prob, 1-prob])
    
    return S


def gen_ISL_env(num_xy, num_s, num_sample, num_env, probs = [0.1, 0.2, 0.3], scale=None, shift=None, sem_type='mlp'):
    """
    generate ISL samples for different environments 
    """
    # probs=[0.1, 0.2, 0.8]
    X_envs = np.zeros([num_env, num_sample, num_xy+num_s])

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)
    
    g = ig.Graph(n=num_xy, directed=True)

    # generate y-realted edges 
    edges = [(i, 0) for i in range(1,num_xy,1)]
    g.add_edges(edges)
    B = _graph_to_adjmat(g)
    X = utils.simulate_nonlinear_sem(B, num_sample * num_env, sem_type=sem_type, noise_scale=scale)
    
    for i in range(num_env):
        X_envs[i][:, :num_xy] = X[i * num_sample : (i + 1) * num_sample]

    for env_id in range(num_env):
        prob = probs[env_id]

        for i in range(num_sample):

            for j in range(num_s):
                W1 = np.random.uniform(low=0.5, high=2.0, size=1)
                gs = sigmoid(X[i][0] * W1) + np.random.normal(scale=1.0, size = 1)
                gauss = np.random.normal(loc=gs, scale=1)
                random_s = np.random.normal(scale=1.0, size = 1)

                X_envs[env_id][i][num_xy+j] = np.random.choice([np.squeeze(gauss), -np.squeeze(gauss)], p=[prob, 1-prob])

        
    return X_envs


def parse_args():
    parser = argparse.ArgumentParser(description='synthetic dataset generation')
    parser.add_argument('--num_sample', type=int, default=1000, help='number of examples to generate')

    parser.add_argument('--num_xy', type=int, default=3, help='')
    parser.add_argument('--num_s', type=int, default=1, help='number of xy variables')
    parser.add_argument('--num_cluster', type=int, default=3, help='number of spurious correlation variables')
    parser.add_argument('--num_env', type=int, default=3, help='number of environments')
    parser.add_argument('--scale', type=float, default=1, help='number of environments')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gen_ISL(args.num_xy, args.nums, args.num_sample, args.num_env, probs=np.random.uniform(low=0, high=1, size=len(args.num_env)))
