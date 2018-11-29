#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:19:20 2018

@author: mguo123
"""

# max-pool matching from cho 2014, simonovsky 2018 (GRAPHVAE), and Graph RNN 
import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

MAX_NUM_NODES=6 #TODO: find a place to put this in

# def recover_adj_lower(l, max_num_nodes=MAX_NUM_NODES):
#     # NOTE: Assumes 1 per minibatch
#     # return upper triangular matrix with all values as l
#     # input: int (l)l nonzero values

#     A = torch.zeros(max_num_nodes, max_num_nodes)
#     A[torch.triu(torch.ones(max_num_nodes, max_num_nodes)) == 1] = l
#     return A

# def recover_full_adj_from_lower( lower):
#     diag = torch.diag(torch.diag(lower, 0))
#     return lower + torch.transpose(lower, 0, 1) - diag

def adj_recon_loss(self, A_truth, A_pred):
    return F.binary_cross_entropy(A_truth, A_pred)

def feature_similarity(f1, f2):
    """
    find similarity between two feature fectors 
    f1 and f2: are either are floats or vectors of the same length
    return: float or vector that represents a similarity between two feature vectors
    """
    return 1 / (abs(f1.float() - f2.float()) + 1.)

def edge_similarity_matrix( A_label, A_pred, 
                            feat_label, feat_pred, 
                            sim_func=feature_similarity, max_num_nodes=MAX_NUM_NODES):
    """
    Calculate Similarity Matrix between node and node pairs of two graphs G_1(i,j), G_2(a,b)

    S(i,j, a, b) is given in Simonovsky, et al (2018) eq (4) 
        S((i, j),(a, b)) =
                (E_{i,j},E_{a,b})A_{i,j}A_{a,b}A_{a,a}A_{b,b}[i 6= j ^ a 6= b] + 
                (F_{i},F_{a})A_{a,a}[i = j ^ a = b]
    We note that the default feature similarity function is the number of degrees of the node

    Inputs:
        A_label: adjacency matrices for each molecule in batch; batch_size x max_num_nodes x max_num_nodes
        A_pred: same above for the prediction
        feat_label: features for nodes to be taken into account: batch_size x max_num_nodes 
        feat_pred: same as above for prediction
        sim_func: applied to the features of the label and prediction
        max_num_nodes: number of nodes of the graph 

    """
    batch_size = A_label.shape[0]
    max_num_nodes = A_label.shape[1]### TODO: remove

    S = torch.zeros(batch_size, max_num_nodes, max_num_nodes,
                    max_num_nodes, max_num_nodes)
    for i in range(max_num_nodes):
        for j in range(max_num_nodes):
        	# for the node-node pairs
            if i == j:
                for a in range(max_num_nodes):
                    S[:, i, i, a, a] = A_pred[:, a, a] * sim_func(feat_label[:, i], feat_pred[:, a])
                    # with feature not implemented
                    # if input_features is not None:
            # for the edges/bond pairs
            else:
                for a in range(max_num_nodes):
                    for b in range(max_num_nodes):
                        if b == a:
                            continue
                        S[:, i, j, a, b] = A_label[:, i, j] * A_pred[:, a, b] * A_pred[:, a, a] * A_pred[:, b, b]
                        # A_label[:, i, j] * A_label[:, i, i] * A_label[:, j, j] * \
                        #                 A_pred[:, a, b] * A_pred[:, a, a] * A_pred[:, b, b]
    return S



def mpm(x_init, S, max_iters=50, max_num_nodes=MAX_NUM_NODES):
    """
    Implement quadratic programming task:
        x^* = argmax_x (x^T S x)
        for max_iterations loop under convergence or max interations (defualt max iterations)
        x_{ia} <- x_{ia}S_{ia;ia} + sum_{j in N_i} max_{b in N_a} x_{jb} S_{ia;jb}

    Input: 
        x_init: batch_size x num_nodes x num_nodes (32,6,6)
        S: batch_size x num_nodes x num_nodes x num_nodes x num_nodes (32,6,6,6,6)
    Return: 
        optimized x: same size as x_init
    """
    # TODO: vectorize
    x = x_init
    batch_size = x.shape[0]
    # iterate through the number of iterations (instead of until convergence) 
    for it in range(max_iters):
        x_new = torch.zeros_like(x)
        for i in range(max_num_nodes):
            ind = range(i) + range(i+1,max_num_nodes) 
            for a in range(max_num_nodes):
                x_new[:,i, a] = x[:, i, a] * S[:, i, i, a, a]
                #the result of max tuple of two output tensors (max, max_indices)
                pooled = torch.max(x[:, ind, :] * S[:, i, ind, a, :], 2)[0]
                sum_pooled = torch.sum(pooled, 1)
                x_new[:, i, a] += sum_pooled
                # pooled = [torch.max(x[:, j, :] * S[:, i, j, a, :])
                #           for j in range(max_num_nodes) if j != i]
                # neigh_sim = sum(pooled)
                # x_new[i, a] += neigh_sim
        norm = torch.norm(x_new.view(batch_size, -1),dim=1).view(batch_size,1,1) + 1e-6 #pseudocount
        # print('norm', norm.shape, norm)
        # print('x_new',x_new.shape, x_new)
        x = x_new / norm
    return x 



def permute( in_matrix, curr_ind, target_ind, max_num_nodes=MAX_NUM_NODES):
    ''' Permute adjacency matrix and nodes list
        order curr_ind according to target ind
    '''
    # 
    ind = np.zeros(max_num_nodes, dtype=np.int)
    ind[curr_ind] = target_ind
    permuted = torch.zeros_like(in_matrix)
    if len(permuted.shape) ==2:
        permuted[:, :] = in_matrix[ind, :]
        permuted[:, :] = permuted[:, ind]
    elif len(permuted.shape)==1:
        permuted[:] = in_matrix[ind]
    else:   
        print('permuted shape',permuted.shape)
        raise "shape wrong"
    return permuted
