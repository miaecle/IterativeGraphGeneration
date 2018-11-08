#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:27:09 2018

@author: zqwu
"""
from torch import nn
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WeaveLayer(nn.Module):

  def __init__(self,
               n_atom_input_feat=23,
               n_pair_input_feat=15,
               n_atom_output_feat=64,
               n_pair_output_feat=64,
               n_hidden_AA=64,
               n_hidden_PA=64,
               n_hidden_AP=64,
               n_hidden_PP=64,
               update_pair=True,
               **kwargs):
    """
    Parameters
    ----------
    n_atom_input_feat: int, optional
      Number of features for each atom in input.
    n_pair_input_feat: int, optional
      Number of features for each pair of atoms in input.
    n_atom_output_feat: int, optional
      Number of features for each atom in output.
    n_pair_output_feat: int, optional
      Number of features for each pair of atoms in output.
    n_hidden_XX: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    """
    super(WeaveLayer, self).__init__(**kwargs)
    self.update_pair = update_pair
    self.n_hidden_AA = n_hidden_AA
    self.n_hidden_PA = n_hidden_PA
    self.n_hidden_AP = n_hidden_AP
    self.n_hidden_PP = n_hidden_PP
    self.n_hidden_A = n_hidden_AA + n_hidden_PA
    self.n_hidden_P = n_hidden_AP + n_hidden_PP

    self.n_atom_input_feat = n_atom_input_feat
    self.n_pair_input_feat = n_pair_input_feat
    self.n_atom_output_feat = n_atom_output_feat
    self.n_pair_output_feat = n_pair_output_feat

    self.linear_AA = nn.Sequential(
        nn.Linear(self.n_atom_input_feat, self.n_hidden_AA),
        nn.ReLU(True))
    
    self.linear_PA = nn.Sequential(
        nn.Linear(self.n_pair_input_feat, self.n_hidden_PA),
        nn.ReLU(True))

    self.linear_A_merged = nn.Sequential(
        nn.Linear(self.n_hidden_A, self.n_atom_output_feat),
        nn.ReLU(True))
    
    if self.update_pair:
      self.linear_AP = nn.Sequential(
          nn.Linear(self.n_atom_input_feat * 2, self.n_hidden_AP),
          nn.ReLU(True))
      
      self.linear_PP = nn.Sequential(
          nn.Linear(self.n_pair_input_feat, self.n_hidden_PP),
          nn.ReLU(True))
  
      self.linear_P_merged = nn.Sequential(
          nn.Linear(self.n_hidden_P, self.n_pair_output_feat),
          nn.ReLU(True))

  def forward(self, node_feats, pair_feats):
    """ node_feats: batch_size * n_nodes * node_feat
        pair_feats: batch_size * n_nodes * n_nodes * pair_feat
    """
    AA = self.linear_AA(node_feats)
    PA = self.linear_PA(pair_feats)
    A = self.linear_A_merged(t.cat([AA, PA.sum(1)], 2))

    if self.update_pair:
      AP_ij = self.linear_AP(t.cat([node_feats.unsqueeze(2).expand(-1, -1, 9, -1), 
                                    node_feats.unsqueeze(1).expand(-1, 9, -1, -1)], 3))
      AP_ji = self.linear_AP(t.cat([node_feats.unsqueeze(1).expand(-1, 9, -1, -1), 
                                    node_feats.unsqueeze(2).expand(-1, -1, 9, -1)], 3))      
      
      PP = self.linear_PP(pair_feats)
      P = self.linear_P_merged(t.cat([AP_ij + AP_ji, PP], 3))
    else:
      P = pair_feats

    return A, P

class GraphConvLayer(nn.Module):
  def __init__(self,
               n_atom_input_feat=23,
               n_hidden=64,               
               **kwargs):
    """
    Parameters
    ----------
    n_atom_input_feat: int, optional
      Number of features for each atom in input.
    n_hidden: int, optional
      Number of units(convolution depths)
    """
    super(GraphConvLayer, self).__init__(**kwargs)
    self.n_hidden = n_hidden
    self.n_atom_input_feat = n_atom_input_feat
    self.linear_AA = nn.Linear(self.n_atom_input_feat, self.n_hidden, bias=False)
    self.bias = nn.Parameter(t.zeros(self.n_hidden), requires_grad=True)

  def forward(self, node_feats, A):
    """ node_feats: batch_size * n_nodes * node_feat
        A: batch_size * n_nodes * n_nodes
    """
    x = self.linear_AA(node_feats)
    x = t.matmul(A, x)
    x = x + self.bias
    outputs = F.relu(x)
    return outputs
