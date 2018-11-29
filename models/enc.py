#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:27:44 2018

@author: zqwu
"""

from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
from layers import WeaveLayer, GraphConvLayer

class WeaveEnc(nn.Module):
  def __init__(self,
               n_node_feat=23,
               n_pair_feat=15,
               n_node_hidden=64,
               n_pair_hidden=64,
               n_latent_feat=128,
               n_weave_layers=3):
    """
    Parameters
    ----------
    n_node_feat: int, optional
      Number of features for each atom in input.
    n_pair_feat: int, optional
      Number of features for each pair of atoms in input.
    n_node_hidden: int, optional
      Size of hidden layer for each atom.
    n_pair_hidden: int, optional
      Size of hidden layer for each pair of atoms.
    n_graph_feat: int, optional
      Size of latent embedding for each atom
    n_weave_layers: int, optional
      Number of convolutional layers applied
    """
    super(WeaveEnc, self).__init__()
    self.n_node_feat = n_node_feat
    self.n_pair_feat = n_pair_feat
    self.n_node_hidden = n_node_hidden
    self.n_pair_hidden = n_pair_hidden
    self.n_latent_feat = n_latent_feat
    self.n_weave_layers = n_weave_layers
    
    weave_module = []
    in_node_feat = self.n_node_feat
    in_pair_feat = self.n_pair_feat
    for i in range(self.n_weave_layers):
      if i == self.n_weave_layers - 1: update_pair = False
      else: update_pair = True
      weave_module.append(WeaveLayer(in_node_feat, 
                                     in_pair_feat, 
                                     self.n_node_hidden,
                                     self.n_pair_hidden,
                                     update_pair=update_pair))
      in_node_feat = self.n_node_hidden
      in_pair_feat = self.n_pair_hidden
    self.weave_module = nn.ModuleList(weave_module)
    
    self.mean_fc = nn.Sequential(
          nn.Linear(self.n_node_hidden, self.n_node_hidden),
          nn.ReLU(True),
          nn.Linear(self.n_node_hidden, self.n_node_hidden),
          nn.ReLU(True),
          nn.Linear(self.n_node_hidden, self.n_latent_feat))
    self.logstd_fc = nn.Sequential(
          nn.Linear(self.n_node_hidden, self.n_node_hidden),
          nn.ReLU(True),
          nn.Linear(self.n_node_hidden, self.n_node_hidden),
          nn.ReLU(True),
          nn.Linear(self.n_node_hidden, self.n_latent_feat))

  def forward(self, node_feats, pair_feats, A):
    """ node_feats: batch_size * n_nodes * node_feat
        pair_feats: batch_size * n_nodes * n_nodes * pair_feat
    """
    for i in range(self.n_weave_layers):
      node_feats, pair_feats = self.weave_module[i](node_feats, pair_feats)
    z_mean = self.mean_fc(node_feats)
    z_log_std = self.logstd_fc(node_feats)
    return z_mean, z_log_std

class GraphConvEnc(nn.Module):

  def __init__(self,
               n_node_feat=23,
               n_pair_feat=15,
               n_graphconv=[32, 64, 128],
               n_latent_feat=128,
               **kwargs):
    """
    Parameters
    ----------
    n_node_feat: int, optional
      Number of atom features
    n_graphconv: list of int
      Width of channels for the Graph Convolution Layers
    n_graph_feat: int, optional
      Size of latent embedding for each atom
    """
    super(GraphConvEnc, self).__init__(**kwargs)
    self.n_latent_feat = n_latent_feat
    self.n_graphconv = n_graphconv
    self.n_node_feat = n_node_feat
    self.n_pair_feat = n_pair_feat
    
    gc_module = []
    in_node_feat = self.n_node_feat
    in_pair_feat = self.n_pair_feat
    for n_hidden in self.n_graphconv:
      gc_module.append(GraphConvLayer(in_node_feat, in_pair_feat, n_hidden))
      in_node_feat = n_hidden
      in_pair_feat = n_hidden
      
    self.gc_module = nn.ModuleList(gc_module)
    
    self.mean_fc = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.ReLU(True),
          nn.Linear(n_hidden, self.n_latent_feat))
    self.logstd_fc = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.ReLU(True),
          nn.Linear(n_hidden, self.n_latent_feat))

  def forward(self, node_feats, pair_feats, A):
    """ node_feats: batch_size * n_nodes * node_feat
        A: batch_size * n_nodes * n_nodes
    """
    for i in range(len(self.n_graphconv)):
      node_feats, pair_feats = self.gc_module[i](node_feats, pair_feats, A)
    z_mean = self.mean_fc(node_feats)
    z_log_std = self.logstd_fc(node_feats)
    return z_mean, z_log_std
