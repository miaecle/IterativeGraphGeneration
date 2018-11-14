#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:28:02 2018

@author: zqwu
"""
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np

class InnerProductDecoder(nn.Module):
  def __init__(self, 
               n_atom_type=4, 
               n_latent_feat=128,
               **kwargs):
    super(InnerProductDecoder, self).__init__(**kwargs)
    self.n_latent_feat = n_latent_feat
    self.n_atom_type = n_atom_type
    self.atom_dec = nn.Sequential(
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, self.n_atom_type))
    self.bond_dec1 = nn.Sequential(
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, self.n_latent_feat))
    self.bond_dec2 = nn.Sequential(
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, 2))
    

  def forward(self, z):
    atom_pred = F.log_softmax(self.atom_dec(z), 2)
    bond_feat = self.bond_dec1(z)
    n_atoms = z.shape[1]
    inner_product = bond_feat.unsqueeze(2).expand(-1, -1, n_atoms, -1) *\
        bond_feat.unsqueeze(1).expand(-1, n_atoms, -1, -1)
    bond_logit = self.bond_dec2(inner_product)
    bond_pred = F.log_softmax(bond_logit, 3)
    return atom_pred, bond_pred

class AffineDecoder(nn.Module):
  def __init__(self, 
               n_atom_type=4,
               n_latent_feat=128,
               **kwargs):
    super(AffineDecoder, self).__init__(**kwargs)
    self.n_latent_feat = n_latent_feat
    self.n_atom_type = n_atom_type
    self.atom_dec = nn.Sequential(
          nn.Linear(self.n_latent_feat, self.n_latent_feat),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat, self.n_atom_type))
    self.bond_dec = nn.Sequential(
          nn.Linear(self.n_latent_feat * 2, self.n_latent_feat * 2),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat * 2, self.n_latent_feat * 2),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat * 2, self.n_latent_feat * 2),
          nn.ReLU(True),
          nn.Linear(self.n_latent_feat * 2, 2))

  def forward(self, z):
    atom_pred = F.log_softmax(self.atom_dec(z), 2)
    n_atoms = z.shape[1]
    bond_ij = t.cat([z.unsqueeze(2).expand(-1, -1, n_atoms, -1), 
                     z.unsqueeze(1).expand(-1, n_atoms, -1, -1)], 3)
    bond_ji = t.cat([z.unsqueeze(1).expand(-1, n_atoms, -1, -1), 
                     z.unsqueeze(2).expand(-1, -1, n_atoms, -1)], 3)
    bond_pred = F.log_softmax(self.bond_dec(bond_ij) + self.bond_dec(bond_ji), 3)
    return atom_pred, bond_pred
