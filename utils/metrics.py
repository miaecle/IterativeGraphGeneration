#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:35:47 2018

@author: zqwu
"""

import numpy as np

def eval_reconstruction_rate(mols, preds):
  n_samples = len(mols)
  assert len(preds) == n_samples
  correct_ct = 0
  mask = None
  for mol, pred in zip(mols, preds):
    if mask is None:
      mask = 1 - np.eye(mol[2].shape[0])
    # labels
    node_label = np.argmax(mol[0][:, :4], 1)
    A_label = mol[2]
    bond_type_label = (np.argmax(mol[1][:, :, :4], 2) + 1) * A_label * mask   # Separate bonded and non-bonded
    #preds
    node_pred = np.argmax(pred[0], 1)   
    A_pred = (pred[2] > 0.5)*1
    bond_type_pred = (np.argmax(pred[1], 2)  + 1) * A_pred * mask
    #check
    if np.all([np.allclose(A_label, A_pred), 
      np.allclose(bond_type_label, bond_type_pred), 
      np.allclose(node_pred, node_label)]):
      correct_ct += 1
  return float(correct_ct)/n_samples