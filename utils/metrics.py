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
  for mol, pred in zip(mols, preds):
    node_label = np.argmax(mol[0][:, :4], 1)
    A_label = mol[2]
    node_pred = np.argmax(pred[0], 1)
    A_pred = (pred[1] > 0.5)*1
    if np.allclose(A_label, A_pred) and np.allclose(node_pred, node_label):
      correct_ct += 1
  return float(correct_ct)/n_samples