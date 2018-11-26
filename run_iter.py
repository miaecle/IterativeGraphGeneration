#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:31:27 2018

@author: zqwu
"""
import os
import numpy as np
import torch as t
from models.enc import GraphConvEnc, WeaveEnc
from models.dec import InnerProductDecoder, AffineDecoder
from models.vae import GraphVAE, IterativeRefGraphVAE, Trainer
from utils.load_molecules import load_molecules
from utils.metrics import eval_reconstruction_rate

class Config:
    lr = 0.001
    batch_size = 128
    max_epoch = 2000
    gpu = True
opt=Config()

subset = 6
Mols = load_molecules('./data/qm9_subset' + str(subset) + '.smi')
n_mols = len(Mols)
np.random.seed(123)
np.random.shuffle(Mols)
train_mols = Mols[:int(0.8*n_mols)]
valid_mols = Mols[int(0.8*n_mols):int(0.9*n_mols)]
test_mols = Mols[int(0.9*n_mols):]

enc = GraphConvEnc(n_node_feat=Mols[0][0].shape[1])  
dec = AffineDecoder()
vae = IterativeRefGraphVAE(enc, dec, n_iterref=2, gpu=opt.gpu)
model = Trainer(vae, opt, lambd=0.5, kl_rate=0.)

best_valid_score = 0.
train_scores = []
valid_scores = []
if __name__ == '__main__':
  for i in range(100):
    model.train(train_mols, n_epochs=10)
    train_scores.append(eval_reconstruction_rate(train_mols, model.predict(train_mols)))
    valid_scores.append(eval_reconstruction_rate(valid_mols, model.predict(valid_mols)))
    if valid_scores[-1] > best_valid_score:
      best_valid_score = valid_scores[-1]
      print("New Best: %f" % best_valid_score)
      model.save('./model_iter_' + str(subset) + '.pth')
  model.load('./model_iter_' + str(subset) + '.pth')
  print(eval_reconstruction_rate(train_mols, model.predict(train_mols)))
  print(eval_reconstruction_rate(valid_mols, model.predict(valid_mols)))
  print(eval_reconstruction_rate(test_mols, model.predict(test_mols)))
