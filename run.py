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
from models.vae import GraphVAE, Trainer
from utils.load_molecules import load_molecules

class Config:
    lr = 0.001
    batch_size = 32
    max_epoch = 2000
    gpu = True
opt=Config()

Mols = load_molecules('./data/qm9_subset6.smi')
n_mols = len(Mols)
train_mols = Mols[:int(32)]
valid_mols = Mols[int(0.8*n_mols):]

enc = GraphConvEnc()
dec = AffineDecoder()
vae = GraphVAE(enc, dec, gpu=opt.gpu)
model = Trainer(vae, opt)

if __name__ == '__main__':
  model.train(train_mols)
  model.save('./temp_save.pth')
  model.load('./temp_save.pth')
  train_preds = model.predict(train_mols)
  print(train_preds[0][1][0, :, :, 1])
  print(train_mols[0][2])
  print(train_preds[0][0])
  print(train_mols[0][0][:, :4])
  
