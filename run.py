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
    max_epoch = 1000
    gpu = True
opt=Config()

Mols = load_molecules('./data/qm9_subset7.smi')
train_mols = Mols[:2557]
valid_mols = Mols[2557:]

enc = GraphConvEnc()
dec = AffineDecoder()
vae = GraphVAE(enc, dec, gpu=opt.gpu)
model = Trainer(vae, opt)

model.train(train_mols)