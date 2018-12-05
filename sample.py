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
from rdkit import Chem

class Config:
    lr = 0.0001
    batch_size = 128
    max_epoch = 2000
    gpu = True
    mpm = False
opt=Config()

enc = GraphConvEnc(n_node_feat=32)  
dec = AffineDecoder()
#vae = IterativeRefGraphVAE(enc, dec, n_iterref=2, gpu=opt.gpu)
vae = GraphVAE(enc, dec,  gpu=opt.gpu)
model = Trainer(vae, opt, lambd=0.5, kl_rate=0.1)
model.load('./model_qm9_1000.pth')

atom_types = ['C', 'N', 'O', 'F']
bond_types = [Chem.rdchem.BondType.SINGLE, 
                             Chem.rdchem.BondType.DOUBLE, 
                             Chem.rdchem.BondType.TRIPLE,
                             Chem.rdchem.BondType.AROMATIC]
ct = 0
while True:
  if ct == 100:
    break
  try:
    sample = model.sample(1, 8)[0]
    sample = [sample[0][0], sample[1][0], sample[2][0]]
    new_mol = Chem.Mol()
    new_mol = Chem.EditableMol(new_mol)
    # add nodes
    for i in range(8):
      new_atom_type = atom_types[np.argmax(sample[0][i])]
      new_atom = Chem.Atom(new_atom_type)
      ind = new_mol.AddAtom(new_atom)
      assert ind == i
    # add bonds with bond type

    for i in range(8):
      for j in range(i+1, 8):
        if sample[2][i, j, 1] > 0.5:
          bond_type = bond_types[np.argmax(sample[1][i,j,:])]
          new_mol.AddBond(i, j, bond_type)
    mol = new_mol.GetMol()
    Chem.SanitizeMol(mol)
    smi = Chem.MolToSmiles(mol)
    if '.' in smi:
      continue
    print(smi)
    ct += 1
  except:
    pass
