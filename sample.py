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
smi_arr = []
mol_arr = []
ct = 0
len_iter = 0
while True:
    len_iter+=1
    if ct == 100:
        break
    try:
        sample = model.sample(1, 8)[0]
        sample = [sample[0][0], sample[1][0],sample[2][0]]
        new_mol = Chem.Mol()
        new_mol = Chem.EditableMol(new_mol)
        for i in range(8):
            new_atom_type = atom_types[np.argmax(sample[0][i])]
            new_atom = Chem.Atom(new_atom_type)
            ind = new_mol.AddAtom(new_atom)
            assert ind == i
        for i in range(8):
            for j in range(i+1, 8):
                if sample[2][i, j, 1] > 0.5:
                    new_mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
        mol = new_mol.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        if '.' in smi:
            continue
        print(smi)
        smi_arr.append(smi)
        mol_arr.append(mol)
        ct += 1
    except:
        pass
