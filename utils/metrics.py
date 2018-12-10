#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:35:47 2018

@author: zqwu
"""

import numpy as np
from rdkit import Chem

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

def sample_molecules(model, n, return_cts=False):
  atom_types = ['C', 'N', 'O', 'F']
  bond_types = [Chem.rdchem.BondType.SINGLE, 
                               Chem.rdchem.BondType.DOUBLE, 
                               Chem.rdchem.BondType.TRIPLE,
                               Chem.rdchem.BondType.AROMATIC]
  ct = 0
  valid_ct = 0
  smiles = []
  while True:
    if valid_ct == n:
      break
    try:
      sample = model.sample(1, 9)[0]
      sample = [sample[0][0], sample[1][0], sample[2][0]]
      new_mol = Chem.Mol()
      new_mol = Chem.EditableMol(new_mol)
      # add nodes
      for i in range(9):
        new_atom_type = atom_types[np.argmax(sample[0][i])]
        new_atom = Chem.Atom(new_atom_type)
        ind = new_mol.AddAtom(new_atom)
        assert ind == i
      # add bonds with bond type
  
      for i in range(9):
        for j in range(i+1, 9):
          if sample[2][i, j, 1] > 0.5:
            bond_type = bond_types[np.argmax(sample[1][i,j,:])]
            new_mol.AddBond(i, j, bond_type)
      mol = new_mol.GetMol()
      Chem.SanitizeMol(mol)
      smi = Chem.MolToSmiles(mol)
      if '.' in smi:
        continue
      smiles.append(smi)
      valid_ct += 1
    except:
      pass
    finally:
      ct += 1
  print("Count: %d\t Valid Count: %d" % (ct, valid_ct))
  if return_cts:
    return smiles, valid_ct, ct
  else:
    return smiles

def validity(model):
  _, valid_ct, ct = sample_molecules(model, 1000)
  return float(valid_ct)/ct

def variety(model, smis=None):
  if smis is None:
    smis = sample_molecules(model, 1000)
  return len(set(smis))/float(len(smis))

def novelty(model, train_smis, smis=None):
  if smis is None:
    smis = sample_molecules(model, 1000)
  novel_ct = 0
  for smi in smis:
    if not smi in train_smis:
      novel_ct += 1
  return float(novel_ct) / len(smis)