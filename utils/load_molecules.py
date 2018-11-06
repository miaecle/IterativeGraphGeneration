#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:05:20 2018

@author: zqwu
"""
from rdkit import Chem
import numpy as np

NUM_ATOM_FEATURES = 23
NUM_BOND_FEATURES = 7
NUM_PAIR_FEATURES = 15

def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def atom_features(mol):
  N = mol.GetNumAtoms()
  atom_feats = []
  for atom in mol.GetAtoms():
    results = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'UNK']) + \
              one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 'UNK']) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 'UNK']) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                                    Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetIsAromatic()]
    atom_feats.append(np.array(results).astype(float))
  assert len(atom_feats) == N
  atom_feats = np.stack(atom_feats, 0)
  return atom_feats


def bond_features(bond):
  bt = one_of_k_encoding_unk(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, 
                             Chem.rdchem.BondType.DOUBLE, 
                             Chem.rdchem.BondType.TRIPLE,
                             Chem.rdchem.BondType.AROMATIC, 'UNK']) + \
       [bond.GetIsConjugated(), bond.IsInRing()]
  return np.array(bt).astype(float)

  
def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
  distance = np.zeros((num_atoms, max_distance))
  radial = 0
  # atoms `radial` bonds away from `a1`
  adj_list = set(canon_adj_list[a1])
  # atoms less than `radial` bonds away
  all_list = set([a1])
  while radial < max_distance:
    distance[list(adj_list), radial] = 1
    all_list.update(adj_list)
    # find atoms `radial`+1 bonds away
    next_adj = set()
    for adj in adj_list:
      next_adj.update(canon_adj_list[adj])
    adj_list = next_adj - all_list
    radial = radial + 1
  return distance


def pair_features(mol):
  max_distance = 7
  N = mol.GetNumAtoms()
  features = np.zeros((N, N, NUM_BOND_FEATURES + max_distance + 1))
  
  edge_list = []
  for b in mol.GetBonds():
    edge_list.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    features[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), :NUM_BOND_FEATURES] = bond_features(b)
    features[b.GetEndAtomIdx(), b.GetBeginAtomIdx(), :NUM_BOND_FEATURES] = bond_features(b)
  
  
  canon_adj_list = [[] for _ in range(N)]
  for edge in edge_list:
    canon_adj_list[edge[0]].append(edge[1])
    canon_adj_list[edge[1]].append(edge[0])
    
  rings = mol.GetRingInfo().AtomRings()
  for a1 in range(N):
    # graph distance between two atoms
    distance = find_distance(a1, N, canon_adj_list, max_distance=max_distance)
    features[a1, :, NUM_BOND_FEATURES:(NUM_BOND_FEATURES+max_distance)] = distance
    for ring in rings:
      if a1 in ring:
        # `bt_len`-th feature is if the pair of atoms are in the same ring
        features[a1, ring, -1] = 1
        features[a1, a1, -1] = 0.
  return features
  
def load_molecules(path, raw=False, padding=False):
  Mols = []
  i = 0
  with open(path, 'r') as f:
    for smi in f.readlines():
      i += 1
      if i%1000 == 0:
        print("On molecule %d" % i)
      # Cut \n
      mol = Chem.MolFromSmiles(smi[:-1])
      assert mol is not None
      if raw:
        Mols.append(mol)
      else:
        atom_f = atom_features(mol)
        pair_f = pair_features(mol)
        A = pair_f[:, :, NUM_BOND_FEATURES] # Adjacency Matrix
        Mols.append([atom_f, pair_f, A])
  
  if padding:
    max_num_atoms = max([m[0].shape[0] for m in Mols])
    for i, mol in enumerate(Mols):
      n = mol[0].shape[0]
      Mols[i][0] = np.pad(mol[0], ((0, max_num_atoms - n), (0, 0)), 'constant')
      Mols[i][1] = np.pad(mol[1], ((0, max_num_atoms - n), (0, max_num_atoms - n), (0, 0)), 'constant')
      Mols[i][2] = np.pad(mol[2], ((0, max_num_atoms - n), (0, max_num_atoms - n)), 'constant')
  return Mols