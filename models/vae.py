#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:19:20 2018

@author: zqwu
"""
from torch import nn
import torch as t
import torch.nn.functional as F
import numpy as np
from layers import kl_normal, IterativeRef
from torch.autograd import Variable
from torch.optim import Adam
import mpm 
import scipy.optimize


class GraphVAE(nn.Module):
  def __init__(self, enc, dec, gpu=False, **kwargs):
    super(GraphVAE, self).__init__(**kwargs)
    self.enc = enc
    self.dec = dec
    assert self.enc.n_latent_feat == self.dec.n_latent_feat
    self.gpu = gpu
    self.max_num_nodes = 6# TODO: reference NUM_ATOM_FEATURES, also change name to num_feat

  
  def forward(self, node_feats, pair_feats, A, random=True):
    # print('shapes', 'node_feats, pair_feats, A')
    # print(node_feats.shape, pair_feats.shape, A.shape)
    z_mean, z_logstd = self.enc(node_feats, pair_feats, A)
    if random:
      r = t.randn_like(z_mean)
      if self.gpu: r = r.cuda()
      z = z_mean + r * t.exp(z_logstd)
    else:
      z = z_mean

    atom_pred, bond_pred = self.dec(z)

    # recon_adj_lower = mpm.recover_adj_lower(atom_pred)
    # recon_adj_tensor = mpm.recover_full_adj_from_lower(recon_adj_lower)

    # Find the similarity matrix, S(i,j, a,b) that is of size: 
    #(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes,self.max_num_nodes)


    # print('shapes', 'atom_pred, bond_pred')
    # print(atom_pred.shape, bond_pred.shape)
    return atom_pred, bond_pred, z_mean, z_logstd
  
  def predict(self, node_feats, pair_feats, A):
    outs = self.forward(node_feats, pair_feats, A, random=False)
    return t.exp(outs[0]), t.exp(outs[1])

class IterativeRefGraphVAE(nn.Module):
  def __init__(self, 
               enc, 
               dec, 
               n_iterref=2,
               n_extra_input=0,
               gpu=False, 
               **kwargs):
    super(IterativeRefGraphVAE, self).__init__(**kwargs)
    self.enc = enc
    self.dec = dec
    self.n_iterref = n_iterref
    assert self.enc.n_latent_feat == self.dec.n_latent_feat
    
    
    self.n_iterref_input = self.enc.n_latent_feat + n_extra_input
    self.iterref_layers = []
    for _ in range(n_iterref):
      self.iterref_layers.append(IterativeRef(self.enc.n_latent_feat, 
                                              self.n_iterref_input))
    self.iterref_layers = nn.ModuleList(self.iterref_layers)
    self.gpu = gpu

  def forward(self, node_feats, pair_feats, A, random=True):
    z_mean, z_logstd = self.enc(node_feats, pair_feats, A)
    if random:
      r = t.randn_like(z_mean)
      if self.gpu: r = r.cuda()
      z = z_mean + r * t.exp(z_logstd)
    else:
      z = z_mean
    
    z_intermediate = z
    for layer in self.iterref_layers:
      z_intermediate = layer(z, z_intermediate)
      
    atom_pred, bond_pred = self.dec(z_intermediate)
    return atom_pred, bond_pred, z_mean, z_logstd

  def predict(self, node_feats, pair_feats, A):
    outs = self.forward(node_feats, pair_feats, A, random=False)
    return t.exp(outs[0]), t.exp(outs[1])


class Trainer(object):
  def __init__(self, net, opt, mpm=True, lambd=0.1, kl_rate=1.):
    self.net = net
    self.atom_label_loss = nn.NLLLoss(reduce=False)
    self.bond_label_loss = nn.NLLLoss(reduce=False)
    self.opt = opt

    self.mpm = opt.mpm

    if self.opt.gpu:
      self.net = self.net.cuda()
      self.atom_label_loss = self.atom_label_loss.cuda()
      self.bond_label_loss = self.bond_label_loss.cuda()
    self.lambd = lambd
    self.kl_rate = kl_rate
  
  def assemble_batch(self, data, sample_weights=None, batch_size=None):
    if batch_size is None:
      batch_size = self.opt.batch_size
    if sample_weights is None:
      sample_weights = [1] * len(data)

    # Assemble samples with similar lengths to a batch
    data_batches = []
    for i in range(int(np.ceil(len(data)/float(batch_size)))):
      batch = data[i * batch_size:min((i+1) * batch_size, len(data))]
      batch_weights = sample_weights[i * batch_size:min((i+1) * batch_size, len(data))]
      
      batch_node_feats = [b[0] for b in batch]
      batch_pair_feats = [b[1] for b in batch]
      batch_A = [b[2] for b in batch]
      
      if len(batch_weights) < batch_size:
        pad_length = batch_size - len(batch_weights)
        batch_weights.extend([0.] * pad_length)
        batch_node_feats.extend([batch_node_feats[0]] * pad_length)
        batch_pair_feats.extend([batch_pair_feats[0]] * pad_length)
        batch_A.extend([batch_A[0]] * pad_length)
        
      data_batches.append([np.stack(batch_node_feats, axis=0), 
                           np.stack(batch_pair_feats, axis=0), 
                           np.stack(batch_A, axis=0),
                           np.array(batch_weights).reshape(-1)])
    return data_batches
  
  def save(self, path):
    t.save(self.net.state_dict(), path)
  
  def load(self, path):
    s_dict = t.load(path, map_location=lambda storage, loc: storage)
    self.net.load_state_dict(s_dict)
  
  def train(self, train_data, sample_weights=None, n_epochs=None, **kwargs):
    
    optimizer = Adam(self.net.parameters(),
                     lr=self.opt.lr,
                     betas=(.9, .999))
    self.net.zero_grad()
    
    if n_epochs is None:
      n_epochs = self.opt.max_epoch
    n_points = len(train_data)
    data_batches = self.assemble_batch(train_data, sample_weights=sample_weights)
      
    for epoch in range(n_epochs):      
      accum_rec_loss = 0
      accum_kl_loss = 0
      print ('start epoch {epoch}'.format(epoch=epoch))
      for dat in data_batches:
        batch = []
        for i, item in enumerate(dat):
          batch.append(Variable(t.from_numpy(item).float()))
        if self.opt.gpu:
          for i, item in enumerate(batch):
            batch[i] = item.cuda()
            
        node_feats, pair_feats, A, weights = batch
        atom_pred, bond_pred, z_mean, z_logstd = self.net(node_feats, pair_feats, A, random=True)

        if self.mpm:
          atom_label, bond_label = self.match_pool_matching(atom_pred, bond_pred, node_feats, A)
        else:
          atom_label = t.argmax(node_feats[:, :, :4], 2).float() #(32, 6)
          bond_label = A.float()  #(32, 6, 6), 

        rec = self.atom_label_loss(atom_pred.transpose(1, 2), # 32x4x6
                                   atom_label.long()).sum(1) + \
              self.bond_label_loss(bond_pred.transpose(2, 3).transpose(1, 2), #32x2x6x6
                                   bond_label.long()).sum(1).sum(1) * self.lambd
        rec = (rec * weights).sum()
        kl = kl_normal(z_mean, t.exp(z_logstd), t.zeros_like(z_mean), t.ones_like(z_logstd)).sum(1)
        kl = (kl * weights).sum()
        
        loss = rec + self.kl_rate*kl
        loss.backward()
        accum_rec_loss += rec
        accum_kl_loss += kl
        optimizer.step()
        self.net.zero_grad()
      print ('epoch {epoch} loss: {rec_loss}, {kl_loss}'.format(epoch=epoch, 
          rec_loss=accum_rec_loss.data[0]/n_points,
          kl_loss=accum_kl_loss.data[0]/n_points))
    return
      
  def predict(self, test_data):
    test_batches = self.assemble_batch(test_data, batch_size=1)
    preds = []
    for dat in test_batches:
      sample = []
      for i, item in enumerate(dat):
        sample.append(Variable(t.from_numpy(item).float()))
      if self.opt.gpu:
        for i, item in enumerate(sample):
          sample[i] = item.cuda() # No extra first dimension
      node_feats, pair_feats, A, weights = sample
      
      node_pred, bond_pred = self.net.predict(node_feats, pair_feats, A)
      node_pred = node_pred.data.to(t.device('cpu')).numpy()
      bond_pred = bond_pred.data.to(t.device('cpu')).numpy()
      preds.append((node_pred[0], bond_pred[0, :, :, 1]))
    return preds

  def match_pool_matching(self, atom_pred, bond_pred, node_feats, A,max_iters=50):
    # decoding: 
    # node_feats size: batch_size x num_nodes x num_node_feat (32 x6x 23) 
    # atom_pred size: batch_size x num_nodes x num_node_classes (32 x6x 4) 
    # pair_feats size: batch_size x num_nodes x num_nodes x num_pair_feat (32x6x6x15)
    # bond_pred size: batch_size x num_nodes x num_edge classes (32x6x6x2)

    atom_label = t.argmax(node_feats[:, :, :4], 2).float() #(32, 6)
    bond_label = A.float()  #(32, 6, 6), 
    bond_label_deg = bond_label.sum(2)# 32x6 -
    bond_label_diag = bond_label.clone() #+ t.diag(t.ones(self.net.max_num_nodes)).unsqueeze(0)  #need to make diagonals one
    bond_label_diag = bond_label_diag.float() #(32, 6, 6), 
    bond_pred_bin = t.argmax(t.exp(bond_pred),3).float() # 32x 6x6batch_size x num_nodes x num_nodes (last two dim are A for each batch)
    bond_pred_deg = bond_pred_bin.sum(2).float() # 32x6
    # print('finished getting feats')
    # get similarity matrix (32,6,6,6,6)
    S = mpm.edge_similarity_matrix( A_label=bond_label_diag, A_pred=t.exp(bond_pred[:,:,:,1]), 
                        feat_label = bond_label_deg, feat_pred =bond_pred_deg)
    # print('S', S.shape)

    # initialization quadratic programming task
    batch_size = atom_label.shape[0]
    init_corr = 1 / self.net.max_num_nodes
    init_assignment = t.ones(batch_size, self.net.max_num_nodes, self.net.max_num_nodes) * init_corr
    assignment = mpm.mpm(init_assignment, S,max_iters=max_iters)
    # print('Assignment: ', assignment)

    # matching
    for idx in range(batch_size):
      row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment[idx,:,:].detach().numpy())
      if any(row_ind!=col_ind):
        print(row_ind,col_ind)
      atom_label[idx,:] = mpm.permute(atom_label[idx,:], row_ind, col_ind)
      bond_label_diag[idx,:,:] = mpm.permute(bond_label_diag[idx,:,:], row_ind, col_ind)
    # print('after matching')
    # raise
    return atom_label, bond_label_diag
