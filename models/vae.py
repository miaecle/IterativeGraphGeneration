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
from layers import kl_normal
from torch.autograd import Variable
from torch.optim import Adam

class GraphVAE(nn.Module):
  def __init__(self, enc, dec, gpu=False, **kwargs):
    super(GraphVAE, self).__init__(**kwargs)
    self.enc = enc
    self.dec = dec
    assert self.enc.n_latent_feat == self.dec.n_latent_feat
    self.gpu = gpu
  
  def forward(self, node_feats, pair_feats, A, random=True):
    z_mean, z_logstd = self.enc(node_feats, pair_feats, A)
    if random:
      r = t.randn_like(z_mean)
      if self.gpu: r = r.cuda()
      z = z_mean + r * t.exp(z_logstd)
    else:
      z = z_mean
    atom_pred, bond_pred = self.dec(z)
    return atom_pred, bond_pred, z_mean, z_logstd
  
  def predict(self, node_feats, pair_feats, A):
    outs = self.forward(node_feats, pair_feats, A, random=False)
    return t.exp(outs[0]), t.exp(outs[1])

class Trainer(object):
  def __init__(self, net, opt, lambd=0.1):
    self.net = net
    self.atom_label_loss = nn.NLLLoss(reduce=False)
    self.bond_label_loss = nn.NLLLoss(reduce=False)
    self.opt = opt
    if self.opt.gpu:
      self.net = self.net.cuda()
      self.atom_label_loss = self.atom_label_loss.cuda()
      self.bond_label_loss = self.bond_label_loss.cuda()
    self.lambd = lambd
  
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
        atom_label = t.argmax(batch[0][:, :, :4], 2).long()
        bond_label = batch[2].long()

        rec = self.atom_label_loss(atom_pred.transpose(1, 2), 
                                   atom_label).sum(1) + \
              self.bond_label_loss(bond_pred.transpose(2, 3).transpose(1, 2), 
                                   bond_label).sum(1).sum(1) * self.lambd
        rec = (rec * weights).sum()
        kl = kl_normal(z_mean, t.exp(z_logstd), t.zeros_like(z_mean), t.ones_like(z_logstd)).sum(1)
        kl = (kl * weights).sum()
        
        loss = rec + 0.01*kl
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
