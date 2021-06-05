#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vae.py
@Time    :   2020/09/30 15:07:10
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

# GSM model
class GSM(nn.Module):
    def __init__(self,voc_size, n_topic=50, dropout=0.0):

        super(GSM, self).__init__()
        encode_dims=[voc_size,1024,300,n_topic]#[3000,1024,300,50]
        decode_dims=[n_topic,300,voc_size]#[50,300,3000]
        print("encode_dims", encode_dims)
        print("decode_dims", decode_dims)
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1]) #[768,20]
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1]) #[768,20]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])

        self.v  = nn.Embedding( decode_dims[-1], decode_dims[-2]) #[2000, 64] 是反着来的
        self.t = nn.Embedding(decode_dims[0], decode_dims[-2]) #[20, 64]
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.t.weight)
        self.theta = None
        self.beta = None

    def encode(self, x):
        hid = x
        for i,layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid) #[batch,20] = [batch,500] * [500,20]
        return mu, log_var

    def inference(self,x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x,dim=1)
        return theta
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, theta): #theta = [batch, topics]
        self.beta = torch.mm(self.t.weight, self.v.weight.transpose(0,1)) #[topic, |v|] = [20, 768] * [768 * 2000]
        self.beta = F.softmax(self.beta, 1) #[topics,2000]
        p_x = torch.mm(theta, self.beta) #[batch,2000] = [batch,topics] * [topics,2000]
        return p_x

        
    def forward(self, x, use_gsm=True):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 
        if use_gsm:
            self.theta = torch.softmax(_theta,dim=1) #[batch,topics]
        else:
            self.theta = _theta
        x_reconst = self.decode(self.theta)
        return x_reconst, mu, log_var, self.theta, self.beta, self.t.weight
