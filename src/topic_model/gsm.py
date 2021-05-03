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
    def __init__(self, encode_dims=[2000,768,20],decode_dims=[20,768,2000],dropout=0.8):

        super(GSM, self).__init__()
        self.hidden_encoder = nn.Linear(encode_dims[0],encode_dims[1]) #[2000,768]
        self.fc_mu = nn.Linear(encode_dims[1],encode_dims[2]) #[768,20]
        self.fc_logvar = nn.Linear(encode_dims[1],encode_dims[2]) #[768,20]

        self.hidden_decoder_1 = nn.Linear(decode_dims[0],decode_dims[1]) #[20,768]
        self.hidden_decoder_2 = nn.Linear(decode_dims[1],decode_dims[2]) #[768,2000]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])

        # self.v  = nn.Linear( decode_dims[-2], decode_dims[-1]) #[2007, 768] 是反着来的
        # self.t = nn.Linear(decode_dims[-2], decode_dims[0]) #[20, 768]
        # nn.init.xavier_uniform_(self.v.weight)
        # nn.init.xavier_uniform_(self.t.weight)
        # self.t = torch.empty(decode_dims[0], 768).cuda() #[ topics, 100]
        # self.v = torch.empty(decode_dims[-1], 768).cuda() #[ |V|, 100] 是反着来的
        # nn.init.xavier_normal_(self.t)
        # nn.init.xavier_normal_(self.v)
        self.v  = nn.Embedding( decode_dims[-1], decode_dims[-2]) #[2000, 768] 是反着来的
        self.t = nn.Embedding(decode_dims[0], decode_dims[-2]) #[20, 768]
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.t.weight)
        self.theta = None
        self.beta = None

    def encode(self, x):
        hid = F.relu(self.dropout(self.hidden_encoder(x))) #[batch,768] = [batch,2000] * [2000,768]
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid) #[batch,20] = [batch,768] * [768,20]
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

    # def decode(self, theta):#theta = [batch, topics]
    #     hid = self.hidden_decoder_1(theta) #[batch,500] = [batch,20] * [20,500]
    #     hid = F.relu(self.dropout(hid))
    #     p_x = self.hidden_decoder_2(hid) #[batch,2000] = [batch,500] * [500,2000]
    #     return p_x
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

if __name__ == '__main__':
    model = GSM(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)