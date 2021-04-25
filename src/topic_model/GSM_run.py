#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NTM_run.py
@Time    :   2020/09/30 15:52:35
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import os
import re
import torch
# torch.cuda.set_device(3)
import pickle
import argparse
import logging
import time
from models import GSM
from utils import *
from dataset import DocDataset
from multiprocessing import cpu_count
#from torch.utils.data import Dataset,DataLoader

parser = argparse.ArgumentParser('GSM topic model')
parser.add_argument('--train_data_file',type=str,default='20news_train',help='Taskname e.g cnews10k')
parser.add_argument('--test_data_file',type=str,default='20news_test',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.0134,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=1000,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=50,help='Num of topics')
parser.add_argument('--hidden_size',type=int,default=500,help='hidden_size')
parser.add_argument('--learning_rate',type=int,default=5e-5,help='learning_rate')
parser.add_argument('--log_every',type=int,default=1,help='log_every')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--batch_size',type=int,default=64,help='Batch size (default=512)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--ckpt',type=str,default=None,help='Checkpoint path')

args = parser.parse_args()

def main():
    global args
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    log_every = args.log_every
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    batch_size = args.batch_size
    auto_adj = args.auto_adj
    ckpt = args.ckpt

    device = torch.device('cuda')
    trainSet = DocDataset(train_data_file,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    testSet = DocDataset(test_data_file,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    # if auto_adj:
    #     no_above = docSet.topk_dfs(topk=20)
    #     docSet = DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    
    voc_size = trainSet.vocabsize
    print('train voc size:',voc_size)
    print("train:", type(trainSet), len(trainSet))
    print("test:", type(testSet), len(testSet))
    if ckpt:
        checkpoint=torch.load(ckpt)
        param.update({"device": device})
        model = GSM(**param)
        model.train(train_data=trainSet,test_data=testSet,batch_size=batch_size,learning_rate=learning_rate,num_epochs=num_epochs,log_every=log_every,ckpt=checkpoint)
    else:
        model = GSM(bow_dim=voc_size,n_topic=n_topic,hidden_size=hidden_size,device=device)
        model.train(train_data=trainSet,test_data=testSet,batch_size=batch_size,learning_rate=learning_rate,num_epochs=num_epochs,log_every=log_every)
    #model.evaluate(test_data=docSet)
    save_name = f'./ckpt/GSM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save(model.vae.state_dict(),save_name)
    txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
    with open('topic_dist_gsm.txt','w',encoding='utf-8') as wfp:
        for t,e in zip(txt_lst,embeds):
            wfp.write(f'{e}:{t}\n')
    pickle.dump({'txts':txt_lst,'embeds':embeds},open('gsm_embeds.pkl','wb'))

if __name__ == "__main__":
    main()
