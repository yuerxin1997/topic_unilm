import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from .vae import VAE
import matplotlib.pyplot as plt
import sys
import codecs
import time
sys.path.append('..')
from utils import evaluate_topic_quality, smooth_curve

class GSM:
    def __init__(self,bow_dim=2000,n_topic=50,hidden_size=500,device=None):
        self.bow_dim = bow_dim #voc_size 3007
        self.n_topic = n_topic #20
        self.hidden_size = hidden_size
        #TBD_fc1
        self.vae = VAE(encode_dims=[bow_dim,hidden_size,n_topic],decode_dims=[n_topic,hidden_size,bow_dim],dropout=0.8)
        self.device = device
        self.id2token = None
        if device!=None:
            self.vae = self.vae.to(device)

    def train(self,train_data,test_data,learning_rate,batch_size,num_epochs,log_every,ckpt=None):
        self.vae.train()
        self.id2token = {v:k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=train_data.collate_fn)

        optimizer = torch.optim.Adam(self.vae.parameters(),lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        if ckpt:
            self.load_model(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst, valloss_lst = [], []
        recloss_lst, klloss_lst = [],[]
        c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
        for epoch in range(start_epoch, num_epochs):
            epochloss_lst = []
            loss_sum = 0.0
            ppx_sum = 0.0
            word_count = 0.0
            doc_count = 0.0
            for iter,data in enumerate(data_loader):
                optimizer.zero_grad()
                word_count_list = []
                txts,bows = data #bows=[batch,2000]
                # print("bows", len(bows), type(bows), bows[0])
                # print("txts", len(txts), txts)
                bows = bows.to(self.device) #[batch_size, |V|]
                p_x,mus,log_vars = self.vae(bows) #p_x = [batch_size,|V|]
                #logsoftmax = torch.log_softmax(p_x,dim=1) #logsoftmax = [batch_size,|V|],GSM这里不用softmax，再做log，因为beta和theta都已经做了 
                logsoftmax = torch.log(p_x + 1e-10) 
                rec_loss = -1.0 * torch.sum(bows*logsoftmax) #bows*logsoftmax = [batch_size, |V|], 其中torch.sum 把所有的loss全部加起来了，也可以只用加某一维度。               
                rec_loss_per = -1.0 * torch.sum(bows*logsoftmax, dim=1)
                rec_loss_per = rec_loss_per.cpu().detach().numpy()

                kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
                loss = rec_loss + kl_div 

                # cal perplexity
                loss_sum += loss.item()
                for txt in txts:
                    word_count_list.append(len(txt))
                    word_count += len(txt)
                word_count_np = np.array(word_count_list)
                doc_count += len(txts)
                ppx_sum += np.sum(np.true_divide(rec_loss_per,word_count_np))
                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))
                if (iter+1) % 10==0:
                    print(f'Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss:{loss.item()/len(bows):<.7f}\tRec Loss:{rec_loss.item()/len(bows):<.7f}\tKL Div:{kl_div.item()/len(bows):<.7f}')
            #scheduler.step()
            if (epoch+1) % log_every==0:
                print("word_count", word_count)
                ppx = np.exp(loss_sum / word_count)
                ppx_document = np.exp(ppx_sum / doc_count)
                print("ppx", ppx)
                print("ppx_document",ppx_document)
                save_name = f'./ckpt/GSM_{train_data}_tp{self.n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_ep{epoch+1}.ckpt'
                checkpoint = {
                    "net": self.vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                    }
                }
                torch.save(checkpoint,save_name)
                # The code lines between this and the next comment lines are duplicated with WLDA.py, consider to simpify them.
                print(f'Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}')
                # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
                # print('='*30)
                smth_pts = smooth_curve(trainloss_lst)
                plt.plot(np.array(range(len(smth_pts)))*log_every,smth_pts)
                plt.xlabel('epochs')
                plt.title('Train Loss')
                plt.savefig('gsm_trainloss.png')

        #         if test_data!=None:
        #             c_v,c_w2v,c_uci,c_npmi,mimno_tc, td = self.evaluate(test_data,calc4each=False)
        #             c_v_lst.append(c_v), c_w2v_lst.append(c_w2v), c_uci_lst.append(c_uci),c_npmi_lst.append(c_npmi), mimno_tc_lst.append(mimno_tc), td_lst.append(td)
        # scrs = {'c_v':c_v_lst,'c_w2v':c_w2v_lst,'c_uci':c_uci_lst,'c_npmi':c_npmi_lst,'mimno_tc':mimno_tc_lst,'td':td_lst}
        # '''
        # for scr_name,scr_lst in scrs.items():
        #     plt.cla()
        #     plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst)
        #     plt.savefig(f'wlda_{scr_name}.png')
        # '''
        # plt.cla()
        # for scr_name,scr_lst in scrs.items():
        #     if scr_name in ['c_v','c_w2v','td']:
        #         plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst,label=scr_name)
        # plt.title('Topic Coherence')
        # plt.xlabel('epochs')
        # plt.legend()
        # plt.savefig(f'gsm_tc_scores.png')
        # The code lines between this and the last comment lines are duplicated with WLDA.py, consider to simpify them.


    def evaluate(self,test_data,calc4each=False):
        topic_words = self.show_topic_words()
        return evaluate_topic_quality(topic_words, test_data, taskname=self.taskname, calc4each=calc4each)


    def inference_by_bow(self,doc_bow):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow,np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.reshape(-1,self.bow_dim).to(self.device)
        with torch.no_grad():
            mu,log_var = self.vae.encode(doc_bow)
            mu = self.vae.fc1(mu)
            theta = F.softmax(mu,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()


    def inference(self, doc_tokenized, dictionary,normalize=True):
        doc_bow = torch.zeros(1,self.bow_dim)
        for token in doc_tokenized:
            try:
                idx = dictionary.token2id[token]
                doc_bow[0][idx] = 1.0
            except:
                print(f'{token} not in the vocabulary.')
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            mu,log_var = self.vae.encode(doc_bow)
            mu = self.vae.fc1(mu)
            if normalize:
                theta = F.softmax(mu,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()

    def get_embed(self,train_data, num=1000):
        self.vae.eval()
        data_loader = DataLoader(train_data, batch_size=512,shuffle=False, num_workers=4, collate_fn=train_data.collate_fn)
        embed_lst = []
        txt_lst = []
        cnt = 0
        for data_batch in data_loader:
            txts, bows = data_batch
            embed = self.inference_by_bow(bows)
            embed_lst.append(embed)
            txt_lst.append(txts)
            cnt += embed.shape[0]
            if cnt>=num:
                break
        embed_lst = np.array(embed_lst,dtype=object)
        txt_lst = np.array(txt_lst,dtype=object)
        embed_lst = np.concatenate(embed_lst,axis=0)[:num]
        txt_lst = np.concatenate(txt_lst,axis=0)[:num]
        return txt_lst, embed_lst

    def get_topic_word_dist(self,normalize=True):
        self.vae.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.vae.decode(idxes)  # word_dist: [n_topic, vocab.size]
            if normalize:
                word_dist = F.softmax(word_dist,dim=1)
            return word_dist.detach().cpu().numpy()

    def show_topic_words(self,topic_id=None,topK=15, dictionary=None):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device) #就是对标准的对角线为1的矩阵，目的是把theta*beta原封不动取出来
        
        word_dist = self.vae.decode(idxes) #[K * V]
       
        #word_dist = torch.softmax(word_dist,dim=1)
        vals,indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if self.id2token==None and dictionary!=None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id==None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
        return topic_words

    def load_model(self, model):
        self.vae.load_state_dict(model)


if __name__ == '__main__':
    model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
