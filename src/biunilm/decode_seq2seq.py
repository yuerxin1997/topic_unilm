"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import sys
sys.path.append("/home/yuerxin/topic_unilm/src/")
from topic_model.gsm import GSM
from topic_model.utils import evaluate_topic_quality, smooth_curve,calc_topic_diversity
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader as seq2seq_loader
import jieba
import pickle
import gensim
from gensim.corpora import Dictionary

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
# globa var
loss_sum = 0.0
ppx_sum = 0.0
word_count = 0.0
doc_count = 0.0

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

def show_topic_words(vae,n_topic, device, id2token, topic_id=None,topK=5):
    topic_words = []
    idxes = torch.eye(n_topic).to(device) #就是对标准的对角线为1的矩阵，目的是把beta原封不动取出来
    
    word_dist = vae.decode(idxes) #[K * V]
    
    #word_dist = torch.softmax(word_dist,dim=1)
    vals,indices = torch.topk(word_dist,topK,dim=1)
    vals = vals.cpu().tolist()
    indices = indices.cpu().tolist()
    if topic_id==None:
        for i in range(n_topic):
            topic_words.append([id2token[idx] for idx in indices[i]])
    else:
        topic_words.append([id2token[idx] for idx in indices[topic_id]])
    return topic_words

def cal_ppl(batch_bow, p_x, log_vars, mus):
    global loss_sum 
    global ppx_sum 
    global word_count 
    global doc_count 
    # topic_model evaluate
    logsoftmax = torch.log(p_x + 1e-10) 
    rec_loss = -1.0 * torch.sum(batch_bow*logsoftmax) #bows*logsoftmax = [batch_size, |V|], 其中torch.sum 把所有的loss全部加起来了，也可以只用加某一维度。               
    rec_loss_per = -1.0 * torch.sum(batch_bow*logsoftmax, dim=1)
    rec_loss_per = rec_loss_per.cpu().detach().numpy()
    kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
    loss = rec_loss + kl_div 
    # cal perplexity
    word_count_list = []
    loss_sum += loss.item()
    for bow in batch_bow:
        word_num = torch.sum(bow).cpu().numpy()
        word_count_list.append(word_num)
        word_count += word_num
    word_count_np = np.array(word_count_list)
    doc_count += len(batch_bow)
    ppx_sum += np.sum(np.true_divide(rec_loss_per,word_count_np))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--unilm_model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining unilm model.")
    parser.add_argument("--topic_model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining topic model.")
    parser.add_argument("--topic_data_path", default=None, type=str,
                        help="The file of  topic model data.")
    parser.add_argument("--topic_num", default=50, type=int,
                        help="topic_num.")
    parser.add_argument("--data_path", default=None, type=str,
                        help="The file of  topic model data.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--topic_mode', default=1, type=float,
                        help="1:idea1 1.1:idea1_wo_theta 2:idea2 ")
    parser.add_argument("--topic_model_dict_path", default=None, type=str,
                        help="The file of fine-tuned pretraining topic model.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()
    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,
                                                                    mode="s2s", num_qkv=args.num_qkv, s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment, s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift))
    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        # logger.info("enable fp16 with amp")
    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])
    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r
    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)
    not_predict_set = _get_token_id_set(args.not_predict_token)
    unilm_model_recover = torch.load(args.unilm_model_recover_path)
    unilm = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=unilm_model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                    length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set, ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode, max_position_embeddings=args.max_seq_length, ffn_type=args.ffn_type, num_qkv=args.num_qkv, seg_emb=args.seg_emb, pos_shift=args.pos_shift)
    topic_model_recover = torch.load(args.topic_model_recover_path)
    dictionary = Dictionary.load_from_text(args.topic_model_dict_path)
    gsm = GSM(len(dictionary))
    gsm.load_state_dict(topic_model_recover)
    del unilm_model_recover
    del topic_model_recover

    if args.fp16:
        unilm.half()
        gsm.half()
    unilm.to(device)
    gsm.to(device)

    if n_gpu > 1:
        unilm = torch.nn.DataParallel(unilm)
        gsm = torch.nn.DataParallel(gsm)
    torch.cuda.empty_cache()
    unilm.eval()
    gsm.eval()
    next_i = 0
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    with open(args.input_file, encoding="utf-8") as fin:
        input_lines = [x.strip() for x in fin.readlines()]
        if args.subset > 0: #==0 可忽略
            # logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    input_lines = [data_tokenizer.tokenize(
        x)[:max_src_length] for x in input_lines]

    input_lines = sorted(list(enumerate(input_lines)),
                            key=lambda x: -len(x[1])) #input_lines = [(ori_index,[tokens]), (ori_index,[tokens])] 按照文本长度倒着排

    output_lines = [""] * len(input_lines) #一维[]
    score_trace_list = [None] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size) 
    # get topic_model bows
    def detokenize(tk_list):
        r_list = []
        src = " ".join(tk_list)
        src = src.replace("[UNK]","")
        tk_list = src.split()
        for tk in tk_list:
            if tk.startswith('##') and len(r_list) > 0:
                r_list[-1] = r_list[-1] + tk[2:]
            else:
                r_list.append(tk)
        src = " ".join(r_list)
        src = src.replace("UNK","")
        r_list = src.split()
        return r_list

    txtLines = []
    for input_line in input_lines:
        textline = " ".join(detokenize(input_line[1]))
        txtLines.append(textline) 
    cwd = os.getcwd()
    dictionary = Dictionary.load_from_text(args.topic_model_dict_path)
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty be default, it is a bug.
    stopwords = set([l.strip('\n').strip() for l in open(os.path.join(cwd,'data/topic_model','stopwords.txt'),'r',encoding='utf-8')])
    topic_tokenizer = seq2seq_loader.SpacyTokenizer(stopwords=stopwords)
    docs = topic_tokenizer.tokenize(txtLines)

    # convert to BOW representation
    bows, _docs = [],[]
    vocabsize = len(dictionary)
    print("vocabsize", vocabsize)
    for doc in docs:
        _bow = dictionary.doc2bow(doc)
        if _bow!=[]:
            _docs.append(list(doc))
            bows.append(_bow)
        else:
            bows.append([(vocabsize-1,1)])
    docs = _docs
    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size] #如果超过就到最后一个，这是list[a:b]的特性
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]              
            max_a_len = max([len(x) for x in buf])
            instances = []
            batch_bow = []
            for i in range(next_i, next_i + args.batch_size):
                if i < len(input_lines):
                    bow = torch.zeros(vocabsize)
                    item = list(zip(*bows[i])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
                    bow[list(item[0])] = torch.tensor(list(item[1])).float()  
                    batch_bow.append(bow) 
            next_i += args.batch_size
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline: #proc 是 Preprocess4Seq2seqDecoder  相当于可以把数据给padding
                    instances.append(proc(instance)) 
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                batch_bow = torch.stack(batch_bow)
                batch_bow = batch_bow.to(device)
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                p_x,mus,log_vars,theta,beta,topic_embedding = gsm(batch_bow)  
                traces = unilm(input_ids, theta,beta, topic_embedding,args.topic_mode,token_type_ids,
                                position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                cal_ppl(batch_bow, p_x, log_vars, mus)

                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in ("[SEP]", "[PAD]"):
                            break
                        output_tokens.append(t)
                    output_sequence = ' '.join(detokenize(output_tokens))
                    output_lines[buf_id[i]] = output_sequence
                    if args.need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}    
            pbar.update(1)
    print("word_count", word_count)
    ppx = np.exp(loss_sum / word_count)
    ppx_document = np.exp(ppx_sum / doc_count)
    print("ppx", ppx)
    print("ppx_document",ppx_document)
    topic_words = show_topic_words(gsm.module, args.topic_num, device, dictionary.id2token, topic_id=None,topK=10)
    # evaluate_topic_quality(topic_words, docs, dictionary, taskname="unilm", calc4each=False)
    topic_diversity = calc_topic_diversity(topic_words)
    print("topic_diversity", topic_diversity)
    # print('\n'.join([str(lst) for lst in topic_words]))
    # print('='*30)
    
    if args.output_file:
        fn_out = args.output_file
    else:
        fn_out = args.unilm_model_recover_path+'.'+args.split
    with open(fn_out, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")

    if args.need_score_traces:
        with open(fn_out + ".trace.pickle", "wb") as fout_trace:
            pickle.dump(
                {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
            for x in score_trace_list:
                pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()
