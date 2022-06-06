import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from collections import Counter
import numpy as np
import random
import math
import json

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import time

from timing_dataset import *

C = 2 # context window
K = 2 # number of negative samples
epochs = 200
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 30
batch_size = 2 ** 19
lr = 0.2
CUDA = True
gpu = 'cuda:0'

with open('./dataset/sns_embedding_quant.txt') as f:
    text = f.read() 

text = text.lower().split() 
# text = text[:DATASET_LEN]
print("Num Words:", len(text))
vocab_dict = dict(Counter(text)) 
idx2word = ff2ff_timing_dataset.vocab['i2w']
word2idx = ff2ff_timing_dataset.vocab['w2i']
word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)

vocab_size = len(idx2word)
print("Vocab Size: ", vocab_size)


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            idx2word: index to word mapping
            word_freqs: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__() # #通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx[word] for word in text] # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded) # nn.Embedding需要传入LongTensor类型
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
        
        self.dataset_ret = []
        for i in range(len(self)):
            self.dataset_ret.append(self.getitem(i))
            if i%100000==0:
                print( (i+0.0) / (len(self)+0.0)  *100.0)
        
        # self.dataset_ret = pool.map(self.getitem, range(len(self)))

    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数
    
    def getitem(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx] # 取得中心词
        text_len = len(self.text_encoded)
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        # pos_indices = pos_indices % text_len
        # pos_indices = pos_indices.long()
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        return center_words, pos_words, neg_words
    
    def __getitem__(self, index):
        return self.dataset_ret[index]

dataset = WordEmbeddingDataset(text, word2idx, idx2word, word_freqs, word_counts)

dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.in_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.in_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1) # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_pos + log_neg
        
        return -loss
    
    def input_embedding(self):
        emb = self.in_embed.weight.cpu().detach().numpy()
        emb[0] = th.zeros(EMBEDDING_SIZE)
        return emb


model = EmbeddingModel(vocab_size, EMBEDDING_SIZE).to(gpu)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())
items = dataloader.__len__()
print("#Batches: ", items)

start_time = time.time()
for e in range(epochs):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        # print(input_labels, pos_labels, neg_labels)
        #exit()
        input_labels = input_labels.to(gpu)
        pos_labels = pos_labels.to(gpu)
        neg_labels = neg_labels.to(gpu)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()

        # print("epoch: {} @ {}%, loss={}".format(e, (i+0.0)/items*100, loss))
        if i % (items // 4) == 0:
            time_elapsed = time.time() - start_time
            print("epoch: {} @ {:.2f}%, loss={:.2f}, time_elapsed={:.2f}s".format(e, (i+0.0)/items*100, loss, time_elapsed))

embedding_weights = model.input_embedding()
torch.save(embedding_weights, "sns_quant_embedding-{}.pt".format(EMBEDDING_SIZE))
embed_dict = {"w2i": word2idx, "i2w": idx2word}
embed_dict_str = json.dumps(embed_dict)
embed_dict_file = open("sns_embed_dict.json", "w")
embed_dict_file.write(embed_dict_str)
embed_dict_file.close()
