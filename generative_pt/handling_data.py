from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import shutil
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils import clip_grad_norm_

import re
import os
from io import open
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


# USE_CUDA = torch.cuda.is_available()
# device = 0 if USE_CUDA else -1
device = -1

datafile = ['data/demo.train','data/demo.dev']
embed_file = ' '
# prepared_vocab_file = 'data/demo_30000.vocab.pt'
# prepared_data_file = 'data/demo_30000.data.pt'

MAX_VOCAB_SIZE = 30000
MAX_LEN = 500
MIN_FREQUENCY = 0
BATCH_SIZE = 2
LR = 0.0001
EPOCHES = 20
MAX_LENGTH = 10  # Maximum sentence length

PAD_token = 0  # Used for padding short sentences
UNK_token = 1
BOS_token = 2  # Start-of-sentence token
EOS_token = 3  # End-of-sentence token

# ~~~~~~~~~~~~~~~~~~~~Data Handling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Dataset(torch.utils.data.Dataset):
    """
    自己定义你的数据类继承这个抽象类时，只需要定义__len__和__getitem__这两个方法就可以
    """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def read_data(data_file, max_len, data_type="train"):
    """
    return:
        train_raw_iter:  [ [{'src':str},{'tgt':str},{'cue':[str,str,...]}],
                            ...         89901个                                    ]
        valid_raw_iter:  [ [{'src':str},{'tgt':str},{'cue':[str,str,...]}],
                            ...         9054个                                    ]
    """
    data_iter_key = ['train_iter','valid_iter']
    data_iter_value = []
    for iter in data_file:
        data = []
        with open(iter, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt, knowledge = line.strip().split('\t')[:3]
                filter_knowledge = []
                for sent in knowledge.split(''):
                    filter_knowledge.append(' '.join(sent.split()[:max_len]))
                data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge})
        data_iter_value.append(data)
    data_raw_iter = dict(zip(data_iter_key,data_iter_value))
    train_raw_iter = data_raw_iter['train_iter']
    valid_raw_iter = data_raw_iter['valid_iter']
    return train_raw_iter, valid_raw_iter

def tokenize(s):
    s = re.sub('\d+', '<num>', s).lower().split()
    return s

def max_lens(X):
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")

def list2tensor(X):
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: '<pad>', UNK_token: '<unk>', BOS_token: '<bos>', EOS_token: '<eos>'}
        self.num_words = 4  # Count BOS, EOS, PAD ,UNK       # Count default tokens

    def addSentence(self, data_iter):
        counter = Counter()
        for datas in data_iter:
            for name, sentence in datas.items():
                if isinstance(sentence, str):
                    sentence = tokenize(sentence)
                    counter.update(sentence)
                    for word in counter.keys():
                        self.addWord(word)
                elif isinstance(sentence, list):
                    for sentence_list in sentence:
                        sentence_list = tokenize(sentence_list)
                        counter.update(sentence_list)
                        for word in counter.keys():
                            self.addWord(word)
        # sort by frequency, then alphabetically
        self.word2count = sorted(counter.items(), key=lambda tup: tup[0])
        self.word2count.sort(key=lambda tup: tup[1], reverse=True)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words  # 不含 special words
            self.index2word[self.num_words] = word
            self.num_words += 1  # 词汇表里每添加一个单词，下标就 +1

    # Remove words below a certain count threshold          trim:修剪
    def trim(self, min_count,max_vocab_size):
        keep_words = []
        for k, v in self.word2count:
            if v >= min_count and len(self.word2count) <= max_vocab_size:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: '<pad>', UNK_token: '<unk>', BOS_token: '<bos>', EOS_token: '<eos>'}
        self.num_words = 4  # Count default tokens
        for word in keep_words:
            self.addWord(word)

def build_vocab(train_raw_iter,min_freq, max_vocab_size,embed_file=None):
    if embed_file:
        embeds = [e_file for e_name, e_file in embed_file.items()]
    else:
        embeds = None
    voc = Voc('train_iter')
    voc.addSentence(train_raw_iter)
    voc.trim(min_freq,max_vocab_size)
    vocabulary = {'vocab':voc.word2index,
                  'embeddings':embeds}
    # print("Saving prepared vocab ...")
    # torch.save(vocabulary, prepared_vocab_file)
    # print("Saved prepared vocab to '{}'".format(prepared_vocab_file))
    print('Finished building vocabulary!!!')
    return vocabulary

def indexesFromDatas(vocabulary, raw_iter, datatype):
    def indexesFromSentence(sentence):
        data_list = [BOS_token]
        for word in sentence:
            if word in vocabulary.keys():
                data_list.append(vocabulary[word])
            else:
                data_list.append(1)
        data_list.append(EOS_token)
        return data_list
    data_iter = []
    vocabulary = vocabulary['vocab']
    for data in raw_iter:
        data_dict = {}
        for name, sentence in data.items():
            if isinstance(sentence,str):
                sentence = tokenize(sentence)
                data_list = indexesFromSentence(sentence)
                data_dict[name] = data_list
            elif isinstance(sentence,list):
                data_lists = []
                for string in sentence:
                    string = tokenize(string)
                    data_list = indexesFromSentence(string)
                    data_lists.append(data_list)
                data_dict[name] = data_lists
        data_iter.append(data_dict)
    print('Finished building {} sentence token!!!'.format(datatype))
    return data_iter

# ~~~~~~~~~~~~~~~~Prepare Data for Seq2SeqModels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_batches(batch_size,data_iter,datatype):
    def collate_fn(device=-1):
        def collate(data_list):
            batch = {}
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch

        return collate

    data_loader = Data.DataLoader(dataset=data_iter,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=collate_fn(device), #merges a list of samples to form a mini-batch
                                          pin_memory=False)
    print('Finished preparing the corpus of {} !!!'.format(datatype))
    return data_loader

# ~~~~~~~~~~~~~~~~~~~~~~~build model : Seq2Seq~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ generate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ evaluate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ train（iterate） ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



if __name__ == '__main__':
    train_raw_iter, valid_raw_iter = read_data(datafile, MAX_LEN)
    #~~~~~~~~~~~~~~~~~~1.data handling ~~~~~~~~~~~~~~~~~~~~
    vocabulary = build_vocab(train_raw_iter,MIN_FREQUENCY,MAX_VOCAB_SIZE)
    train_token_iter = indexesFromDatas(vocabulary,train_raw_iter, 'train')
    valid_token_iter = indexesFromDatas(vocabulary,valid_raw_iter, 'valid')
    #~~~~~~~~~~~~~2.Prepare Data for Seq2SeqModels~~~~~~~~~~~~~~~~
    train_iter = Dataset(train_token_iter)
    valid_iter = Dataset(valid_token_iter)
    train_iter = create_batches(BATCH_SIZE, train_iter, 'train')  # without embedding
    valid_iter = create_batches(BATCH_SIZE, valid_iter, 'valid')
    # ~~~~~~~~~~~~~~~~~~~ 3.Seq2SeqModels~~~~~~~~~~~~~~~~~~~~
    model = Seq2Seq()
    encoder = RNNEncoder()
    decoder = RNNDecoder()
    atten = Attn()
    # ~~~~~~~~~~~~~~~~~~~ 4.Generation~~~~~~~~~~~~~~~~~~~~

