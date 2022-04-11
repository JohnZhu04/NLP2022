# %%
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple
from urllib.parse import urljoin

import torch
import torch.nn.functional as F

import re
import os
import pickle

# %% [markdown]
# ## Create corpus, vocabulary, word2idx and idx2word

# %%
def clean_input(sentence: list) -> list:
    # sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', " ", sentence)
    sentence = sentence.replace('-', ' ').replace('\\', ' ').replace(',', ' ').split()
    return sentence

# %%
def make_corpus(full_path: str) -> list: 
    corpus = []
    df_full = pd.read_csv(full_path, sep=';', header=None)
    for _, row in df_full.iterrows():
        corpus.append(clean_input(row[0]))
    return corpus

# %%
def make_vocabulary(corpus: list) -> Tuple:
    '''
    make vocabulary
    '''
    MIN_CNT = 0  # trim words
    cnter = Counter()
    for sentence in corpus:
        for token in sentence:
            cnter[token] += 1

    trimmed_vocabulary = set()

    for sentence in corpus:
        for token in sentence:
            if cnter[token] > MIN_CNT:
                trimmed_vocabulary.add(token)

    trimmed_vocabulary = sorted(trimmed_vocabulary)

    word2idx = {w: idx+1 for (idx, w) in enumerate(trimmed_vocabulary)}
    word2idx.update({"<PAD>": 0})

    idx2word = {idx+1: w for (idx, w) in enumerate(trimmed_vocabulary)}
    idx2word.update({0: "<PAD>"})

    print("trimmed_vocabulary size: {}".format(len(trimmed_vocabulary)))
    print("word2idx size: {}".format(len(word2idx)))
    print("idx2word size: {}".format(len(idx2word)))


    return trimmed_vocabulary, word2idx, idx2word

# %%
def save_to_pickle(path, trimmed_vocabulary, word2idx, idx2word, corpus):
    name = path.split('.')[0].split('/')[1] + '/'
    save_url = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/'
    save_url = urljoin(save_url, name)

    print("Saving to {}".format(save_url))
    with open(urljoin(save_url, 'voc.p'), 'wb') as f:
        pickle.dump(trimmed_vocabulary, f)
        print("Saved: {}".format(urljoin(save_url, 'voc.p')))
    with open(urljoin(save_url, 'word2idx.p'), 'wb') as f:
        pickle.dump(word2idx, f)
        print("Saved: {}".format(urljoin(save_url, 'word2idx.p')))
    with open(urljoin(save_url, 'idx2word.p'), 'wb') as f:
        pickle.dump(idx2word, f)
        print("Saved: {}".format(urljoin(save_url, 'idx2word.p')))
    with open(urljoin(save_url, 'corpus.p'), 'wb') as f:
        pickle.dump(corpus, f)
        print("Saved: {}".format(urljoin(save_url, 'corpus.p')))
    print()

# %%
def execute():
    for full_path in ['tokenize/name_of_item_texts.full', 'tokenize/name_of_collection_texts.full', 'tokenize/notes_texts.full']:
        corpus = make_corpus(full_path)
        print("path: {}".format(full_path))
        trimmed_vocabulary, word2idx, idx2word = make_vocabulary(corpus)
        save_to_pickle(full_path, trimmed_vocabulary, word2idx, idx2word, corpus)

execute()

# %%
with open('name_of_item_list_18492.p', 'rb') as f:
    name_of_item_list = pickle.load(f)

assert len(name_of_item_list) == 18492

# %%
def make_mapping(path):
    # load pickles
    voc_url = urljoin(path, 'voc.p')
    idx2word_url = urljoin(path, 'idx2word.p')
    word2idx_url = urljoin(path, 'word2idx.p')
    corpus_url = urljoin(path, 'corpus.p')
    with open(voc_url, 'rb') as f:
        voc = pickle.load(f)
    with open(idx2word_url, 'rb') as f:
        idx2word = pickle.load(f)
    with open(word2idx_url, 'rb') as f:
        word2idx = pickle.load(f)
    with open(corpus_url, 'rb') as f:
        corpus = pickle.load(f)
    
    max_len = 0
    idx_vectors = []

    def sentence2vector(sentence: list):
        # return torch.tensor([word2idx[w] for w in sentence if w in word2idx], dtype=torch.long)
        return [word2idx[w] for w in sentence if w in word2idx]
        # return np.array([word2idx[w] for w in sentence if w in word2idx])

    for sentence in corpus:
        idx_vector = sentence2vector(sentence)
        idx_vectors.append(idx_vector)
        max_len = max(max_len, len(idx_vector))
    
    assert len(idx_vectors) == 18492
    assert len(corpus) == 18492

    name2vecs = defaultdict(lambda: defaultdict(list))
    key = path.split("corpus/")[1].replace('_texts/', '')

    for name, sentence in zip(name_of_item_list, corpus):
        # print("name: {}, sentence: {}".format(name, sentence))
        sentence_vec = sentence2vector(sentence)
        # sentence_vec = sentence_vec + [0] * (max_len - len(sentence_vec))  # zero padding
        sentence_vec = np.array(sentence_vec)
        # assert len(sentence_vec) == max_len
        # while len(sentence_vec) < max_len:
            # sentence_vec.append(0)  
        # sentence_vec = torch.tensor(sentence_vec, dtype=torch.long)
        name2vecs[name][key].append(sentence_vec)

    print("corpus: {}, max_len = {}".format(path, max_len))
    return dict(name2vecs), max_len

# %%
# corpus_url = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/notes_texts/'                  # max_len = 410
# corpus_url = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/name_of_item_texts/'           # max_len = 44
corpus_url = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/name_of_collection_texts/'       # max_len = 35

name2vecs, max_len = make_mapping(corpus_url)

# %%
save_url = urljoin(corpus_url, 'name2vecs.p')
print(save_url)
with open(save_url, 'wb') as f:
    pickle.dump(name2vecs, f)

# %%
# len(name2vecs)
name2vecs['疱瘡紅摺絵松洛画（おきあがりかるくはづむやはねのおと）']

# %%
save_url = urljoin(corpus_url, 'voc.p')
with open(save_url, 'rb') as f:
    voc = pickle.load(f)


