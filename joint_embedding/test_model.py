# %% [markdown]
# # Implement median ranking (MedR) and Recall@k

# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
from utils import *

from networks.network import *
# from datasets.RekihakuDataset import *

from PIL import Image
import pickle
import os
import pandas as pd
import random
import numpy as np

import matplotlib.pyplot as plt
# import japanize_matplotlib

from typing import *

from statistics import median

from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix seed
random.seed(314)
# torch.manual_seed(314)
torch.manual_seed(torch.initial_seed())
np.random.seed(314)

# %%
TEST_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/datasets/test_names.p'
NAME2IMG_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/features/name2img_vec.p'
# NAME2TXT_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/features/name2text_vec_18492.p'

NAME2NAME_OF_ITEM_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/name_of_item_texts/name2vecs.p'
NAME2NAME_OF_COLLECTION_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/name_of_collection_texts/name2vecs.p'
NAME2NOTES_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/all_data/corpus/notes_texts/name2vecs.p'

IMAGES_DIR = "/mnt/LSTA6/data/zhu/Rekihaku/all_data/images"
NAME2ID_DIR = '/mnt/LSTA6/data/zhu/Rekihaku/text_side/name2id.p'

# %%
def load_network(model_path, model: nn.Module) -> nn.Module:
    model.load_state_dict(torch.load(model_path))
    print(model.eval())
    return model

# %%
def make_samples(name_list) -> List:
    samples = []
    for name in name_list:
        sample = {
            "image_vector" : random.choice(name2img[name]),
            "item_name_vector" : random.choice(name2name_of_item[name]['name_of_item']),
            "collection_name_vector" : random.choice(name2name_of_collection[name]['name_of_collection']),
            "notes_vector" : random.choice(name2notes[name]['notes']),
            "name": name
        }
        samples.append(sample)
    assert len(samples) == len(name_list)
    return samples

# %%
def make_recall_dicts(samples: List, save_flag=False):
    recall_image_dicts = []
    recall_text_dicts = []

    for i, sample in enumerate(samples):
        image_vector = sample["image_vector"]
        item_name_vector = sample["item_name_vector"]
        collection_name_vector = sample["collection_name_vector"]
        notes_vector = sample["notes_vector"]
        name = sample["name"]

        image_vector = torch.Tensor(image_vector).type(torch.FloatTensor).unsqueeze(0)
        item_name_vector = torch.Tensor(item_name_vector).type(torch.LongTensor).unsqueeze(0)
        collection_name_vector = torch.Tensor(collection_name_vector).type(torch.LongTensor).unsqueeze(0)
        notes_vector = torch.Tensor(notes_vector).type(torch.LongTensor).unsqueeze(0)

        # maybe wrong
        # if notes_vector.shape == torch.Size([0]):
        if notes_vector.shape[1] == 0:
            notes_vector = torch.zeros_like(item_name_vector).long()
        try:
            text_emb, image_emb = model(item_name_vector, collection_name_vector, notes_vector, image_vector)
        except:
            print("notes_vector.shape = {}".format(notes_vector.shape))
        
        text_emb = text_emb.detach().numpy().reshape(1, -1)
        image_emb = image_emb.detach().numpy().reshape(1, -1)

        text_dict = { "idx": i, "emb": text_emb, "name": name }

        img_dict = { "idx": i, "emb": image_emb, "name": name }

        recall_image_dicts.append(img_dict)
        recall_text_dicts.append(text_dict)
    
    if save_flag:
        save_pkl(recall_image_dicts, "recall_image_dicts.pkl")
        save_pkl(recall_text_dicts, "recall_text_dicts.pkl")
    
    return recall_image_dicts, recall_text_dicts
    

# %% [markdown]
# ## Utils

# %%
def calculate_recall(sources: dict, targets: dict, k: int, debug=False) -> int:
    cnt = 0
    for i, source in enumerate(sources):
        source_name = source['name']
        source_emb = source['emb']
        # source_idx = source['idx']
        result_dicts = []
        for target in targets:
            similarity = cosine_similarity(source_emb, target["emb"])
            result_dict = {
                # 'idx': target['idx'],
                'cosine_similarity': similarity,
                'name': target['name'],
            }
            result_dicts.append(result_dict)
        sorted_dicts = sorted(result_dicts, key=lambda x : x["cosine_similarity"], reverse=True)[:k]
        for n_dict in sorted_dicts:
            if n_dict['name'] == source_name:
                cnt += 1
            # if n_dict['idx'] == source_idx:
            #     cnt += 1
        if debug:
            if i % 100 == 0:
                print("calculate_recall {}/{} done".format(i, len(sources)))
    return cnt



# %%
def rank(k: int, recall_image_dicts: dict, recall_text_dicts: dict) -> Tuple[int, int]:
    assert len(recall_image_dicts) == len(recall_text_dicts)
    size = len(recall_image_dicts)

    i2t_count = calculate_recall(recall_image_dicts, recall_text_dicts, k)
    t2i_count = calculate_recall(recall_text_dicts, recall_image_dicts, k)
    
    return i2t_count/size, t2i_count/size

# %%
def medR(sources: dict, targets: dict) -> float:
    medR = []
    for i, source in enumerate(sources):
        source_name = source['name']
        source_emb = source['emb']
        # source_idx = source['idx']
        result_dicts = []
        for target in targets:
            similarity = cosine_similarity(source_emb, target['emb'])
            result_dict = {
                'name' : target['name'],
                'cosine_similarity' : similarity, 
                # 'idx': target['idx'],
                # 'source' : targets[i]["info"] 
            } 
            result_dicts.append(result_dict)
        
        # cosine_similarityの値でソートする
        sorted_dicts = sorted(result_dicts, key=lambda x : x["cosine_similarity"], reverse=True)

        for r_i, sorted_dict in enumerate(sorted_dicts):
            if sorted_dict["name"] == source_name:
                medR.append(r_i + 1)
            # if sorted_dict["idx"] == source_idx:
            #     medR.append(r_i + 1)

    return median(medR)

# %%
def MAP(sources: dict, targets: dict) -> float:
    '''
    Mean average precision
    '''
    AP = []
    for source in sources:
        source_name = source['name']
        source_emb = source['emb']
        # source_idx = source['idx']
        result_dicts = []
        for target in targets:
            similarity = cosine_similarity(source_emb, target['emb'])
            result_dict = {
                'name' : target['name'],
                'cosine_similarity' : similarity, 
                # 'idx': target['idx'],
                # 'source' : targets[i]["info"] 
            } 
            result_dicts.append(result_dict)
        
        # cosine_similarityの値でソートする
        sorted_dicts = sorted(result_dicts, key=lambda x : x["cosine_similarity"], reverse=True)

        cnt = 0

        for r_i, sorted_dict in enumerate(sorted_dicts):
            if sorted_dict["name"] == source_name:
                cnt += 1
                AP.append(cnt / (r_i + 1))
            # if sorted_dict["idx"] == source_idx:
            #     AP.append(r_i + 1)
    # print(AP)
    return sum(AP) / len(AP)

    

# %%
def evaluate(recall_image_dicts, recall_text_dicts):
    print("Start computing MedR and Recall@k...")
    i2t_r1, t2i_r1 = rank(1, recall_image_dicts, recall_text_dicts)
    i2t_r5, t2i_r5 = rank(5, recall_image_dicts, recall_text_dicts)
    i2t_r10, t2i_r10 = rank(10, recall_image_dicts, recall_text_dicts)

    i2t_mAP, t2i_mAP = MAP(recall_image_dicts, recall_text_dicts), MAP(recall_text_dicts, recall_image_dicts)

    medR_i2t = medR(recall_image_dicts, recall_text_dicts)
    medR_t2i = medR(recall_text_dicts, recall_image_dicts)

    print("=== Image => Text ===")
    print("recall @1 : {:5f}".format(i2t_r1))
    print("recall @5 : {:5f}".format(i2t_r5))
    print("recall @10 : {:5f}".format(i2t_r10))
    print("medR : {:2f}".format(medR_i2t))
    print("mAP : {:2f}".format(i2t_mAP))

    print("")
    print("=== Text => Image ===")
    print("recall @1 : {:5f}".format(t2i_r1))
    print("recall @5 : {:5f}".format(t2i_r5))
    print("recall @10 : {:5f}".format(t2i_r10))
    print("medR : {:2f}".format(medR_t2i))
    print("mAP : {:2f}".format(t2i_mAP))

# %%
if __name__ == "__main__":
    # load model
    model_path = '/mnt/LSTA6/data/zhu/Rekihaku/joint_embedding/model_1207/model_16_0.022408.t7'
    model = ImageSentenceEmbeddingNetwork(2048, 2048)
    model = load_network(model_path, model)

    # load data
    test_names_list = load_pkl(TEST_DIR)
    name2img = load_pkl(NAME2IMG_DIR)
    name2name_of_item = load_pkl(NAME2NAME_OF_ITEM_DIR)
    name2name_of_collection = load_pkl(NAME2NAME_OF_COLLECTION_DIR)
    name2notes = load_pkl(NAME2NOTES_DIR)

    # sample 1k test data
    random.shuffle(test_names_list)
    test_1k_names_list = test_names_list[:1000]
    samples = make_samples(test_1k_names_list)
    print("Sample count: {}".format(len(samples)))

    recall_image_dicts, recall_text_dicts = make_recall_dicts(samples)

    # evaluate
    evaluate(recall_image_dicts, recall_text_dicts)

# %%



