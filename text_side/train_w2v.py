# %%
# !pip install --upgrade gensim

# %%
import pandas as pd
import json
import os
import random

from gensim.models import Word2Vec

# %%
def full_to_sentences(full_path):
    df = pd.read_csv(full_path, sep=';', header=None)
    assert len(df) == 18492
    sentences = [row[0].replace('\\', ' ').replace(',', ' ').split() for _, row in df.iterrows()]
    return sentences

# %%
sentences = full_to_sentences('all_texts_18492.full')

# %%
# train
model = Word2Vec(sentences, sg=1, vector_size=300, window=10, min_count=1)
model.save("word2vec_18492_300D.model")


# %%
model = Word2Vec.load("/mnt/LSTA6/data/zhu/Rekihaku/text_side/w2v_model/word2vec_18492_300D.model")

# %%
def similar_words(word, model: Word2Vec, topn=10):
    print("Query: [{}]".format(word))
    print("Most similar {} words of [{}]:".format(topn, word))
    similar_words = model.wv.most_similar(word, topn=10)
    for idx, w in enumerate(similar_words):
        print("{} {}".format(idx, w))
    print('-'*100)


word = random.choice(model.wv.index_to_key)
similar_words(word, model)

# %%



