# %%
import pandas as pd
import numpy as np
from collections import defaultdict

from gensim.models import Word2Vec

import pickle

df = pd.read_json("combined_results_18492.json")
df = df[["資料名称", "コレクション名", "資料番号", "備考"]]

df_name_of_item = df[["資料名称"]]
df_name_of_collection = df[["コレクション名"]]
df_notes = df[["備考"]]

df_name_of_item.to_csv('tokenize/name_of_item_texts.raw', sep=';', index=False, header=False)
df_name_of_collection.to_csv('tokenize/name_of_collection_texts.raw', sep=';', index=False, header=False)
df_notes.to_csv('tokenize/notes_texts.raw', sep=';', index=False, header=False)

# !kytea < tokenize/name_of_item_texts.raw > tokenize/name_of_item_texts.full -notags
# !kytea < tokenize/name_of_collection_texts.raw > tokenize/name_of_collection_texts.full -notags
# !kytea < tokenize/notes_texts.raw > tokenize/notes_texts.full -notags


