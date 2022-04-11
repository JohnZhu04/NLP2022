# %%
import pickle
import os
import pandas as pd
from collections import defaultdict
import json

# %%
# 'A-12-25_0.jpg' -> image vector
with open('file_to_vector.p', 'rb') as f:
    filename_to_feat = pickle.load(f)

assert len(filename_to_feat) == 79166  # count of images

# %%
IMAGES_FILES = "/mnt/LSTA5/data/zhu/Rekihaku/all_data/images"

# files = os.listdir(IMAGES_FILES)
files = sorted(os.listdir(IMAGES_FILES))

assert len(files) == 79166  # count of images


# %%
# remove 3 texts

# df = pd.read_json("../text_side/combined_results_18495.json")
# df_with_idx = df.set_index("資料番号")
# df_with_idx.drop(['F-540', 'H-132-1', 'H-132-2'], inplace=True)
# df_18492 = df_with_idx.reset_index()
# df_18492.head()

# with open("../text_side/combined_results_18492.json", 'w') as f:
#     f.write(df_18492.to_json())

# %%
df = pd.read_json("../text_side/combined_results_18492.json")

id2name = {
    row["資料番号"]: row["資料名称"] for _, row in df.iterrows()
}

assert len(id2name) == 18492

# %%
# print(id2name['A-12-25'])

# %%
name2img_vec = defaultdict(list)

for name, feat in filename_to_feat.items():
    file_id = name.split('_')[0]
    # print(file_id)
    # print(id2name[file_id])
    feat = feat.reshape((2048, ))
    name2img_vec[id2name[file_id]].append(feat)
    # print(filename, feat.shape)

print(len(name2img_vec))

# %%
# y = os.listdir('/mnt/LSTA5/data/zhu/Rekihaku/all_data/texts')
# y = sorted([name.split('.')[0] for name in y])
# y = set(y)

# y-x
# {'F-540', 'H-132-1', 'H-132-2'}: no pictures!

# %%
with open('name2img_vec.p', 'wb') as f:
    pickle.dump(name2img_vec, f)

# %%
# test
arr = name2img_vec['疱瘡紅摺絵松洛画（おきあがりかるくはづむやはねのおと）'][0]
arr.shape  # (2048,)

# %% [markdown]
# 


