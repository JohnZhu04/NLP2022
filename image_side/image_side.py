# %%
# !pip3 install torch torchvision torchaudio

# %%
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch
import os
import pickle
import random
import IPython

%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %%
IMAGES_FILES = "/mnt/LSTA5/data/zhu/Rekihaku/all_data/images"

files = os.listdir(IMAGES_FILES)

# %%
from scipy import spatial

def cos_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)

# %%
# load map
with open('file_to_vector.p', 'rb') as f:
    mp = pickle.load(f)

# %%
assert len(mp) == len(files) # 79166

# %%
# def show_image(filename):
#     filename = os.path.join(IMAGES_FILES, filename)
#     print(filename)
#     im = Image.open(filename)
#     display(im)

# %%
def query(topn=8):
    assert topn > 0
    
    res = []
    file_1 = random.choice(files)
#     print("file_1: {}".format(file_1))
    feat_1 = mp[file_1]
    for file_2 in files:
        if file_2 == file_1:
            continue
        feat_2 = mp[file_2]
        val = (cos_similarity(feat_1, feat_2), file_2)
        res.append(val)
    res.sort(reverse=True)

    images = [x[1] for x in res[:topn]]
    images = [file_1] + images
    show_images(images)
    return res[:topn]
    

# %%
def show_images(images, col=3):
    plt.figure(figsize=(40,40))
    
    save_name = images[0].split('.')[0]

    for i, image in enumerate(images):
        filename = image
        image = Image.open(os.path.join(IMAGES_FILES, image))
        ax = plt.subplot(len(images) / col + 1, col, i + 1)
        if i == 0:
            ax.set_title("query: {}".format(filename), fontsize=30)
        else:
            ax.set_title("top {} similar: {}".format(i, filename), fontsize=30)
        plt.imshow(image)

#     plt.savefig('similar_results/{}.png'.format(save_name), bbox_inches='tight')

# %%
for _ in range(8):
    query()
    print("Done")


res = []
file_1 = random.choice(files)
print("file_1: {}".format(file_1))
feat_1 = mp[file_1]
for file_2 in files:
    if file_2 == file_1:
        continue
    feat_2 = mp[file_2]
    val = (cos_similarity(feat_1, feat_2), file_2)
    res.append(val)
res.sort(reverse=True)