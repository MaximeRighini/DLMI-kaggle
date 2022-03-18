import numpy as np
import pandas as pd
import os
import cv2
import skimage.io
from tqdm import tqdm

df_train = pd.read_csv("train.csv").reset_index(drop=True)
df_test = pd.read_csv("test.csv").reset_index(drop=True)

def get_tiles(img, tile_size_in, n_tiles, tile_size_out):
    # find by how much we must pad the image so that the image dims are multiples of tile_size_in
    h, w, c = img.shape
    pad_h = (tile_size_in - h % tile_size_in) % tile_size_in
    pad_w = (tile_size_in - w % tile_size_in) % tile_size_in
    # pad the image with constant values (255)
    img = np.pad(img, [[pad_h//2, pad_h - pad_h//2], [pad_w//2, pad_w - pad_w//2], [0, 0]],
                 constant_values=255)
    # reshape the image and swap the order of the dims
    img = img.reshape(img.shape[0]//tile_size_in, tile_size_in, img.shape[1]//tile_size_in, tile_size_in, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size_in, tile_size_in, 3)
    # if the image has less tiles than n_tiles we pad
    if len(img) < n_tiles:
        img = np.pad(img,[[0, n_tiles - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)   
    # select the tiles with the most tissue
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
    img = img[idxs]
    # store the tiles
    tiles = []
    for i in range(len(img)):
        tiles.append({'img':cv2.resize(img[i], (tile_size_out, tile_size_out)), 'idx':i})
    return tiles


all_tiles_test = {}
for i in tqdm(range(df_test.shape[0])):
    row = df_test.loc[i]
    img_id = row.image_id    
    tiff_file = os.path.join("./test/test", f"{img_id}.tiff")
    img = skimage.io.MultiImage(tiff_file)[-1]
    all_tiles_test[i] = get_tiles(img, 1024, 36, 128)
df_test["tiles"] = list(all_tiles_test.values())
df_test.to_csv("test_with_tiles.csv", index=False)
    

all_tiles_train = {}
for i in tqdm(range(df_train.shape[0])):
    row = df_train.loc[i]
    img_id = row.image_id    
    tiff_file = os.path.join("./train/train", f"{img_id}.tiff")
    img = skimage.io.MultiImage(tiff_file)[-1]
    all_tiles_train[i] = get_tiles(img, 1024, 36, 128)
df_train["tiles"] = list(all_tiles_train.values())
df_train.to_csv("train_with_tiles.csv", index=False)