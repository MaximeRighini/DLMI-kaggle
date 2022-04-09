############################################################################################################
#The purpose of this python file is to create the reduced images with the tiles method. We use exactly the same method as in the #exploration notebook, but we just resize the tiles to a size (128, 128, 3). We chose this size because we noticed that the higher the #resolution of the images, the better the results. A larger size would crash the colab ram.
############################################################################################################

import numpy as np
import pandas as pd
import os
import cv2
import skimage.io
from tqdm import tqdm

def get_tiles(img, tile_size_in, n_tiles, tile_size_out):
    # find by how much we must pad the image so that the image dims are multiples of tile_size_in
    h, w, c = img.shape
    pad_h = (tile_size_in - h % tile_size_in) % tile_size_in
    pad_w = (tile_size_in - w % tile_size_in) % tile_size_in
    # pad the image with constant values (255)
    img = np.pad(img, [[pad_h//2, pad_h - pad_h//2], [pad_w//2, pad_w - pad_w//2], [0, 0]],
                 constant_values=0)
    # reshape the image and swap the order of the dims
    img = img.reshape(img.shape[0]//tile_size_in, tile_size_in, img.shape[1]//tile_size_in, tile_size_in, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size_in, tile_size_in, 3)
    # if the image has less tiles than n_tiles we pad
    if len(img) < n_tiles:
        img = np.pad(img,[[0, n_tiles - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=0)   
    # select the tiles with the most tissue
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[-n_tiles:]
    img = img[idxs]
    # store the tiles
    tiles = []
    for i in range(len(img)):
        # add the tiles and reshape them
        tiles.append(cv2.resize(img[i], (tile_size_out, tile_size_out)).tolist())
    return tiles

# We'll store the tiles in a DataFrame
def add_column_tiles(df, path_imgs):
    all_tiles = {} # This dictionary will store all the files
    for i in tqdm(range(df.shape[0])): # Go through each index of the dataframe
        row = df.iloc[i] # Find the row associated to that index
        img_id = row.image_id  # Find the slide
        # Load the slide
        tiff_file = os.path.join(path_imgs, f"{img_id}.tiff")
        img = skimage.io.MultiImage(tiff_file)[-1]
        # Remove the background
        img[img.mean(axis=-1)>=235] = 0
        # Create the tiles (we reshape in size (128, 128, 3)
        all_tiles[i] = get_tiles(img, int(np.sqrt((img!=0).sum())/14), 100, 128)
    # Convert the dictionary in a dataframe column
    df["tiles"] = list(all_tiles.values())


# Add the tiles to the train and test dataframes and export them
df_train = pd.read_csv("train.csv").reset_index(drop=True)
df_test = pd.read_csv("test.csv").reset_index(drop=True)

add_column_tiles(df_train, "./train/train")
df_train.to_csv("train_with_tiles.csv", index=False)

add_column_tiles(df_test, "./test/test")
df_test.to_csv("test_with_tiles.csv", index=False)