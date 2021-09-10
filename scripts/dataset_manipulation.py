import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copyfile
from tqdm import tqdm
from pathlib import Path


def merge_datasets(ds_1_dir, ds_2_dir, ds_merged_dir):
    ds_1 = pd.read_csv(os.path.join(ds_1_dir, 'labels', 'labels.csv'))
    ds_2 = pd.read_csv(os.path.join(ds_2_dir, 'labels', 'labels.csv'))

    # create target folders
    Path(ds_merged_dir).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(ds_merged_dir, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(ds_merged_dir, 'labels')).mkdir(parents=True, exist_ok=True)
    
    # copy images to target folder
    for i in tqdm(range(len(ds_1))):
        src = os.path.join(ds_1_dir, 'images', ds_1.iloc[i]['image_id'])
        dst = os.path.join(ds_merged_dir, 'images', ds_1.iloc[i]['image_id'])
        copyfile(src, dst+'temp')

    for i in tqdm(range(len(ds_2))):
        src = os.path.join(ds_2_dir, 'images', ds_2.iloc[i]['image_id'])
        dst = os.path.join(ds_merged_dir, 'images', ds_2.iloc[i]['image_id'])
        copyfile(src, dst)
        os.rename(dst, dst+'temp')
    
    # merge labels
    ds_merged = pd.concat([ds_1, ds_2]).reset_index(drop=True)
    # shuffle
    ds_merged = ds_merged.sample(frac=1).reset_index(drop=True)

    # generate new img names
    new_img_ids = [str(i) + '.jpg' for i in range(len(ds_merged))]

    print(ds_merged.head())

    # update names
    for i in range(len(ds_merged)):
        old_id = ds_merged.iloc[i]['image_id']
        new_id = new_img_ids[i]
        # rename file
        os.rename(os.path.join(ds_merged_dir, 'images', old_id)+'temp', os.path.join(ds_merged_dir, 'images', new_id))
        # rename in table
        ds_merged.at[i, 'image_id'] = new_id

    print(ds_merged.head())
    ds_merged.to_csv(os.path.join(ds_merged_dir, 'labels', 'labels.csv'), index=False)


if __name__ == "__main__":
    ds_1 = os.path.join(os.getcwd(), 'datasets', 'eval', 'being_prepared', 'org+HF+IC+RR')
    ds_2 = os.path.join(os.getcwd(), 'datasets', 'eval', 'being_prepared', 'RealSet_RandomTranslate')
    ds_merged = os.path.join(os.getcwd(), 'datasets', 'eval', 'being_prepared', 'org+HF+IC+RR+RT')
    
    merge_datasets(ds_1, ds_2, ds_merged)
