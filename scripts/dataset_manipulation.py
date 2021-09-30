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
    new_img_ids = [str(i) + '.jpg' for i in range(len(ds_1))]
    for i in tqdm(range(len(ds_1))):
        src = os.path.join(ds_1_dir, 'images', ds_1.iloc[i]['image_id'])
        dst = os.path.join(ds_merged_dir, 'images', new_img_ids[i])
        copyfile(src, dst)
        ds_1.at[i, 'image_id'] = new_img_ids[i]

    new_img_ids = [str(i+len(ds_1)) + '.jpg' for i in range(len(ds_2))]
    for i in tqdm(range(len(ds_2))):
        src = os.path.join(ds_2_dir, 'images', ds_2.iloc[i]['image_id'])
        dst = os.path.join(ds_merged_dir, 'images', new_img_ids[i])
        copyfile(src, dst)
        ds_2.at[i, 'image_id'] = new_img_ids[i]

    # merge labels
    ds_merged = pd.concat([ds_1, ds_2]).reset_index(drop=True)
    # shuffle
    ds_merged = ds_merged.sample(frac=1).reset_index(drop=True)

    print(ds_merged.head())
    
    ds_merged.to_csv(os.path.join(ds_merged_dir, 'labels', 'labels.csv'), index=False)


if __name__ == "__main__":

    ds_1 = os.path.join('E:', os.sep, 'My Datasets', 'Excavator_Pose_2D', 'AutCon2020', 'Archive', 'Experiment_2', 'prepared_datasets', '8')
    ds_2 = os.path.join('E:', os.sep, 'My Datasets', 'Excavator_Pose_2D', 'AutCon2020', 'Archive', 'Experiment_2', 'prepared_datasets', '9')
    ds_merged = os.path.join('E:', os.sep, 'My Datasets', 'Excavator_Pose_2D', 'AutCon2020', 'Archive', 'Experiment_2', 'prepared_datasets', '10')
    
    merge_datasets(ds_1, ds_2, ds_merged)
