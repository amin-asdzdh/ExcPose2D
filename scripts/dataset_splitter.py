import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
from pathlib import Path

# notes
# set the source_dir
# set the target_dir
# modifications are required if it is to be used on Real dataset 

# source folder (containing all the images and labels)

def split(source_dir, target_dir, split_size=0.2):
        
    # split the labels 
    df_keypoints = pd.read_csv(os.path.join(source_dir, 'labels', 'labels.csv'))
    df_keypoints_train, df_keypoints_val = train_test_split(df_keypoints, test_size=split_size, shuffle=True)

    train_path = os.path.join(target_dir, 'train')
    val_path = os.path.join(target_dir, 'val')

    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(train_path, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(train_path, 'labels')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(val_path, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(val_path, 'labels')).mkdir(parents=True, exist_ok=True)

    # write labels to csv files
    df_keypoints_train.to_csv(os.path.join(train_path, 'labels', 'labels.csv'))
    df_keypoints_val.to_csv(os.path.join(val_path, 'labels', 'labels.csv'))

    # copy images to train set
    imgIds = df_keypoints_train['image_id'].to_numpy()


    for i in tqdm(range(len(imgIds))):
        img_name = imgIds[i]
        src = os.path.join(source_dir, 'images', img_name)
        dst = os.path.join(target_dir, 'train', 'images', img_name)
        copyfile(src, dst)


    # copy images to val set
    imgIds = df_keypoints_val['image_id'].to_numpy()

    for j in tqdm(range(len(imgIds))):
        img_name = imgIds[j]
        src = os.path.join(source_dir, 'images', img_name)
        dst = os.path.join(target_dir, 'val', 'images', img_name)
        copyfile(src, dst)    


if __name__ == "__main__":
    
    source_dir = os.path.join('E:', os.sep, 'My Datasets', 'Excavator_Pose_2D', 'AutCon2020', 'Archive', 'Experiment_2', 'prepared_datasets', 'Synthetic_78750')
    target_dir = os.path.join('E:', os.sep, 'My Datasets', 'Excavator_Pose_2D', 'AutCon2020', 'Archive', 'Experiment_2', 'prepared_datasets', 'Synthetic_78750_train')

    split(source_dir, target_dir, split_size=0.2)