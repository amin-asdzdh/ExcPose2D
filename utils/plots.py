import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix()) # add ExcPose2D/ to path

from utils.datasets import PoseDataset


if __name__ == '__main__':

    dataset_name = 'FDR_1k'

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device, '\n')

    # directories
    dataset_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, 'train')
    
    dataset = PoseDataset(dataset_dir=dataset_dir, is_train=False)

    print(dataset)
