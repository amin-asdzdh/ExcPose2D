import os
import sys
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix()) # add ExcPose2D/ to path

from utils.datasets import PoseDataset
from models.hrnet.hrnet import HRNet
from misc.checkpoint import load_checkpoint


if __name__ == '__main__':

    # inputs
    dataset_name = 'FDR_1k'
    checkpoint_dir = os.path.join(os.getcwd(), 'logs', 'test_exp', 'test_checkpoint.pth')

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\ndevice: ', device)

    # load dataset
    dataset_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, 'train')
    dataset = PoseDataset(dataset_dir=dataset_dir, is_train=False, vis_enabled='False')

    # load model
    model = HRNet(c=48, nof_joints=6).to(device)
    _, model, _, _ = load_checkpoint(checkpoint_dir, model, device=device)
    model.eval()

    # get a random sample
    sample_no = int(dataset.__len__()*random.random())
    image, heatmaps_gt, target_weights, labels = dataset.__getitem__(sample_no)

    # make preds
    heatmaps_pred = model(image.unsqueeze(0).to(device))

    print(labels['imgPath'])
    img = plt.imread(labels['imgPath'])

    plt.imsave(os.path.join(os.getcwd(), 'test.jpg'), img)
    # convert heatmaps_gt to torch tensor
    #heatmaps_gt = torch.from_numpy(heatmaps_gt)
    #heatmaps_gt = heatmaps_gt.unsqueeze(0)

