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
from misc.helper_functions import denormalize_image


def plot_heatmaps(image, 
                  gt_heatmaps,
                  pred_heatmaps,
                  save_to,
                  file_name,
                  num_of_keypts = 6
                  ):
    """
    Args:
      image (numpy.ndarray)
      gt_heatmaps (torch.tensor)
      pred_heatmapts (torch.tensor)
      save_to (str): path to save the figure
      file_name (str): figure name without the extension
      num_of_keypts (int)
    """
    gt_heatmaps = gt_heatmaps.cpu().detach()
    pred_heatmaps = pred_heatmaps.cpu().detach()
  
    fig, ax = plt.subplots(2, num_of_keypts + 1, figsize=(14, 5))
    title = 'ground-truth (top) and predictions (bottom)'
    fig.suptitle(title, fontsize=16)

    ax[0, 0].imshow(image)
    ax[1, 0].imshow(image)

    plt.axis('off')
    for row in range(2):
        for i in range(6):
            # un-transform the image data    
            heatmap = gt_heatmaps[i] if row==0 else pred_heatmaps[i] 
            # convert torch to numpy, and scale to [0, 255]
            heatmap = heatmap.numpy()*255
            # convert to uint8 (8 bit format)
            heatmap = heatmap.astype(np.uint8)
        
            # resize the heatmap
            heatmap_width, heatmap_height = int(heatmap.shape[1]*4), int(heatmap.shape[0]*4)
            heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            image_n_heatmap = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

            ax[row, i+1].imshow(image_n_heatmap)  
            ax[row, i+1].axis('off')

    plt.savefig(os.path.join(save_to, file_name + '.jpg'))


if __name__ == '__main__':

    # inputs
    dataset_name = 'FDR_1k'
    checkpoint_dir = os.path.join(os.getcwd(), 'logs', 'test_exp', 'test_checkpoint.pth')
    fig_output_dir = os.path.join(os.getcwd(), 'temp')

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

    # convert heatmaps_gt to torch tensor and squeeze
    heatmaps_gt = torch.from_numpy(heatmaps_gt).squeeze()

    # make preds
    heatmaps_pred = model(image.unsqueeze(0).to(device)).squeeze()
    image = denormalize_image(image.cpu(), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # save sample original img
    img = plt.imread(labels['imgPath'])
    fig_name = 'sample'
    plt.imsave(os.path.join(fig_output_dir, fig_name + '.jpg'), img)
    
    # save plot_heatmaps
    plot_heatmaps(image, heatmaps_gt, heatmaps_pred, save_to=fig_output_dir, file_name='plot_heatmaps')
