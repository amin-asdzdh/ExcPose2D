"""Train a HRNet model on a custom dataset

Usage: python path/to/train.py --dataset_dir './datasets/FDR_1k' -p './pretrained_models/pose_hrnet_w48_384x288.pth' --vis_enabled False

"""

import argparse
import ast
import os
import random
import sys
import cv2
import torch
import numpy as np
from datetime import datetime

from utils.train_engine import Train
from utils.datasets import PoseDataset
from misc.Transforms import Rescale


'''
# pass in the arguments
# '--checkpoint_path', '/content/drive/My Drive/Colab Notebooks/My Projects/Excavator_Pose_Estimation/logs/20200730_0409/checkpoint_last.pth'

sys.argv = ['train_colab',
            '--pretrained_weight_path', './pretrained_models/pose_hrnet_w48_384x288.pth', 
            '--dataset_dir', './datasets/FDR_1k',
            '--vis_enabled', 1
            ]
'''

def main(exp_name,
         epochs,
         batch_size,
         num_workers,
         lr,
         disable_lr_decay,
         lr_decay_steps,
         lr_decay_gamma,
         optimizer,
         weight_decay,
         momentum,
         nesterov,
         pretrained_weight_path,
         checkpoint_path,
         log_path,
         disable_tensorboard_log,
         model_c,
         model_nof_joints,
         model_bn_momentum,
         disable_flip_test_images,
         image_resolution,
         seed,
         device,
         dataset_dir,
         vis_enabled
         ):

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True  # Enables cudnn
        torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

    # torch device
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    lr_decay = not disable_lr_decay
    use_tensorboard = not disable_tensorboard_log
    flip_test_images = not disable_flip_test_images
    image_resolution = ast.literal_eval(image_resolution)
    lr_decay_steps = ast.literal_eval(lr_decay_steps)
    
    print("\nLoading train and validation datasets...")
    print('\nvisibility status is enabled: ', vis_enabled, '\n')
    train_dataset_dir = os.path.join(dataset_dir, 'train')
    val_dataset_dir = os.path.join(dataset_dir, 'val')
    # load train and val datasets
    ds_train = PoseDataset(dataset_dir = train_dataset_dir,
               is_train = True,
               vis_enabled = vis_enabled
               )

    ds_val = PoseDataset(dataset_dir = val_dataset_dir,
               is_train = False,
               vis_enabled = vis_enabled
               )

    train = Train(
        exp_name=exp_name,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        loss='JointsMSELoss',
        lr=lr,
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps,
        lr_decay_gamma=lr_decay_gamma,
        optimizer=optimizer,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        pretrained_weight_path=pretrained_weight_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        use_tensorboard=False,
        model_c=model_c,
        model_nof_joints=model_nof_joints,
        model_bn_momentum=model_bn_momentum,
        flip_test_images=False,
        device=device,
        train_dataset_dir = train_dataset_dir,
        val_dataset_dir = val_dataset_dir
    )

    train.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=5)
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=8)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=4)
    parser.add_argument("--lr", "-l", help="initial learning rate", type=float, default=0.001)
    parser.add_argument("--disable_lr_decay", help="disable learning rate decay", action="store_true")
    parser.add_argument("--lr_decay_steps", help="learning rate decay steps", type=str, default="(170, 200)")
    parser.add_argument("--lr_decay_gamma", help="learning rate decay gamma", type=float, default=0.1)
    parser.add_argument("--optimizer", "-o", help="optimizer name. Currently, only `SGD` and `Adam` are supported.",
                        type=str, default='Adam')
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.)
    parser.add_argument("--momentum", "-m", help="momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", help="enable nesterov", action="store_true")
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--checkpoint_path", "-c",
                        help="previous checkpoint path. Checkpoint will be loaded before training starts. It includes "
                             "the model, the optimizer, the epoch, and other parameters.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./logs')
    parser.add_argument("--disable_tensorboard_log", "-u", help="disable tensorboard logging", action="store_true")
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=6)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--vis_enabled", type=str, default='False')
    

    args = parser.parse_args()

    main(**args.__dict__)