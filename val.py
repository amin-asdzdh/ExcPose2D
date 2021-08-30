import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm

from models.hrnet.hrnet import HRNet


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/eval/RealSet1', help='dataset path')
    parser.add_argument('--weights', type=str, default=None, help='checkpoint.pth path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default=None, help='device')
    opt = parser.parse_args()
    return opt

def run(dataset,
        weights,
        batch_size,
        device):

    # set device and load model
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    model = HRNet(c=mode_c, nof_joints=model_nof_joints, bn_momentum=model_bn_momentum).to(device)
    

    # configure
    # 
    # dataloader
    # 
    #   

    pass

def main(opt):
    pass


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)