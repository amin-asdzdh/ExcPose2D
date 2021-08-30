import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from models.hrnet.hrnet import HRNet
from utils.loss import JointsMSELoss

# !!!!
# Note: 
# parameters passed to the model are curretly hardcoded (need to add as argument)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/eval/RealSet1', help='./path/to/dataset')
    parser.add_argument('--weights', type=str, default=None, help='./path/to/checkpoint.pth')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default=None, help='device')
    opt = parser.parse_args()
    return opt

def run(dataset,
        weights,
        batch_size,
        device,
        num_workers=1):

    # set device and load model
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    model = HRNet(c=48, nof_joints=6, bn_momentum=0.1).to(device)
    
    # define loss
    loss_fn = JointsMSELoss().to(device)

    # load checkpoint
    print("Loading checkpoint ...\n", weights, '\n')
    checkpoint = torch.load(weights, map_location=device)
    epoch = checkpoint['epoch']
    print('epoch ', epoch)
    model.load_state_dict(checkpoint['model'])
    
    # load dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    print('Dataset length : ', len(dataloader))

    # initialize variables 



    # configure
    # 
    # dataloader
    # 
    #   

    pass

def main(opt):
    print('this is main!')
    print(type(opt))
    run(**vars(opt))
    pass


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
