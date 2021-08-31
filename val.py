import argparse
import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from models.hrnet.hrnet import HRNet
from utils.loss import JointsMSELoss
from utils.datasets import PoseDataset

# !!!!
# Note: 
# parameters passed to the model are curretly hardcoded (need to add as argument)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/eval/RealSet1', help='./path/to/dataset')
    parser.add_argument('--weights', type=str, default=None, help='./path/to/checkpoint.pth')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
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
    print('device: ', device)

    model = HRNet(c=48, nof_joints=6, bn_momentum=0.1).to(device)
    model.eval()

    # define loss
    loss_fn = JointsMSELoss().to(device)

    # load checkpoint
    print("Loading checkpoint ...\n", weights, '\n')
    checkpoint = torch.load(weights, map_location=device)
    epoch = checkpoint['epoch']
    print('epoch ', epoch)
    model.load_state_dict(checkpoint['model'])
    
    # load dataset and dataloader
    ds = PoseDataset(dataset_dir=dataset, is_train=False, vis_enabled='False')
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    print('Dataset length : ', len(dataloader)*batch_size)

    # initialize variables 
    loss_all = []
    acc_all = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluation')
        for step, (image, target, target_weight, joints_data) in enumerate(pbar):

            image = image.to(device)
            target = target.to(device)
            target_weight = target_weight.to(device)

            output = model(image)

            # calculate loss
            loss = loss_fn(output, target, target_weight)

            # calculation accuracy
            accs, avg_acc, cnt, joints_preds, joints_targets = ds.evaluate_accuracy(output, target)

            loss_all.append(loss)
            acc_all.append(avg_acc)
    
    mean_loss = np.average(loss_all)
    mean_acc = np.average(acc_all)

    print('mean_loss: ', mean_loss)
    print('mean_acc :', mean_acc)

    print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
