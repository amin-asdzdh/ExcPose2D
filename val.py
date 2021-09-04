import argparse
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from models.hrnet.hrnet import HRNet
from utils.loss import JointsMSELoss
from utils.datasets import PoseDataset
from misc.utils import get_max_preds
from utils.metrics import evaluate_pck_accuracy

# !!!!
# Note: 
# parameters passed to the model are curretly hardcoded (need to add as argument)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/eval/RealSet_test', help='./path/to/dataset')
    parser.add_argument('--weights', type=str, default=None, help='./path/to/checkpoint.pth')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default=None, help='device')
    parser.add_argument('--pck_thr', type=float, default=0.05, help='pck threshold as a ratio of img diag')
    opt = parser.parse_args()
    return opt


def _calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)     # pred joint coords
    target = target.type(torch.float32)   # target joint coords
    dists = torch.zeros((preds.shape[1], preds.shape[0])).to(preds.device)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # # dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                dists[c, n] = torch.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def run2(dataset,
        weights,
        batch_size,
        device,
        pck_thr,
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

    files = os.listdir('./experiments/archived')
    with open(os.path.join('./experiments/archived/', 'epochs.txt'), 'w') as fd:
        fd.writelines('checkpoint' + ', ' + 'epoch' + '\n')
    for file in files:
        # load checkpoint
        print("Loading checkpoint ...\n", file)
        checkpoint = torch.load(os.path.join('./experiments/archived', file), map_location=device)
        epoch = checkpoint['epoch']
        print("Checkpoint's epoch: ", epoch, '\n')
        with open(os.path.join('./experiments/archived/', 'epochs.txt'), 'w') as fd:
            fd.writelines(str(os.path.join('./experiments/archived', file)) + ', ' + str(epoch) + '\n')


def run(dataset,
        weights,
        batch_size,
        device,
        pck_thr,
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
    print("Checkpoint's epoch: ", epoch)
    model.load_state_dict(checkpoint['model'])
    
    # load dataset and dataloader
    ds = PoseDataset(dataset_dir=dataset, is_train=False, vis_enabled='False')
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    print('batch_size: ', batch_size)
    print('dataset length : ', len(dataloader)*batch_size)

    # initialize variables 
    loss_all = []
    acc_all = []
    NE_all = []
    results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluation')
        for step, (image, target, target_weight, joints_data) in enumerate(pbar):

            image = image.to(device)
            target = target.to(device)
            target_weight = target_weight.to(device)

            output = model(image)

            # calculate loss
            loss = loss_fn(output, target, target_weight)

            # calculation accuracy (pck)
            accs, avg_acc, cnt, joints_preds, joints_targets, NEs = evaluate_pck_accuracy(output, target, thr=pck_thr)
            
            loss_all.append(loss.to('cpu'))
            acc_all.append(avg_acc.to('cpu'))
            NE_all.append(torch.mean(NEs).cpu().numpy())
            NEs = NEs.cpu().numpy()
            results.append([joints_data['imgId'].pop(), loss.to('cpu').item(), NEs[0].item(), NEs[1].item(), NEs[2].item(), NEs[3].item(), NEs[4].item(), NEs[5].item(), np.mean(NEs)])
            break
    
    results_df_cols = ['imgId', 'MSEloss', 'NE1', 'NE2', 'NE3', 'NE4', 'NE5', 'NE6', 'NEavg']
    results_df = pd.DataFrame(results, columns=results_df_cols)
    mean_loss = np.average(loss_all)
    mean_acc = round(np.average(acc_all), 4)
    NEavg = round(np.average(NE_all), 4)

    # save results
    log_path = os.path.join(os.getcwd(), 'logs', 'eval_results', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_path, 0o755, exist_ok=False)  # exist_ok=False to avoid overwriting
    parameters = [str(vars(opt))]
    with open(os.path.join(log_path, 'parameters.txt'), 'w') as fd:
        fd.writelines(parameters)

    results_df.to_csv(os.path.join(log_path, 'results.csv'))
    print('mean_loss: ', mean_loss)
    print(f'PCK@{pck_thr}: {mean_acc}')
    print(f'NEavg: {NEavg}')

    print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
def main(opt):
    #run(**vars(opt))
    run2(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
