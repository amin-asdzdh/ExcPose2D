import numpy as np
import torch

def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % width  # column
    preds[:, :, 1] = torch.floor(idx / width)  # row

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def dist_acc(dists, thr=0.5):
    """
    Return percentage below threshold while ignoring values with a -1
    """
    dist_cal = torch.ne(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.lt(dists[dist_cal], thr).float().sum() / num_dist_cal
    else:
        return -1


def calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)
    target = target.type(torch.float32)
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


def evaluate_pck_accuracy(output, target, hm_type='gaussian', thr=0.05):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than y,x locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))

    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = torch.ones((pred.shape[0], 2)) * torch.tensor([np.sqrt(h**2 + w**2), np.sqrt(h**2 + w**2)], dtype=torch.float32)
        norm = norm.to(output.device)
    else:
        raise NotImplementedError
    dists = calc_dists(pred, target, norm)

    acc = torch.zeros(len(idx)).to(dists.device)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc, cnt, pred, target, dists