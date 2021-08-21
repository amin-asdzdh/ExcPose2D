import numpy as np
import matplotlib.pyplot as plt
import cv2

def merge_heatmaps(heatmaps):
  # takes a set of heatmaps for one sample -> torch.Size([6, 96, 72])
  
  heatmaps = heatmaps.detach()
  heatmaps = heatmaps.cpu()

  heatmaps_merged = np.zeros((384, 288), dtype=float)
  num_of_joints = 6
  
  for i in range(num_of_joints):

    heatmap = heatmaps[i]
    # convert torch to numpy
    heatmap = heatmap.numpy()
    # convert the range to [0, 255]
    heatmap = heatmap*255
    # convert to uint8 (8 bit format)
    heatmap = heatmap.astype(np.uint8)

    # resize the heatmap
    heatmap_width = int(heatmap.shape[1]*4)
    heatmap_height = int(heatmap.shape[0]*4)
    heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

    heatmaps_merged -= heatmap

  return heatmaps_merged 


def denormalize_image(image, mean, std):
  # denormalize torch image - torch.Size([3, 384, 288])

  # convert image to numpy
  image = image.numpy()
  # reshape the torch image shape (C x H x W) into a numpy image shape (H x W x C)
  image = image.transpose((1, 2, 0))
  
  # denormalzie 
  image[:, :, 0] = image[:, :, 0] * std[0] + mean[0]
  image[:, :, 1] = image[:, :, 1] * std[1] + mean[1]
  image[:, :, 2] = image[:, :, 2] * std[2] + mean[2]
  image = image*255
  
  # convert to 8 bit int type
  image = image.astype(np.uint8)

  return image


def show_all_heatmaps(image, heatmaps, 
                      separate = False, 
                      image_resolution = (384, 288), 
                      num_of_joints = 6,
                      fig_title = 'heatmaps',
                      fig_no = 1):
  # plot all heatmaps in a subplot
  # heatmaps -> torch
  
  heatmaps = heatmaps.cpu().detach()
  
  if separate == 0:

    heatmaps_merged = np.zeros(image_resolution, dtype=float)
    for i in range(num_of_joints):

      heatmap = heatmaps[i]
      # convert torch to numpy
      heatmap = heatmap.numpy()
      # convert the range to [0, 255]
      heatmap = heatmap*255
      # convert to uint8 (8 bit format)
      heatmap = heatmap.astype(np.uint8)

      # resize the heatmap
      heatmap_width = int(heatmap.shape[1]*4)
      heatmap_height = int(heatmap.shape[0]*4)
      heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

      heatmaps_merged -= heatmap

    heatmaps_merged = heatmaps_merged.astype(np.uint8)
    heatmaps_merged = cv2.applyColorMap(heatmaps_merged, cv2.COLORMAP_JET)

    image_n_heatmaps = cv2.addWeighted(heatmaps_merged, 0.5, image, 0.5, 0)
    
    plt.figure(fig_no, figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax = plt.subplot(1, 2, 2)
    plt.imshow(image_n_heatmaps)
    plt.show(block=False)

  elif separate == 1:

    plt.figure(fig_no, figsize=(15,3))
    ax = plt.subplot(1, num_of_joints+1, 1)
    #ax.set_title(fig_title, color='gray')
    plt.imshow(image)
    plt.text(4, 1.1, fig_title,
         horizontalalignment='center',
         fontsize=15,
         transform = ax.transAxes)
    plt.axis('off')

    for i in range(6):
            ax = plt.subplot(1, num_of_joints+1, i+2)

            # un-transform the image data    
            heatmap = heatmaps[i]
            # convert torch to numpy
            heatmap = heatmap.numpy()
            # convert the range to [0, 255]
            heatmap = heatmap*255
            # convert to uint8 (8 bit format)
            heatmap = heatmap.astype(np.uint8)

            # resize the heatmap
            heatmap_width = int(heatmap.shape[1]*4)
            heatmap_height = int(heatmap.shape[0]*4)
            heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            image_n_heatmap = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

            #plt.tight_layout()
            plt.imshow(image_n_heatmap)  
            plt.axis('off')

    plt.show(block=False)


def show_all_heatmaps_2(image, 
                        gt_heatmaps,
                        pred_heatmaps,
                        separate = False, 
                        image_resolution = (384, 288), 
                        num_of_joints = 6,
                        fig_no = 1,
                        exp_name = None,
                        fig_title = None
                        ):
  # plot all heatmaps in a subplot
  # heatmaps -> torch
  
  gt_heatmaps = gt_heatmaps.cpu().detach()
  pred_heatmaps = pred_heatmaps.cpu().detach()
  
  if separate == 0:

    heatmaps_merged = np.zeros(image_resolution, dtype=float)
    
    for i in range(num_of_joints):

      heatmap = heatmaps[i]
      # convert torch to numpy
      heatmap = heatmap.numpy()
      # convert the range to [0, 255]
      heatmap = heatmap*255
      # convert to uint8 (8 bit format)
      heatmap = heatmap.astype(np.uint8)

      # resize the heatmap
      heatmap_width = int(heatmap.shape[1]*4)
      heatmap_height = int(heatmap.shape[0]*4)
      heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

      heatmaps_merged -= heatmap

    heatmaps_merged = heatmaps_merged.astype(np.uint8)
    heatmaps_merged = cv2.applyColorMap(heatmaps_merged, cv2.COLORMAP_JET)

    image_n_heatmaps = cv2.addWeighted(heatmaps_merged, 0.5, image, 0.5, 0)
    
    plt.figure(fig_no, figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax = plt.subplot(1, 2, 2)
    plt.imshow(image_n_heatmaps)
    plt.show(block=False)

  elif separate == 1:

    fig, ax = plt.subplots(2, num_of_joints+1, figsize=(14, 5))
    title = 'ground-truth (top) and predictions (bottom) - ' + exp_name
    fig.suptitle(title, fontsize=16)
    
    #ax.set_title(fig_title, color='gray')
    ax[0, 0].imshow(image)
    ax[1, 0].imshow(image)
    
    plt.axis('off')

    for i in range(6):
            #ax = plt.subplot(1, num_of_joints+1, i+2)

            # un-transform the image data    
            heatmap = gt_heatmaps[i]
            # convert torch to numpy
            heatmap = heatmap.numpy()
            # convert the range to [0, 255]
            heatmap = heatmap*255
            # convert to uint8 (8 bit format)
            heatmap = heatmap.astype(np.uint8)

            # resize the heatmap
            heatmap_width = int(heatmap.shape[1]*4)
            heatmap_height = int(heatmap.shape[0]*4)
            heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            image_n_heatmap = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

            #plt.tight_layout()
            ax[0, i+1].imshow(image_n_heatmap)  
            ax[0, i+1].axis('off')

    for i in range(6):
            #ax = plt.subplot(2, num_of_joints+1, i+2)

            # un-transform the image data    
            heatmap = pred_heatmaps[i]
            # convert torch to numpy
            heatmap = heatmap.numpy()
            # convert the range to [0, 255]
            heatmap = heatmap*255
            # convert to uint8 (8 bit format)
            heatmap = heatmap.astype(np.uint8)

            # resize the heatmap
            heatmap_width = int(heatmap.shape[1]*4)
            heatmap_height = int(heatmap.shape[0]*4)
            heatmap = cv2.resize(heatmap, (heatmap_width, heatmap_height), interpolation = cv2.INTER_AREA)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            image_n_heatmap = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

            #plt.tight_layout()
            ax[1, i+1].imshow(image_n_heatmap)  
            ax[1, i+1].axis('off')

    plt.show(block=False)


def net_sample_output(model, datasetLoader, device):
    
    # iterate through the test dataset
    for i, sample in enumerate(datasetLoader):
        
        image_batch = sample[0]
        target_batch = sample[1]
        target_weight_batch = sample[2]
        joints_data_batch = sample[3]
  
        # get sample data: ground truth keypoints
        keypoints_batch = joints_data_batch['joints']
        #print(image.shape)
        #print(keypoints_batch.shape)
        

        # place images and labels on GPU
        image_batch = image_batch.to(device)

        # forward pass to get model output
        output_pts = model(image_batch)

       
        # reshape to batch_size x 68 x 2 pts
        # output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return image_batch, target_batch, target_weight_batch, keypoints_batch, output_pts


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


def calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)
    target = target.type(torch.float32)
    dists = torch.zeros((preds.shape[1], preds.shape[0])).to(preds.device)
    # for n in range(batch_size)
    for n in range(preds.shape[0]):
        # for c in range(no_of_joints)
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                # normalize preds and targets
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # # dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                dists[c, n] = torch.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """
    Return percentage below threshold while ignoring values with a -1
    """
    dist_cal = torch.ne(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        # torch.lt is an element-wise Less-Than (<) operator, returns tensor with boolean values
        return torch.lt(dists[dist_cal], thr).float().sum() / num_dist_cal
    else:
        return -1
    
    
    
def print_dataset_stats(dataset):    
    # iterate through the dataset and print some stats about the first few samples
    print('\nDataset length: ', len(dataset))
    
    print('\nLoading a few samples from the dataset')
    for i in range(2):
        image, target, target_weight, joints_data = dataset[i]
        print('\n\tsample no.', i,':')
        print('\timage: ', image.shape)
        print('\theatmap: ', target.shape)
        print('\tjoints: ', joints_data['joints'].shape)
