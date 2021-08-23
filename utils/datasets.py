import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix()) # add ExcPose2D/ to path

from misc.transforms import Rescale
from misc.transforms import Rotate_90_CC
from misc.utils import evaluate_pck_accuracy
from misc.helper_functions import denormalize_image


class PoseDataset(Dataset):
  """
  PoseDataset class
  Excavator 2D pose
  """
  def __init__(self,
               dataset_dir = None,
               is_train = False,
               image_width = 288,
               image_height = 384,
               color_rgb = True,
               heatmap_sigma = 3,
               no_of_keypts = 6,
               vis_enabled = False):
    
    super(PoseDataset, self).__init__()

    self.dataset_dir = dataset_dir
    self.is_train = is_train
    self.image_width = image_width
    self.image_height = image_height
    self.color_rgb = color_rgb
    self.heatmap_sigma = heatmap_sigma
    self.no_of_keypts = no_of_keypts
    self.vis_enabled = vis_enabled
    
    self.images_path = os.path.join(self.dataset_dir, 'images')
    self.labels_path = os.path.join(self.dataset_dir, 'labels','labels.csv')

    self.image_size = (self.image_width, self.image_height)
    self.aspect_ratio = self.image_width * 1.0 / self.image_height
    self.heatmap_size = (int(self.image_width / 4), int(self.image_height / 4))
    self.heatmap_type = 'gaussian'
    
    self.transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
    
    # load labels as a dataframe
    self.annotations = pd.read_csv(self.labels_path)
    # get all imgIds
    self.imgIds = self.annotations['image_id']

    # load and format annotations for each image
    print('\nloading annotations from: ', self.labels_path)
    self.data = []
    for imgId in tqdm(self.imgIds):

      sample_labels = self.annotations.loc[self.annotations['image_id'] == imgId]
      
      # get kypts, convert to numpy array, and reshape
      keypts = sample_labels[['body_end_x', 'body_end_y',
                              'cab_boom_x', 'cab_boom_y',
                              'boom_arm_x', 'boom_arm_y',
                              'arm_bucket_x', 'arm_bucket_y',
                              'bucket_end_left_x', 'bucket_end_left_y',
                              'bucket_end_right_x', 'bucket_end_right_y']].to_numpy().reshape((self.no_of_keypts, 2))
 
      keypts_v = np.ones((self.no_of_keypts, 2), dtype=np.float)
      
      # use visiblity status labels if enabled
      if self.vis_enabled == 'True':
        print('it is actually true')
        v = sample_labels[['body_end_v',
                           'cab_boom_v',
                           'boom_arm_v',
                           'arm_bucket_v',
                           'bucket_end_left_v',
                           'bucket_end_right_v']].to_numpy()
        keypts_v[:, 0] = v
        keypts_v[:, 1] = v 
      elif self.vis_enabled == 'False':
        pass
      else:
        raise ValueError("Invalid value for vis_enabled, must be 'True' or 'False'")

      self.data.append({
        'imgId': imgId,
        'imgPath': os.path.join(self.images_path, imgId),
        'keypts': keypts,
        'keypts_visibility': keypts_v
        })
    
    # annotations loaded
    print('\n Annotations loaded\n')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    sample_data = self.data[index].copy()

    # read the image from disk
    image = cv2.imread(sample_data['imgPath'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if self.color_rgb:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
      raise ValueError('Fail to read %s' % image)


    # create sample dict for transformation
    sample = {'image': image, 'keypoints': sample_data['keypts']}

    # apply transforms
    rotate_transform = Rotate_90_CC()
    transformed_sample = rotate_transform(sample)
    rescale_transform = Rescale((384, 288))
    transformed_sample = rescale_transform(transformed_sample)

    # update the transformed sample (image & keypoints)
    image = transformed_sample['image']
    sample_data['keypts'] = transformed_sample['keypoints']
    
    # convert image to tensor and normalize
    image = self.transform(image)

    # generate heatmaps for keypoints
    target, target_weight = self._generate_target(sample_data['keypts'], sample_data['keypts_visibility'])

    return image, target.astype(np.float32), target_weight.astype(np.float32), sample_data

  ############## evaluate_accuracy ##############

  # method for evaluating accuracy
  # Calculate accuracy according to PCK (Probability of Correct Keypoints),
  # but uses ground truth heatmap rather than y,x locations
  # First value to be returned is average accuracy across 'idxs',
  # followed by individual accuracies
    
  def evaluate_accuracy(self, output, target, hm_type=None, thr=None):
      if hm_type is not None and thr is not None:
          accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target, hm_type, thr)
      else:
          accs, avg_acc, cnt, joints_preds, joints_target = evaluate_pck_accuracy(output, target)
   
      return accs, avg_acc, cnt, joints_preds, joints_target 



 ######################## helper functions ########################

  def _generate_target(self, joints, joints_vis):
    """
    :param joints: [no_of_keypts, 3]?
    :param joints_vis: [no_of_keypts, 3] I don't understand why there is 3 columns?
    :return: target, target_weight(1: visible, 0: invisible)
    """
    # initialize target_weight
    target_weight = np.ones((self.no_of_keypts, 1), dtype=np.float)
    target_weight[:, 0] = joints_vis[:, 0]

    if self.heatmap_type == 'gaussian':
      # initialize target (target is the gaussian heatmap for each joint)
      target = np.zeros((self.no_of_keypts,
                         self.heatmap_size[1],
                         self.heatmap_size[0]),
                        dtype=np.float32)
      
      tmp_size = self.heatmap_sigma * 3   # don't exactly know why

      for joint_id in range(self.no_of_keypts):
        #print('\n ### INSIDE _generate_target ###\n')
        #print(joints)
        feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        
        
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
          # If not, just return the image as is
          target_weight[joint_id] = 0
          continue
          
        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

        v = target_weight[joint_id]
        
        if v > 0.5:
          target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
    else:
      raise NotImplementedError

    return target, target_weight

if __name__=='__main__':

  dataset_name = 'FDR_1k'
  dataset_subdir = 'train'
  dataset_dir = os.path.join(os.getcwd(), 'datasets', dataset_name, dataset_subdir)
  dataset = PoseDataset(dataset_dir = dataset_dir, vis_enabled=True)
          

  image, heatmaps_gt, target_weight, sample_data = dataset.__getitem__(20)
  image = denormalize_image(image.cpu(), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  
  heatmaps_gt = torch.from_numpy(heatmaps_gt).unsqueeze(0)
  exp_name = '20200910_2241'
  #show_all_heatmaps_rotate(image, heatmaps_gt[0], heatmaps_gt[0], separate=True, fig_no = 1, exp_name= exp_name)
  
  target_weight = torch.from_numpy(target_weight).unsqueeze(0)
  print(target_weight)
