

# Excavator 2D Pose Estimation



## Description

This project utilizes High Representation Network, a state-of-the-art convolutional neural network architecture to estimate the 2D keypoints defining the full body pose of excavators  from images.  

This repo is part of the publication "A Assadzadeh et al., 'Vision-based excavator pose estimation using synthetically generated datasets with domain randomization', Automation in Construction, 2022" that utilizes synthetically generated datasets to train a deep learning model for excavator pose estimation. 



## Demo

Demo of the model trained on a combination of synthetic and real images
![](resources/demo.gif)



## Install

```
$ git clone https://github.com/N1M49/ExcPose2D.git
$ cd ExcPose2D
$ pip install -r requirements.txt
```

## Quick Start Examples

Training
```bash
$ python train.py --dataset_dir './datasets/FDR_1k' -p './pretrained_models/pose_hrnet_w48_384x288.pth' --vis_enabled False
```

Evaluation

```bash
$ python val.py --dataset './datasets/eval/RealSet_test_debug' --weights './experiments/archived/Vis0_FDR_15k_best.pth' --pck_thr 0.05
```




##### References

https://github.com/stefanopini/simple-HRNet