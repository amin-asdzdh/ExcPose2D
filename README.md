# Documentation

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





