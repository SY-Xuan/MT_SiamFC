# The implement of the paper "The Multi-task Fully Convolutional Siamese Network with Correlation Filter Layer for Real-Time Visual Tracking"

## Introduction
In this paper, we combine the correlation filter with the fully convolutional siamese network to enhance the ability of siamese tracker to distinguish similar object.

## Installation
### Requiements
1. Python3
2. Pytorch1.0+
3. GOT10K
4. opencv

### Training the tracker
1. You need to change the root of dataset in the file *train_VGG_fc.py* to your own root
2. Run:
```shell
    python3 train_VGG_fc.py
```

### Evaluate the tracker
1. The trained model is *model_BEST.pth* and you need to change some roots in the file *test.py* to perform experiments.
2. Run:
```shell
    python3 run.py --model the_model_path
```

### Running the demo
1. Run:
```shell
    python3 demo.py --model the model path --video_name the video path
```

## Refer to this Rep.
If you found our work is useful, thanks to you to cite our paper and star.
```
@inproceedings{xuan2019multi,
  title={The Multi-task Fully Convolutional Siamese Network with Correlation Filter Layer for Real-Time Visual Tracking},
  author={Xuan, Shiyu and Li, Shengyang and Zhao, Zifei and Han, Mingfei},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={123--134},
  year={2019},
  organization={Springer}
}
```

