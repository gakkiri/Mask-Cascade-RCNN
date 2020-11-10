# Mask-Cascade-RCNN

A simple Mask-Cascade-RCNN implementation.  

> Paper:
>
> [Masr RCNN](http://cn.arxiv.org/abs/1703.06870v3)  
> [Cascade RCNN](https://arxiv.org/abs/1712.00726)



# Motivation

The purpose of implementing this code is to comb the knowledge system of Mask-RCNN and Cascade-RCNN, and make the code readable and easy for others to learn. 

Most implementations of Mask-RCNN are very complex and hard to read, and this repo, without the need for compilation and complex installation, is very handy. The code structure was a reference to [Detectron2](https://github.com/facebookresearch/detectron2), and it was very easy to run on WINDOWS and LINUX.

# Environment

```
torch >= 1.4
torchvision >= 0.4
albumentations
pillow
timm
tensorboardX
easydict
opencv-python == 4.1.2.30
```

# Dataset

The dataset currently used is [WGISD](https://github.com/thsant/wgisd).

This is a small dataset, which is convenient for experiments.

# Experiments and results

The train curve

![curve](https://github.com/gakkiri/Mask-Cascade-RCNN/blob/master/imgs/curve.png?raw=true)

Result visualization

![sample](https://github.com/gakkiri/Mask-Cascade-RCNN/blob/master/imgs/sample.jpg?raw=true)

![result](https://github.com/gakkiri/Mask-Cascade-RCNN/blob/master/imgs/result.jpg?raw=true)
