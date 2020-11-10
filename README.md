# Mask-Cascade-RCNN

A simple Mask-Cascade-RCNN implementation. **Not perfect, only for learning and reference.**

> Paper:
>
> [Masr RCNN](http://cn.arxiv.org/abs/1703.06870v3)  
> [Cascade RCNN](https://arxiv.org/abs/1712.00726)



# Motivation

The purpose of implementing this code is to comb the knowledge system of Mask-RCNN and Cascade-RCNN, and make the code readable and easy for others to learn. 

Most implementations of Mask-RCNN are very complex and hard to read, and this repo, without the need for compilation and complex installation, is very handy. The code structure was a reference to [Detectron2](https://github.com/facebookresearch/detectron2), and it was very easy to run on WINDOWS and LINUX.

# How to use
* If you want to train this model, you can easily start by modifying ```config.py```and using ```python train.py```.
* If you want to use trained weights, here is a checkpoint with only a simple train without tuning, but it's a little bit bigger (1.1gb, ResNet-101 as backbone). *Note the modification of the checkpoint file path in the demo code.*
* Refer to ```demo.py``` for the test.

If you want to use your **custom dataset**, please refer [here](https://github.com/gakkiri/Mask-Cascade-RCNN/blob/master/dataset/wgisd_dataset.py) to write your own code.

# Environment

```
torch>=1.4
torchvision>=0.4
opencv-python==4.1.2.30
albumentations
pillow
timm
tensorboardX
easydict
fvcore
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
