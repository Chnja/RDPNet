# RDP-Net: Region Detail Preserving Network for Change Detection

![author](https://img.shields.io/badge/Author-Chnja-blue.svg)
![author](https://img.shields.io/badge/Frame-pytorch-important.svg)
![license](https://img.shields.io/badge/License-GPLv3-green.svg)

![RDP-Net](img/RDPNet.png)

The pytorch implementation for RDP-Net.
<!-- The paper is avaliable on arXiV. -->

```python
from RDPNet import RDPNet
net = RDPNet(in_ch=3, out_ch=2).to(device)
```

## Introduction

Change detection (CD) is an essential earth observation technique. It captures the dynamic information of land objects. With the rise of deep learning, neural networks (NN) have shown great potential in CD. However, current NN models introduce backbone architectures that lose the detail information during learning. Moreover, current NN models are heavy in parameters, which prevents their deployment on edge devices such as drones.  In this work, we tackle this issue by proposing RDP-Net: a region detail preserving network for CD. We propose an efficient training strategy that quantifies the importance of individual samples during the warmup period of NN training. Then, we perform non-uniform sampling based on the importance score so that the NN could learn detail information from easy to hard. Next, we propose an effective edge loss that improves the network's attention on details such as boundaries and small regions. As a result, we provide a NN model that achieves the state-of-the-art empirical performance in CD with only 1.70M parameters. We hope our RDP-Net would benefit the practical CD applications on compact devices and could inspire more people to bring change detection to a new level with the efficient training strategy.

## Dataset

### CDD

Paper: [Change Detection in Remote Sensing Images using Conditional Adversarial Networks](https://pdfs.semanticscholar.org/ae15/e5ccccaaff44ab542003386349ef1d3b7511.pdf)

Link: is not avaliable now, but you can contact the author (MLebedev@gosniias.ru) to get this Dataset.

### LEVIR-CD

Paper: [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)

[Link](https://justchenhao.github.io/LEVIR/)

<!-- ## Citation

If you find this work valuable or use our code in your own research, please consider citing us with the following bibtex:

```
``` -->

## Comparison Methods

* FC-Siam-diff: [Fully Convolutional Siamese
Networks for Change Detection](https://ieeexplore.ieee.org/abstract/document/8451652)
* UNet++_MSOF: [End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++](https://www.mdpi.com/2072-4292/11/11/1382)
* DASNet: [DASNet: Dual Attentive Fully Convolutional Siamese Networks for Change Detection in High-Resolution Satellite Images](https://ieeexplore.ieee.org/abstract/document/9259045)
* SNUNet-CD: [SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images](https://ieeexplore.ieee.org/abstract/document/9355573)

## Contact

Hongjia Chen: chj1997@whu.edu.cn