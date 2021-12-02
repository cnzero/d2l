# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-12-01
# @author: cnzero

"""LeNet model and its paradigm
"""

import torch
from torch import nn

lenet = nn.Sequential(nn.Conv2d(in_channels=1,
                                out_channels=6,
                                kernel_size=(5, 5),
                                padding=2),
                      nn.Sigmoid(),
                      nn.AvgPool2d(kernel_size=2,
                                   stride=2),
                      nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=(5, 5)),
                      nn.Sigmoid(),
                      nn.AvgPool2d(kernel_size=2,
                                   stride=2),
                      nn.Flatten(),
                      nn.Linear(in_features=16*5*5,
                                out_features=120),
                      nn.Sigmoid(),
                      nn.Linear(in_features=120,
                                out_features=84),
                      nn.Sigmoid(),
                      nn.Linear(84, 10))

batch_size = 100
num_channels = 1
H, W = 28, 28

inputs = torch.rand(size=(batch_size, num_channels, H, W))

for layer in lenet:
    inputs = layer(inputs)
    print(layer.__class__.__name__, 'output shape: \t', inputs.shape)
