# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-12-14
# @author: cnzero

import torch
from torch import nn


def vgg_block(num_convs,
              in_channels,
              out_channels):
    layers = []

    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                padding=(1, 1)))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2)))

    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1

    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blocks,
                         nn.Flatten(),
                         nn.Linear(in_features=conv_arch[-1][-1]*7*7,
                                   out_features=4096),
                         nn.ReLU(),
                         nn.Dropout(),

                         nn.Linear(in_features=4096,
                                   out_features=4096),
                         nn.ReLU(),
                         nn.Dropout(),

                         nn.Linear(4096, 1000))


if __name__ == '__main__':
    print('Hello world in VGG network.')

    conv_arch = ((1, 64),
                 (1, 128),
                 (2, 256),
                 (2, 512),
                 (2, 512))
    net = vgg(conv_arch=conv_arch)
    inputs = torch.rand(size=(2, 1, 224, 224))

    for block in net:
        outputs = block(inputs)
        print(block.__class__.__name__, 'output shape: \t', outputs.shape)
        inputs = outputs