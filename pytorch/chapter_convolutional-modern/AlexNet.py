# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-12-08
# @author: cnzero

import torch
from torch import nn

AlexNet = nn.Sequential(nn.Conv2d(in_channels=1,
                                  out_channels=96,
                                  kernel_size=(11, 11),
                                  stride=(4, 4),
                                  padding=(1, 1)),
                        nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

                        nn.Conv2d(in_channels=96,
                                  out_channels=256,
                                  kernel_size=(5, 5),
                                  padding=(2, 2)),
                        nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

                        nn.Conv2d(in_channels=256,
                                  out_channels=384,
                                  kernel_size=(3, 3),
                                  padding=(1, 1)),
                        nn.ReLU(),

                        nn.Conv2d(in_channels=384,
                                  out_channels=384,
                                  kernel_size=(3, 3),
                                  padding=(1, 1)),
                        nn.ReLU(),

                        nn.Conv2d(in_channels=384,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  padding=(1, 1)),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                        nn.Flatten(),

                        nn.Linear(in_features=6400,
                                  out_features=4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),

                        nn.Linear(in_features=4096,
                                  out_features=4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),

                        nn.Linear(in_features=4096,
                                  out_features=1000))

if __name__ == '__main__':
    print('Hello world.')

    batch_size = 10
    inputs = torch.rand(size=(batch_size, 1, 224, 224))
    output = AlexNet(inputs)
    print('output shape: ', output.shape)

    for layer in AlexNet:
        outputs = layer(inputs)
        inputs = outputs
        print(layer, '\t\t\noutput shape: \t', outputs.shape, '\n')