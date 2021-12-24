# !/Users/robot1/anaconda3/envs/d2l/bin/python
# -*- coding: utf-8 -*-
# @time: 2021-12-24
# @author: cnzero

import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        """

        Args:
            in_channels (int): in_channels of the Inception block
            out_channels (list of list/int): list length is 4,
                                             which is number of parallel paths in each Inception block
        """
        super(Inception, self).__init__()

        # path1: 1x1 Conv
        self.path1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels[0],
                               kernel_size=(1, 1))

        # path2: 1x1 Conv -> 3x3 Conv pad 1
        self.path21 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels[1][0],
                                kernel_size=(1, 1))
        self.path22 = nn.Conv2d(in_channels=out_channels[1][0],
                                out_channels=out_channels[1][1],
                                kernel_size=(3, 3),
                                padding=(1, 1))

        # path3: 1x1 Conv -> 5x5 Conv, pad 2
        self.path31 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels[2][0],
                                kernel_size=(1, 1))
        self.path32 = nn.Conv2d(in_channels=out_channels[2][0],
                                out_channels=out_channels[2][1],
                                kernel_size=(5, 5),
                                padding=(2, 2))

        # path4: 3x3 MaxPool -> 1x1 Conv
        self.path41 = nn.MaxPool2d(kernel_size=(3, 3),
                                   stride=1,
                                   padding=1)
        self.path42 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels[3],
                                kernel_size=(1, 1))

    def forward(self,
                inputs):
        """forward inference of each Inception block

        Args:
            inputs (torch.tensor): inputs, whose shape (B, num_in_channels, H, W)

        Returns:
            torch.tensor: inference outputs, whose shape (B, num_out_channels, H', W')

        """
        outputs_p1 = F.relu(self.path1(inputs))
        outputs_p2 = F.relu(self.path22(F.relu(self.path21(inputs))))
        outputs_p3 = F.relu(self.path32(F.relu(self.path31(inputs))))
        outputs_p4 = F.relu(self.path42(F.relu(self.path41(inputs))))

        return torch.cat((outputs_p1,
                          outputs_p2,
                          outputs_p3,
                          outputs_p4),
                         dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.block1_conv7x7_maxpool321 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                                 out_channels=64,
                                                                 kernel_size=(7, 7),
                                                                 stride=(2, 2),
                                                                 padding=3),
                                                       nn.ReLU(),
                                                       nn.MaxPool2d(kernel_size=3,
                                                                    stride=2,
                                                                    padding=1))
        self.block2_conv1x1_conv3x3_maxpool321 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                                         out_channels=64,
                                                                         kernel_size=(1, 1)),
                                                               nn.ReLU(),
                                                               nn.Conv2d(in_channels=64,
                                                                         out_channels=192,
                                                                         kernel_size=(3, 3),
                                                                         padding=1),
                                                               nn.ReLU(),
                                                               nn.MaxPool2d(kernel_size=3,
                                                                            stride=2,
                                                                            padding=1))
        self.block3_Inceptionx2_maxpool321 = nn.Sequential(Inception(in_channels=192,
                                                                     out_channels=[64, [96, 128], [16, 32], 32]),
                                                                     # 64+128+32+32 = 256
                                                           Inception(in_channels=256,
                                                                     out_channels=[128, [128, 192], [32, 96], 64]),
                                                                     # 128+192+96+64 = 480
                                                           nn.MaxPool2d(kernel_size=3,
                                                                        stride=2,
                                                                        padding=1))
        self.block4_Inceptionx5_maxpool321 = nn.Sequential(Inception(in_channels=480,
                                                                     out_channels=[192, [96, 208], [16, 48], 64]),
                                                                     # 192+208+48+64 = 512
                                                           Inception(in_channels=512,
                                                                     out_channels=[160, [112, 224], [24, 64], 64]),
                                                                     # 160+224+64+64 = 512
                                                           Inception(in_channels=512,
                                                                     out_channels=[128, [128, 256], [24, 64], 64]),
                                                                     # 128+256+64+64 = 512
                                                           Inception(in_channels=512,
                                                                     out_channels=[112, [144, 288], [32, 64], 64]),
                                                                     # 112+288+64+64 = 528
                                                           Inception(in_channels=528,
                                                                     out_channels=[256, [160, 320], [32, 128], 128]),
                                                                     # 256+320+128+128 = 832
                                                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block5_Inceptionx2_avgpool = nn.Sequential(Inception(in_channels=832,
                                                                  out_channels=[256, [160, 320], [32, 128], 128]),
                                                                  # 256+320+128+128 = 832
                                                        Inception(in_channels=832,
                                                                  out_channels=[384, [192, 384], [48, 128], 128]),
                                                                  # 384+384+128+128 = 1024
                                                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                                        nn.Flatten())
        self.block6_dense = nn.Sequential(nn.Flatten(),
                                          nn.Linear(in_features=1024,
                                                    out_features=10))

    def forward(self,
                inputs):
        """blocks in GoogLeNet forward inference

        Args:
            inputs (torch.tensor): inputs, shape (B, num_in_channels, H, W)

        Returns:
            torch.tensor: outputs, shape(B, num_classes)

        """
        outputs1 = self.block1_conv7x7_maxpool321(inputs)
        outputs2 = self.block2_conv1x1_conv3x3_maxpool321(outputs1)
        outputs3 = self.block3_Inceptionx2_maxpool321(outputs2)
        outputs4 = self.block4_Inceptionx5_maxpool321(outputs3)
        outputs5 = self.block5_Inceptionx2_avgpool(outputs4)
        outputs6 = self.block6_dense(outputs5)

        return outputs6


if __name__ == '__main__':
    print('Hello world in classGoogLeNet.py')

    network = GoogLeNet()
    batch_size = 3
    in_channels = 1
    num_classes = 10
    inputs = torch.rand(size=(batch_size, in_channels, 96, 96))
    outputs = network(inputs)
    print('outputs shape: ', outputs.shape)