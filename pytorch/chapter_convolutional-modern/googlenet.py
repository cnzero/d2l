import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
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
        outputs_p1 = F.relu(self.path1(inputs))
        outputs_p2 = F.relu(self.path22(F.relu(self.path21(inputs))))
        outputs_p3 = F.relu(self.path32(F.relu(self.path31(inputs))))
        outputs_p4 = F.relu(self.path42(F.relu(self.path41(inputs))))

        return torch.cat((outputs_p1,
                          outputs_p2,
                          outputs_p3,
                          outputs_p4),
                         dim=1)


if __name__ == '__main__':
    print('Hello world in googlenet.py')

    b1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                 out_channels=64,
                                 kernel_size=(7, 7),
                                 stride=(2, 2),
                                 padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1))
    b2 = nn.Sequential(nn.Conv2d(in_channels=64,
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
    b3 = nn.Sequential(Inception(in_channels=192,
                                 out_channels=[64, [96, 128], [16, 32], 32]),
                                 # 64+128+32+32 = 256
                       Inception(in_channels=256,
                                 out_channels=[128, [128, 192], [32, 96], 64]),
                                 # 128+192+96+64 = 480
                       nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1))
    b4 = nn.Sequential(Inception(in_channels=480,
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

    b5 = nn.Sequential(Inception(in_channels=832,
                                 out_channels=[256, [160, 320], [32, 128], 128]),
                                 # 256+320+128+128 = 832
                       Inception(in_channels=832,
                                 out_channels=[384, [192, 384], [48, 128], 128]),
                                 # 384+384+128+128 = 1024
                       nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                       nn.Flatten())
    network = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(in_features=1024,
                                                          out_features=10))

    inputs = torch.rand(size=(1, 1, 96, 96))
    outputs = network(inputs)
    print('outputs shape: ', outputs.shape)