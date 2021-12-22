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
                          outputs_p3),
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
                       Inception(in_channels=256,
                                 out_channels=[128, [128, 192], [32, 96], 64]),
                       nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1))
    b4 = nn.Sequential(Inception(in_channels=480))