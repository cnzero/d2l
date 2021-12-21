import torch
from torch import nn


class NiNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(NiNBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.relu = nn.ReLU()

        self.conv2d_1x1_1 = nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, 1))
        self.conv2d_1x1_2 = nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, 1))

    def forward(self,
                inputs):
        outputs_conv2d = self.relu(self.conv2d(inputs))
        outputs_conv2d_1x1_1 = self.relu(self.conv2d_1x1_1(outputs_conv2d))
        outputs_conv2d_1x1_2 = self.relu(self.conv2d_1x1_2(outputs_conv2d_1x1_1))

        return outputs_conv2d_1x1_2


class NiNNetwork(nn.Module):
    def __init__(self,
                 nin_arch):
        """

        Args:
            nin_arch (list of int/tuple): [in_channels, out_channels, kernel_size, stride, padding]
        """
        super(NiNNetwork, self).__init__()

        self.nin_blocks = []
        for arch in nin_arch:
            [in_channels, out_channels, kernels_size, stride, padding] = arch
            self.nin_blocks.append(NiNBlock(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernels_size,
                                            stride=stride,
                                            padding=padding))
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.adaptive_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()

    def forward(self,
                inputs):
        for block in self.nin_blocks[:-1]:
            outputs_nin_block = block(inputs)
            outputs_maxpool2d = self.maxpool2d(outputs_nin_block)

            inputs = outputs_maxpool2d

        outputs_nin_final_block = self.nin_blocks[-1](inputs)
        outputs_adaptive_avgpool2d = self.adaptive_avgpool2d(outputs_nin_final_block)

        return self.flatten(outputs_adaptive_avgpool2d)


if __name__ == '__main__':
    print('Hello world in classNiN.py file')

    in_channels = 1
    num_classes = 10
    inputs = torch.rand(size=(1, in_channels, 224, 224))
    nin_arch = [[in_channels, 96, 11, 4, 0],
                [96,          256, 5, 1, 2],
                [256,         384, 3, 1, 1],
                [384,         10,  3, 1, 1]]

    nin_network = NiNNetwork(nin_arch=nin_arch)

    outputs = nin_network(inputs)
    print('outputs shape: ', outputs.shape)