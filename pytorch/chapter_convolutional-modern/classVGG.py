import torch
from torch import nn


class VggBlock(nn.Module):
    def __init__(self,
                 num_convs,
                 in_channels,
                 out_channels,
                 conv_kernel_size=(3, 3),
                 conv_padding=(1, 1),
                 maxpool_kernel_size=(2, 2),
                 maxpool_stride=(2, 2)):
        """

        Args:
            num_convs (int): how many convolution layers in each block
            in_channels (int): in-channel number
            out_channels (int): out-channel number
            conv_kernel_size (tuple): default (3, 3), which will net change the height and width
            conv_padding (tuple): default (1, 1), which will not change the height and width
            maxpool_kernel_size (tuple): kernel-size for nn.MaxPool
            maxpool_stride (tuple): stride for nn.MaxPool

        """
        super(VggBlock, self).__init__()

        block = []
        for _ in range(num_convs):
            block.append(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv_kernel_size,
                                   padding=conv_padding))
            block.append(nn.ReLU())
            in_channels = out_channels

        block.append(nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                                  stride=maxpool_stride))

        self.block = nn.Sequential(*block)

    def forward(self,
                inputs):
        return self.block(inputs)


class VggNetwork(nn.Module):
    def __init__(self,
                 conv_arch,
                 in_channels=1):
        """

        Args:
            conv_arch (tuple of tuple): ( (num_convs, in/out_channels) )
            in_channels (int): in-channel of the inputs
        """
        super(VggNetwork, self).__init__()

        blocks = []

        for num_convs, out_channels in conv_arch:
            vgg_block = VggBlock(num_convs=num_convs,
                                 in_channels=in_channels,
                                 out_channels=out_channels)
            blocks.append(vgg_block)
            in_channels = out_channels

        self.network = nn.Sequential(*blocks,
                                     # dense-1
                                     nn.Flatten(),
                                     nn.Linear(in_features=conv_arch[-1][-1]*7*7,
                                               out_features=4096),
                                     nn.ReLU(),
                                     nn.Dropout(),

                                     # dense-2
                                     nn.Linear(in_features=4096,
                                               out_features=4096),
                                     nn.ReLU(),
                                     nn.Dropout(),

                                     # dense-3
                                     nn.Linear(4096, 1000))

    def forward(self,
                inputs):
        return self.network(inputs)


if __name__ == '__main__':
    print('Hell world in classVGG.py')

    conv_arch = ((1, 64),
                 (1, 128),
                 (2, 256),
                 (2, 512),
                 (2, 512))
    vgg = VggNetwork(conv_arch=conv_arch)
    inputs = torch.rand(size=(2, 1, 224, 224))
    outputs = vgg(inputs=inputs)
    print('outputs shape: ', outputs.shape)

    # Attention: `VggNetwork` object is not iterable.
    # for layer in vgg:
    #     outputs = layer(inputs)
    #     print(layer.__class__.__name__, 'output shape: \t', outputs.shape)
    #     inputs = outputs