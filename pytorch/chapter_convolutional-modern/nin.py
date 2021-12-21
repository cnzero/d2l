import torch
from torch import nn


def nin_block(in_channels,
              out_channels,
              kernel_size,
              stride,
              padding):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                         nn.ReLU(),

                         nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=(1, 1)),
                         nn.ReLU(),

                         nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=(1, 1)),
                         nn.ReLU())


nin_network = nn.Sequential(nin_block(in_channels=1,
                                      out_channels=96,
                                      kernel_size=11,
                                      stride=4,
                                      padding=0),
                            nn.MaxPool2d(kernel_size=3,
                                         stride=2),

                            nin_block(in_channels=96,
                                      out_channels=256,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2),
                            nn.MaxPool2d(kernel_size=3, stride=2),

                            nin_block(in_channels=256,
                                      out_channels=384,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Dropout(0.5),

                            nin_block(in_channels=384,
                                      out_channels=10,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),

                            # global average pooling
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

                            nn.Flatten())


if __name__ == '__main__':
    print('Hello world in nin.py')

    inputs = torch.rand(size=(1, 1, 224, 224))
    for layer in nin_network:
        outputs = layer(inputs)
        print(layer.__class__.__name__, 'outputs shape: \t', outputs.shape)
        inputs = outputs
