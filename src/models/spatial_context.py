import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .torchsparse_basic_blocks import (
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    ResidualBlock,
)

__all__ = ["TorchSparseSpatialContext"]


class TorchSparseSpatialContext(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=3, **kwargs):
        super(TorchSparseSpatialContext, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.out_features = out_channels

        self.conv1 = spnn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )

        self.relu = spnn.ReLU(True)

        self.conv2 = BasicConvolutionBlock(
            self.out_channels,
            self.out_channels,
            ks=1,
            stride=1,
        )

        self.conv3 = BasicConvolutionBlock(
            self.out_channels,
            self.out_channels,
            ks=3,
            stride=1,
            dilation=2,
        )

    def forward(self, x):
        # print("input", x.C.device)
        # exit()

        x = self.relu(self.conv1(x))
        x_ = self.conv2(x)
        x_ = self.conv3(x)

        return x + x_
