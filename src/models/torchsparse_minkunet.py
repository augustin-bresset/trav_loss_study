"""
Original Script : https://github.com/valeoai/SALUDA/blob/main/networks/backbone/torchsparse_minkunet.py

Modified by : Abhay Dayal Mathur, U2IS, Ensta Paris. (c) 2024
"""

import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch

from .torchsparse_basic_blocks import (
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    ResidualBlock,
)
from .spatial_context import TorchSparseSpatialContext

__all__ = ["TorchSparseMinkUNet"]


class MinkUNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get("cr", 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get("run_up", True)

        in_channels = kwargs["in_channels"]

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs["out_channels"]))

        self.point_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cs[0], cs[4]),
                    nn.BatchNorm1d(cs[4]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(cs[4], cs[6]),
                    nn.BatchNorm1d(cs[6]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(cs[6], cs[8]),
                    nn.BatchNorm1d(cs[8]),
                    nn.ReLU(True),
                ),
            ]
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return {
            "encoded": x4,
            "output": out,
        }


class TorchSparseMinkUNet(MinkUNet):

    def __init__(
        self,
        in_channels,
        out_channels,
        voxel_size=1,
        cylindrical_coordinates=False,
        cr=1.0,
        use_spatial_context=False,
        use_auxiliary_decoder=False,
        spatial_out_channels=96,
        **kwargs,
    ):

        super(TorchSparseMinkUNet, self).__init__(
            in_channels=(
                in_channels if not use_spatial_context else spatial_out_channels
            ),
            out_channels=out_channels,
            cr=cr,
        )  ####"num classes is here used, alt

        if use_spatial_context:
            self.spatial_context = TorchSparseSpatialContext(
                in_channels=in_channels, out_channels=spatial_out_channels, dimension=3
            )
            in_channels = self.spatial_context.out_channels
        else:
            self.spatial_context = None

        self.voxel_size = voxel_size
        self.cylindrical_coordinates = cylindrical_coordinates

    def get_stack_item_list(self):
        return []

    def get_cat_item_list(self):
        return []

    def forward(self, data):

        # forward in the network
        if self.spatial_context is not None:
            data["sparse_input"] = self.spatial_context(data["sparse_input"])
            # Need to confirm if slicing with invmap is required here as well TODO @abhaydmathur

        outputs = super().forward(data["sparse_input"])

        # interpolate the outputs
        try:
            data["sparse_input_invmap"] = data["sparse_input_invmap"].to(
                outputs["output"].device,
                dtype=torch.long,
            )
            outputs = outputs[data["sparse_input_invmap"]]
        except Exception as e:
            print(f"Error in slicing the outputs with invmap {e}")
            outputs = outputs

        return outputs

    @staticmethod
    def get_final_layer_name():
        return "classifier"

    @staticmethod
    def get_linear_layer(in_channels, out_channels):
        return nn.Conv1d(in_channels, out_channels, 1)
