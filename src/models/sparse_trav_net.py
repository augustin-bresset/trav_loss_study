"""Simple TorchSparse UNet for per-voxel binary traversability classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .torchsparse_basic_blocks import (
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    ResidualBlock,
)


class SparseTravNet(nn.Module):
    """3-level UNet with TorchSparse sparse convolutions.

    Outputs one logit per voxel (binary traversability).
    Input features: [x, y, z, intensity] → in_channels=4.
    """

    def __init__(self, in_channels: int = 4, cr: float = 1.0) -> None:
        super().__init__()
        cs = [int(cr * c) for c in [32, 64, 128, 64, 32]]

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1),
        )

        # cs[3] = 64, cs[2] = 128 → cat → 192 in
        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1),
            ),
        ])

        # cs[4] = 32, cs[3] = 64 → cat → 96 in
        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1),
            ),
        ])

        self.head = nn.Linear(cs[4], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torchsparse.SparseTensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)

        y1 = self.up1[0](x2)
        y1 = torchsparse.cat([y1, x1])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x0])
        y2 = self.up2[1](y2)

        return self.head(y2.F).squeeze(-1)  # (N_vox,)
