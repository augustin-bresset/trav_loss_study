import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .spatial_context import TorchSparseSpatialContext

from .torchsparse_basic_blocks import (
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    ResidualBlock,
)

from .spatial_context import TorchSparseSpatialContext


class MinkUNetEncoder(nn.Module):
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

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                nn.init.xavier_normal_(m.kernel)
            elif isinstance(m, spnn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return {"out": x4, "hidden_states": [x0, x1, x2, x3, x4]}


class MinkUNetDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get("cr", 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get("run_up", True)

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

        self.classifier = nn.Sequential(
            nn.Linear(cs[8], kwargs["out_channels"]),
            nn.Softmax(dim=-1),
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(p=0.3, inplace=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                nn.init.xavier_normal_(m.kernel)
            elif isinstance(m, spnn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states):

        x0, x1, x2, x3, x4 = hidden_states

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


class MinkUNetWithAuxDecoder(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes,
        num_auxiliary_classes,
        use_spatial_context=False,
        spatial_context_out_channels=16,
        **kwargs
    ):
        super().__init__()

        self.use_spatial_context = use_spatial_context
        self.spatial_context_out_channels = spatial_context_out_channels
        if self.use_spatial_context:
            self.spatial_context = TorchSparseSpatialContext(
                in_channels=in_features,
                out_channels=self.spatial_context_out_channels,
                **kwargs
            )
            in_features = self.spatial_context_out_channels

        self.device = None

        self.encoder = MinkUNetEncoder(in_channels=in_features, **kwargs)

        self.main_decoder = MinkUNetDecoder(out_channels=num_classes, **kwargs)
        self.aux_decoder = MinkUNetDecoder(out_channels=num_auxiliary_classes, **kwargs)

    def to_(self, device):
        self.device = device
        self.encoder.to(device)
        self.main_decoder.to(device)
        self.aux_decoder.to(device)
        if self.use_spatial_context:
            self.spatial_context.to(device)

    def __repr__(self):
        return f"MinkUNetWithAuxDecoder_spatial_context_{self.use_spatial_context}"

    def forward(self, x, get_aux=True, get_main=True):

        if self.use_spatial_context:
            x["sparse_input"] = self.spatial_context(x["sparse_input"])

        encoded = self.encoder(x["sparse_input"])
        hidden_states = encoded["hidden_states"]

        if get_main:
            main_output = self.main_decoder(hidden_states)
            main_output["output"] = main_output["output"][x["sparse_input_invmap"]]
        if get_aux:
            aux_output = self.aux_decoder(hidden_states)
            aux_output["output"] = aux_output["output"][x["sparse_input_invmap"]]

        # encoded["out"] = encoded["out"][x["sparse_input_invmap"]]

        return {
            "main_output": main_output["output"] if get_main else None,
            "aux_output": aux_output["output"] if get_aux else None,
            "encoded": encoded["out"],
        }
