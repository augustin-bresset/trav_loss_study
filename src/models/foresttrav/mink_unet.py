import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import MinkowskiEngine as ME


class FTEContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=3, **kwargs):
        super(FTEContextBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension

        self.conv1 = ME.MinkowskiConvolution(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            dimension=self.dimension,
        )

        self.bn2 = ME.MinkowskiBatchNorm(self.out_channels)

        self.conv2 = ME.MinkowskiConvolution(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            dimension=self.dimension,
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.out_channels)

        self.conv3 = ME.MinkowskiConvolution(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            dimension=self.dimension,
            dilation=2,
        )

        self.relu = ME.MinkowskiReLU()

    def forward(self, x, coords=None):
        x = self.conv1(x, coordinates=coords)
        x = self.relu(x)

        x_ = self.conv2(x, coordinates=coords)
        x_ = self.relu(x)
        x_ = self.bn2(x)

        x_ = self.conv3(x, coordinates=coords)
        x_ = self.relu(x)
        x_ = self.bn3(x)

        return x + x_


class FTEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        use_gelu=True,
        dimension=3,
        **kwargs
    ):
        super(FTEncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_gelu = use_gelu
        self.dimension = dimension

        self.conv = ME.MinkowskiConvolution(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dimension=self.dimension,
        )

        self.norm = ME.MinkowskiBatchNorm(self.out_channels)

        self.relu = ME.MinkowskiReLU()
        if self.use_gelu:
            self.gelu = ME.MinkowskiGELU()

    def forward(self, x, coords=None):
        x = self.conv(x, coordinates=coords)
        x = self.norm(x)
        x = self.relu(x)

        return x


class FTDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        use_gelu=True,
        dimension=3,
        **kwargs
    ):
        super(FTDecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_gelu = use_gelu
        self.dimension = dimension

        self.convT = ME.MinkowskiConvolutionTranspose(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dimension=self.dimension,
        )

        self.norm = ME.MinkowskiBatchNorm(self.out_channels)

        self.relu = ME.MinkowskiReLU()
        if self.use_gelu:
            self.gelu = ME.MinkowskiGELU()

    def forward(self, x, coords=None):
        x = self.convT(x, coordinates=coords)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ForestTrav(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_layers,
        num_classes,
        use_gelu=True,
        base_channels=16,
        dimension=3,
        add_context=False,
        **kwargs
    ):
        super(ForestTrav, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = hidden_layers
        self.use_gelu = use_gelu
        self.base_channels = base_channels
        self.dimension = dimension
        self.add_context = add_context

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        if self.add_context:
            print("USING CONTEXT MODULE")
            self.context_out_channels = self.base_channels
            self.context1 = FTEContextBlock(
                self.in_channels, self.context_out_channels, dimension=dimension
            )
            self.context2 = FTEContextBlock(
                self.context_out_channels,
                self.context_out_channels,
                dimension=dimension,
            )
            self.context3 = FTEContextBlock(
                self.context_out_channels,
                self.context_out_channels,
                dimension=dimension,
            )
            self.encoder_in_channels = self.context_out_channels
        else:
            self.encoder_in_channels = self.in_channels

        self.encoder_in_channels = [self.encoder_in_channels] + [
            self.base_channels * (2**i) for i in range(self.hidden_layers - 1)
        ]
        self.encoder_out_channels = [
            self.base_channels * (2**i) for i in range(self.hidden_layers)
        ]

        self.decoder_in_channels = [self.encoder_out_channels[-1]]
        for i in range(self.hidden_layers - 2):
            self.decoder_in_channels.append(
                self.decoder_in_channels[-1] // 2 + self.encoder_out_channels[-i - 2]
            )
        self.decoder_in_channels.append(self.decoder_in_channels[-1] // 2)

        self.decoder_out_channels = [x // 2 for x in self.decoder_in_channels[:-1]] + [
            self.out_channels
        ]

        kernel_sizes = [1] + [1] * (self.hidden_layers - 1)
        strides = [1] + [1] * (self.hidden_layers - 1)

        for i in range(self.hidden_layers):
            self.encoders.append(
                FTEncoderBlock(
                    self.encoder_in_channels[i],
                    self.encoder_out_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    use_gelu=self.use_gelu,
                    dimension=self.dimension,
                )
            )

            self.decoders.append(
                FTDecoderBlock(
                    self.decoder_in_channels[i],
                    self.decoder_out_channels[i],
                    kernel_size=2,
                    stride=1,
                    use_gelu=self.use_gelu,
                    dimension=self.dimension,
                )
            )

        self.linear = ME.MinkowskiLinear(self.out_channels, num_classes)
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        # self.softmax = ME.MinkowskiSigmoid()

    def forward(self, x):
        x_enc = []
        # get coordinates from x
        # print(type(x.decomposed_coordinates[0]))
        if self.add_context:
            x = self.context1(x)
            x = self.context2(x)
            x = self.context3(x)

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < self.hidden_layers - 1:
                x_enc.append(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            if i < self.hidden_layers - 2:
                x = ME.cat(x, x_enc.pop())

        x = self.linear(x)
        x = self.softmax(x)

        return x
