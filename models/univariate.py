from torch import nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation,
        padding,
        use_weight_norm=False,
        is_causal=False,
    ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        if is_causal:
            padding = self.receptive_field - 1
            self.padding = nn.ConstantPad1d((padding, 0), 0)
        else:
            padding = self.receptive_field // 2
            self.padding = nn.ConstantPad1d(padding, 0)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

        if use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.activation = activation

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.activation(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        activation=nn.GELU(),
        layers=2,
        is_causal=False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            *[
                ConvLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    activation,
                    is_causal,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        length = x.size(1)
        x = x + self.block(x)
        x = x[:, :length]
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] = [1],
        kernel_size: int = 3,
        activation=nn.GELU(),
        layers: int = 2,
        is_causal: bool = False,
    ):
        super().__init__()
        if not is_causal:
            assert kernel_size % 2 == 1
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=channels[i - 1] if i > 0 else in_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    activation=activation,
                    layers=layers,
                    is_causal=is_causal,
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)
