from torch import nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        args,
        in_channels,
        out_channels,
        dilation,
        is_causal,
        use_weight_norm=False,
    ):
        super().__init__()

        self.receptive_field = (args.kernel_size - 1) * dilation + 1

        if is_causal:
            padding = self.receptive_field - 1
            self.padding = nn.ConstantPad1d((padding, 0), 0)
        else:
            padding = self.receptive_field // 2
            self.padding = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            args.kernel_size,
            dilation=dilation,
            bias=False,
        )

        if use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

        if args.activation == "relu":
            activation = nn.ReLU()
        elif args.activation == "gelu":
            activation = nn.GELU()
        elif args.activation == "tanh":
            activation = nn.Tanh()
        elif args.activation == "identity":
            activation = nn.Identity()
        else:
            raise ValueError("Activation not supported")
        self.activation = activation

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

    # def predict(self, x):
    #     return self.activation(self.conv(x))


class ConvBlock(nn.Module):
    def __init__(
        self,
        args,
        in_channels=1,
        out_channels=1,
        dilation=1,
        is_causal=False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            *[
                ConvLayer(
                    args,
                    in_channels,
                    out_channels,
                    dilation,
                    is_causal,
                )
                for _ in range(args.layers)
            ]
        )

    def forward(self, x):
        return self.block(x)
        # x = x + self.block(x)
        # return x


class DilatedConvEncoder(nn.Module):
    def __init__(
        self,
        args,
        is_causal: bool = False,
    ):
        super().__init__()

        n_blocks = args.n_blocks
        if not is_causal:
            assert args.kernel_size % 2 == 1

        self.net = nn.Sequential(
            *[
                ConvBlock(
                    args,
                    in_channels=1,
                    out_channels=1,
                    dilation=2**i,
                    is_causal=is_causal,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        return self.net(x)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_bytes = param_size + buffer_size
    print(f"model size: {size_all_bytes} bytes")
