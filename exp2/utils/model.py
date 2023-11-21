from torch import nn
import torch


class MLPBlock(nn.Module):
    def __init__(self, hidden_size, bias, activation=nn.ReLU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=bias),
            activation,
            nn.Linear(hidden_size, hidden_size, bias=bias),
            activation,
        )

    def forward(self, x):
        return x + self.block(x)


class WindowMLP(nn.Module):
    def __init__(self, args, network_window_size) -> None:
        super().__init__()
        # Shared encoder layers
        hidden_size = 100
        self.encoder = nn.Sequential(
            nn.Linear(network_window_size, hidden_size, bias=False),
            MLPBlock(hidden_size, bias=False),
            MLPBlock(hidden_size, bias=False),
        )

        # Decoders
        decoder = nn.Sequential(
            MLPBlock(hidden_size, bias=True),
            MLPBlock(hidden_size, bias=True),
            MLPBlock(hidden_size, bias=True),
            MLPBlock(hidden_size, bias=True),
            MLPBlock(hidden_size, bias=True),
            MLPBlock(hidden_size, bias=True),
            nn.Linear(hidden_size, network_window_size, bias=True),
        )
        if args.norm_mode == "minmax":
            decoder.append(nn.Sigmoid())
        self.decoders = nn.ModuleList([decoder for _ in range(args.heads)])

    def forward(self, x):
        x = self.encoder(x)
        predictions = [decoder(x) for decoder in self.decoders]
        return torch.cat(predictions, axis=1)

    def predict(self, x, head_indices=None):
        x = self.encoder(x)
        predictions = [self.decoders[i](x) for i in head_indices]
        return predictions
