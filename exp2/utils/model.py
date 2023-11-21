from torch import nn
import torch


class WindowMLP(nn.Module):
    def __init__(self, args, network_window_size) -> None:
        super().__init__()
        # Shared encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(network_window_size, 50, bias=False),
            nn.ReLU(),
            nn.Linear(50, 50, bias=False),
            nn.ReLU(),
            nn.Linear(50, 50, bias=False),
            nn.ReLU(),
        )

        # Decoders
        decoder = nn.Sequential(
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, network_window_size, bias=True),
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
