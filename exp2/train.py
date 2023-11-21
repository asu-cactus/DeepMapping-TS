from exp2.utils.model import WindowMLP
from exp2.utils.data_utils import (
    load_data,
    normalize_data,
    preprocess,
    postprocess,
    ts_distortion,
    denormalize_data,
)

from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TSDataSet(Dataset):
    def __init__(self, processed_data: np.array, window_size: int, input_col: int):
        assert input_col < processed_data.shape[1]
        assert processed_data.shape[0] % window_size == 0
        self.input_col = input_col
        self.window_size = window_size

        self.processed_data = processed_data
        # # Truncate df
        # self.length = processed_data.shape[0] // self.window_size
        # self.truncated_data = processed_data[: self.length * self.window_size]
        # self.ts_length = self.truncated_data.shape[0]
        # print(f"Truncated time series length: {self.ts_length}")

    def __len__(self):
        return self.processed_data.shape[0] // self.window_size

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = (idx + 1) * self.window_size
        values = self.processed_data[start_idx:end_idx]
        input_ts = values[:, self.input_col].reshape(-1)
        target_ts = np.delete(values, self.input_col, axis=1).reshape(-1)

        return input_ts, target_ts


def get_preprocessed_data(args):
    df, err_bounds = load_data(args)
    df, param1, param2 = normalize_data(args, df)
    reduced_ndarray = preprocess(args, df)
    origin_ndarray = df.values
    return origin_ndarray, reduced_ndarray, param1.values, param2.values


def print_model_size(model):
    param_size = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    buffer_size = sum(
        [buffer.nelement() * buffer.element_size() for buffer in model.parameters()]
    )
    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.2f}MB".format(size_all_mb))


def train(args):
    device = torch.device(args.device)

    # Initialize model
    network_window_size = args.window_size // args.reduce_factor
    print(f"Network window size: {network_window_size}")
    model = WindowMLP(args, network_window_size)
    print_model_size(model)
    model.to(device)
    model.train()

    # Initialize optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # Initialize data loader
    origin_ndarray, reduced_ndarray, param1, param2 = get_preprocessed_data(args)

    dataset = TSDataSet(reduced_ndarray, network_window_size, args.input_col)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    for epoch in range(args.epochs):
        losses = []
        for _, (input_ts, target_ts) in enumerate(dataloader):
            outputs = model(input_ts.to(device)).to("cpu")
            loss = criterion(outputs, target_ts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:4d} Loss {sum(losses) / len(losses):.4f}")

    # Save model
    Path(f"saved_models/").mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"saved_models/model_{args.process_mode}.pt",
    )
    return model, dataloader, origin_ndarray, reduced_ndarray, param1, param2


def eval(
    args, model, dataloader, origin_ndarray, reduced_ndarray, norm_param1, norm_param2
):
    norm_param1 = np.delete(norm_param1, args.input_col)
    norm_param2 = np.delete(norm_param2, args.input_col)
    origin_ndarray = np.delete(origin_ndarray, args.input_col, axis=1)
    reduced_ndarray = np.delete(reduced_ndarray, args.input_col, axis=1)

    model.eval()
    collected_outputs = []
    for _, (input_ts, target_ts) in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(input_ts.to(args.device)).detach().to("cpu").numpy()
        collected_outputs.append(outputs.reshape(-1, reduced_ndarray.shape[1]))
    outputs = np.concatenate(collected_outputs, axis=0)

    denormalized_outputs = denormalize_data(args, outputs, norm_param1, norm_param2)
    # Compute distortion
    for i in range(reduced_ndarray.shape[1]):
        # Compute distortion of reduced column
        dist_reduced = ts_distortion(outputs[:, i], reduced_ndarray[:, i])
        print(
            f"Distortion of reduced column {i + (i >= args.input_col)}: {dist_reduced}"
        )

        # Compute distortion of original column
        denorm_expand_col = denormalized_outputs[:, i].repeat(args.reduce_factor)
        for mode in ["mse", "rmse", "mae"]:
            dist_origin = ts_distortion(denorm_expand_col, origin_ndarray[:, i], mode)
            print(
                f"{mode} mode distortion of original column {i + (i >= args.input_col)}: {dist_origin}"
            )
        print("*" * 50)


def run():
    parser = argparse.ArgumentParser(description="DeepMapping-TS train mlp model")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")

    parser.add_argument("--input_col", type=int, default=0)
    parser.add_argument("--process_mode", type=str, default="average")
    parser.add_argument("--norm_mode", type=str, default="minmax")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--reduce_factor", type=int, default=10)

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:1")
    # Quantization parameters
    parser.add_argument("--m", type=int, default=8, help="Number of quantized bits")
    args = parser.parse_args()

    args.heads = 7  # hard coded for now
    eval(args, *train(args))
