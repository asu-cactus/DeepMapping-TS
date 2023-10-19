from exp.process_data import load_data
from exp.models.univariate import DilatedConvEncoder
import torch

import numpy as np
import pandas as pd
from exp.compress_utils import compress_main

import argparse
from itertools import chain
from pathlib import Path


# import torch.nn.functional as F
# from math import sqrt, exp


def compress_residuals_with_sz3(diff, err_bound, col_i):
    from exp.sz3.pysz import (
        SZ,
    )  # remember to include "third_party" folder in LD_LIBRARY_PATH

    sz = SZ("exp/sz3/libSZ3c.so")
    data_cmpr, cmpr_ratio = sz.compress(
        diff,
        0,
        err_bound,
        0,
        0,
    )
    with open(f"{col_i}.sz", "wb") as f:
        np.save(f, data_cmpr)


# def costom_loss(output, target, epoch, error_bound):
#     l1_loss = F.l1_loss(output, target, reduction="none")
#     sigma = 100 / sqrt(epoch)
#     # sigma = exp(-epoch / 1000) * 100
#     mean = error_bound + 3 * sigma
#     gaussian = torch.exp(-0.5 * ((l1_loss - mean) / sigma) ** 2) / sigma

#     return gaussian.mean()


def get_final_outputs(model, partitions, device, input_key="input"):
    outputs = [
        model(partition[input_key].to(device)).detach().to("cpu").squeeze()
        for partition in partitions
    ]
    outputs = torch.cat(outputs, dim=0)
    return outputs


def train_univariate_ts(partitions, args, is_causal, err_bound):
    device = torch.device(args.device)

    def train_step(model, partition, criterion, optimizer, epoch, error_bound):
        output = model(partition["input"].to(device)).to("cpu")
        # loss = criterion(output, partition["target"], epoch, err_bound)
        loss = criterion(output, partition["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item(), output

    epochs, lr, share_model = args.epochs, args.lr, args.share_model
    # criterion = costom_loss
    criterion = torch.nn.MSELoss()
    if share_model:
        Path(f"saved_models/{args.table_name}/{partitions['col']}").mkdir(
            parents=True, exist_ok=True
        )
        model = DilatedConvEncoder(args, is_causal=is_causal)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            for i, partition in enumerate(partitions):
                loss, _ = train_step(
                    model, partition, criterion, optimizer, epoch + 1, err_bound
                )
                print(f"Epoch {epoch:4d} Partition {i:2d} Loss {loss:.4f}")
        outputs = get_final_outputs(model, partitions, device)
        
    else:
        outputs = []
        for i, partition in enumerate(partitions):
            model = DilatedConvEncoder(args, is_causal=is_causal)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            for epoch in range(epochs):
                loss, output = train_step(
                    model, partition, criterion, optimizer, epoch + 1, err_bound
                )
                print(f"Partition {i:2d}  Epoch {epoch:4d} Loss {loss:.4f}")
            outputs.append(output.detach().to("cpu").squeeze())
        outputs = torch.cat(outputs, dim=0)
    return outputs


def combine_univariate_ts(partitions, args):
    device = torch.device(args.device)

    factor = 0.8
    combine_outputs_func = (
        lambda out_causal, out_noncausal: factor * out_causal
        + (1 - factor) * out_noncausal
    )

    def train_step(
        causal_model,
        non_causal_model,
        partition,
        criterion,
        optimizer,
    ):
        output_causal = causal_model(partition["self_input"].to(device)).to("cpu")
        output_non_causal = non_causal_model(partition["other_input"].to(device)).to(
            "cpu"
        )
        combined_output = combine_outputs_func(output_causal, output_non_causal)
        loss = criterion(combined_output, partition["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item(), combined_output

    epochs, lr, share_model = args.epochs, args.lr, args.share_model
    criterion = torch.nn.MSELoss()
    if share_model:
        causal_model = DilatedConvEncoder(args, is_causal=True)
        non_causal_model = DilatedConvEncoder(args, is_causal=False)
        causal_model.to(device)
        non_causal_model.to(device)
        # Optimizer
        optimizer = torch.optim.AdamW(
            chain(causal_model.parameters(), non_causal_model.parameters()), lr=lr
        )

        for epoch in range(epochs):
            for i, partition in enumerate(partitions):
                loss, _ = train_step(
                    causal_model,
                    non_causal_model,
                    partition,
                    criterion,
                    optimizer,
                )
                print(f"Epoch {epoch:4d} Partition {i:2d} Loss {loss:.4f}")
        outputs_causal = get_final_outputs(
            causal_model, partitions, device, "self_input"
        )
        outputs_non_causal = get_final_outputs(
            non_causal_model, partitions, device, "other_input"
        )
        return combine_outputs_func(outputs_causal, outputs_non_causal)

    else:
        outputs_combined = []
        for i, partition in enumerate(partitions):
            causal_model = DilatedConvEncoder(args, is_causal=True)
            non_causal_model = DilatedConvEncoder(args, is_causal=False)
            causal_model.to(device)
            non_causal_model.to(device)
            optimizer = torch.optim.AdamW(
                chain(causal_model.parameters(), non_causal_model.parameters()), lr=lr
            )
            for epoch in range(epochs):
                loss, combined_output = train_step(
                    causal_model, non_causal_model, partition, criterion, optimizer
                )
                print(f"Partition {i:2d}  Epoch {epoch:4d} Loss {loss:.4f}")
            outputs_combined.append(combined_output.detach().to("cpu").squeeze())
        outputs_combined = torch.cat(outputs_combined, dim=0)
        return outputs_combined


def model_univariate_ts(partitions, args, err_bounds, corr_dep):
    results = []
    for col_i, err_bound in enumerate(err_bounds):
        print(f"\nColumn {col_i}:")
        if args.mode == "from_self":
            partitions_dict = [
                {
                    "col": col_i,
                    "input": partition[col_i, :-1].unsqueeze(0),
                    "target": partition[col_i, 1:].unsqueeze(0),
                }
                for partition in partitions
            ]
            is_causal = True
            output = train_univariate_ts(partitions_dict, args, is_causal, err_bound)

        elif args.mode == "from_another":
            partitions_dict = [
                {
                    "input": partition[corr_dep[col_i]].unsqueeze(0),
                    "target": partition[col_i].unsqueeze(0),
                }
                for partition in partitions
            ]
            is_causal = False
            output = train_univariate_ts(partitions_dict, args, is_causal, err_bound)

        elif args.mode == "combine":
            partitions_dict = [
                {
                    "self_input": partition[col_i, :-1].unsqueeze(0),
                    "other_input": partition[corr_dep[col_i], 1:].unsqueeze(0),
                    "target": partition[col_i, 1:].unsqueeze(0),
                }
                for partition in partitions
            ]
            output = combine_univariate_ts(partitions_dict, args)
        else:
            raise ValueError(f"Unsupported Mode: {args.mode}")

        # Convert output and target to numpy arrays
        target = torch.cat(
            [partition["target"].squeeze() for partition in partitions_dict], dim=0
        )
        output, target = output.numpy(), target.numpy()

        # Compress
        compress_main(args, output, target, err_bound, col_i + 1)

        # Compute accuracy
        diff = output - target
        accuracy = np.sum(np.absolute(diff) < err_bound) / len(diff)
        print(f"Accuracy: {accuracy}\n{'=' * 50}\n")
        results.append({"accuracy": round(accuracy, 4)})
    pd.DataFrame(results).to_csv(
        f"outputs/{args.table_name}_{args.mode}_results.csv", index=False
    )


def run(compress=False):
    parser = argparse.ArgumentParser(description="DeepMapping-TS")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")
    parser.add_argument("--partition_size", type=int, default=4178505)
    # parser.add_argument("--partition_size", type=int, default=10000)
    parser.add_argument("--mode", type=str, default="from_another")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--share_model", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--m", type=int, default=8, help="Number of quantized bits")
    args = parser.parse_args()

    # Load data
    partitions, err_bounds, corr_dep = load_data(args.table_name, args.partition_size)
    print("Number of partitions:", len(partitions))

    # Convert partitions to tensors
    partitions = [
        torch.tensor(partition.to_numpy().transpose(), dtype=torch.float32)
        for partition in partitions
    ]
    model_univariate_ts(partitions, args, err_bounds, corr_dep)

    if compress:
        compress_residuals_with_sz3()