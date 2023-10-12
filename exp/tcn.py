from exp.process_data import load_data
from models.univariate import DilatedConvEncoder
import torch
from itertools import chain
import numpy as np
import torch.nn.functional as F
from math import sqrt, exp


def costom_loss(output, target, epoch, error_bound):
    # l1_loss = F.l1_loss(output, target, reduction="none")
    # return torch.log(l1_loss).mean()
    l1_loss = F.l1_loss(output, target, reduction="none")
    # sigma = 100 / sqrt(epoch)
    # sigma = exp(-epoch / 1000) * 100
    # mean = error_bound + 3 * sigma
    mean = 500 * error_bound
    sigma = 150 * error_bound
    gaussian = torch.exp(-0.5 * ((l1_loss - mean) / sigma) ** 2) / sigma

    return gaussian.sum()


def train_univariate_ts(partitions, args, is_causal, err_bound):
    device = torch.device(args.device)

    def train_step(model, partition, criterion, optimizer, epoch, error_bound):
        output = model(partition["input"].to(device)).to("cpu")
        loss = criterion(output, partition["target"], epoch, err_bound)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), output

    epochs, lr, share_model = args.epochs, args.lr, args.share_model
    criterion = costom_loss
    # criterion = torch.nn.MSELoss()
    if share_model:
        model = DilatedConvEncoder(args, is_causal=is_causal)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            for i, partition in enumerate(partitions):
                loss, _ = train_step(
                    model, partition, criterion, optimizer, epoch + 1, err_bound
                )
                print(f"Epoch {epoch:4d} Partition {i:2d} Loss {loss:.4f}")
        outputs = [
            model(partition["input"].to(device)).detach().to("cpu").squeeze()
            for partition in partitions
        ]
        outputs = torch.cat(outputs, dim=0)
        return outputs

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

    def train_step(
        causal_model,
        non_causal_model,
        partition,
        criterion,
        optimizer,
    ):
        output1 = causal_model(partition["self_input"].to(device)).to("cpu")
        output2 = non_causal_model(partition["other_input"].to(device)).to("cpu")
        loss = criterion(output1 + output2, partition["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

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
                loss = train_step(
                    causal_model,
                    non_causal_model,
                    partition,
                    criterion,
                    optimizer,
                )
                print(f"Epoch {epoch:4d} Partition {i:2d} Loss {loss:.4f}")

    else:
        for i, partition in enumerate(partitions):
            causal_model = DilatedConvEncoder(args, is_causal=True)
            non_causal_model = DilatedConvEncoder(args, is_causal=False)
            causal_model.to(device)
            non_causal_model.to(device)
            optimizer = torch.optim.AdamW(
                chain(causal_model.parameters(), non_causal_model.parameters()), lr=lr
            )
            for epoch in range(epochs):
                loss = train_step(
                    causal_model, non_causal_model, partition, criterion, optimizer
                )
                print(f"Partition {i:2d}  Epoch {epoch:4d} Loss {loss:.4f}")


def model_univariate_ts(partitions, args, err_bounds):
    if args.mode == "from_self":
        partitions = [
            {
                "input": partition[3][:-1].unsqueeze(0),
                "target": partition[3][1:].unsqueeze(0),
            }
            for partition in partitions
        ]
        is_causal = True
        output = train_univariate_ts(partitions, args, is_causal, err_bounds[3])

    elif args.mode == "from_another":
        partitions = [
            {
                "input": partition[2].unsqueeze(0),
                "target": partition[3].unsqueeze(0),
            }
            for partition in partitions
        ]
        is_causal = False
        output = train_univariate_ts(partitions, args, is_causal, err_bounds[3])

    elif args.mode == "combine":
        partitions = [
            {
                "self_input": partition[3][:-1].unsqueeze(0),
                "other_input": partition[2][1:].unsqueeze(0),
                "target": partition[3][1:].unsqueeze(0),
            }
            for partition in partitions
        ]
        combine_univariate_ts(partitions, args)
    else:
        raise ValueError(f"Unsupported Mode: {args.mode}")

    target = torch.cat(
        [partition["target"].squeeze() for partition in partitions], dim=0
    )
    diff = output - target
    return diff.numpy()


def run(args, compress=True):
    # Load data

    partitions, err_bounds = load_data(args.table_name, args.partition_size)
    print("Number of partitions:", len(partitions))

    # Convert partitions to tensors
    partitions = [
        torch.tensor(partition.to_numpy().transpose(), dtype=torch.float32)
        for partition in partitions
    ]
    diff = model_univariate_ts(partitions, args, err_bounds)
    if compress:
        from exp.sz3.pysz import (
            SZ,
        )  # remember to include "third_party" folder in LD_LIBRARY_PATH

        sz = SZ("exp/sz3/libSZ3c.so")
        data_cmpr, cmpr_ratio = sz.compress(
            diff,
            0,
            err_bounds[0],
            0,
            0,
        )
        with open(f"3.sz", "wb") as f:
            np.save(f, data_cmpr)
