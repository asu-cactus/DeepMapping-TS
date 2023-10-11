from exp.process_data import load_data
from models.univariate import DilatedConvEncoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def model_univeriate_ts(partitions, epochs=1000, lr=3e-3, shared_model=True, n_init =3):
#     def train_step(model, partition, criterion, optimizer):
#         output = model(partition["input"].to(device)).to("cpu")
#         loss = criterion(output[:, n_init], partition["target"][:, n_init:])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         return loss.item()
#     criterion = torch.nn.MSELoss()
#     if shared_model:
#         model = DilatedConvEncoder(n_init=3)
#         model.to(device)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#         optimizer.step()
#         return loss.item()


def train_univariate_ts(partitions, epochs, lr, share_model, is_causal):
    def train_step(model, partition, criterion, optimizer):
        output = model(partition["input"].to(device)).to("cpu")
        loss = criterion(output, partition["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    criterion = torch.nn.MSELoss()
    if share_model:
        model = DilatedConvEncoder(is_causal=is_causal)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for i, partition in enumerate(partitions):
                loss = train_step(model, partition, criterion, optimizer)
                print(f"Epoch {epoch:4d} Partition {i:2d} Loss {loss:.4f}")

    else:
        for i, partition in enumerate(partitions):
            model = DilatedConvEncoder(is_causal=is_causal)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            for epoch in range(epochs):
                loss = train_step(model, partition, criterion, optimizer)
                print(f"Partition {i:2d}  Epoch {epoch:4d} Loss {loss:.4f}")


def model_univariate_ts(partitions, args):
    if args.mode == "from_self":
        partitions = [
            {
                "input": partition[2][:-1].unsqueeze(0),
                "target": partition[2][1:].unsqueeze(0),
            }
            for partition in partitions
        ]
        is_causal = True
    elif args.mode == "from_another":
        partitions = [
            {
                "input": partition[1].unsqueeze(0),
                "target": partition[2].unsqueeze(0),
            }
            for partition in partitions
        ]
        is_causal = False
    train_univariate_ts(partitions, args.epochs, args.lr, args.share_model, is_causal)


def run(args):
    # Load data

    partitions, err_bounds = load_data(args.table_name, partition_size=10000)
    print("Number of partitions:", len(partitions))

    # Convert partitions to tensors
    partitions = [
        torch.tensor(partition.to_numpy().transpose(), dtype=torch.float32)
        for partition in partitions
    ]
    model_univariate_ts(partitions, args)
    # receptive_field = (args.kernel_size - 1) * args.dilation + 1

    # match args.mode:
    #     case "univariate" | "combine":
    #         padding = receptive_field - args.n_init
    #         padding_func = torch.nn.ConstantPad1d((padding, 0), 0)
    #     case "from_another":
    #         padding = receptive_field // 2
    #         padding_func = torch.nn.ConstantPad1d(padding, 0)
    #     case _:
    #         raise ValueError(f"Unsupported Mode: {args.mode}")

    # partitions = [
    #     {
    #         "input": partition[1].unsqueeze(0),
    #         "target": partition[2].unsqueeze(0),
    #     }
    #     for partition in partitions
    # ]

    # # Train model
    # model_univariate_ts(partitions, shared_model=True, epochs=50)
