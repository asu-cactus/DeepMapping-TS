import torch
from bitarray import bitarray

from pathlib import Path


def create_auxiliary_structure(
    args, output: torch.Tensor, target: torch.Tensor, err_bound: float, col_name: str
):
    # Save existence bitarray
    diff = output - target
    diff = torch.abs(diff) <= err_bound
    binary_repr = bitarray()
    binary_repr.pack(diff.numpy().tobytes())

    save_dir = f"outputs/{args.table_name}/aux/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/{col_name}.bin", "wb") as f:
        f.write(binary_repr)

    # Save the incorrect values
    aux_partition_sizes = [int(size) for size in args.aux_partition_size.split(",")]
    for aux_partition_size in aux_partition_sizes:
        save_dir = f"outputs/{args.table_name}/aux/{aux_partition_size}"
        incorrects = []
        for is_correct, value in zip(binary_repr, target):
            if not is_correct:
                incorrects.append(value)

        if aux_partition_size == 0:
            torch.save(
                torch.tensor(incorrects, dtype=torch.float32),
                f"{save_dir}/{col_name}.pt",
            )
        else:
            Path(f"{save_dir}/{col_name}").mkdir(parents=True, exist_ok=True)
            size = len(incorrects)
            for idx in range(0, size, aux_partition_size):
                partition = torch.tensor(
                    incorrects[idx : idx + aux_partition_size], dtype=torch.float32
                )
                idx_ = idx // aux_partition_size
                torch.save(partition, f"{save_dir}/{col_name}/{idx_}.pt")


if __name__ == "__main__":
    a = torch.tensor([2, 2, 3], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    create_auxiliary_structure(a, b, 0.1)
