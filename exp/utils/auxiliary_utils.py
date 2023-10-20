import torch
from bitarray import bitarray
from pathlib import Path
from exp.utils.compress_utils import compress_unpredictable


def create_auxiliary_structure(
    args, output: torch.Tensor, target: torch.Tensor, err_bound: float, col_name: str
):
    # Save existence bitarray
    diff = output - target
    diff = torch.abs(diff) <= err_bound
    binary_repr = bitarray()
    binary_repr.pack(diff.numpy().tobytes())

    Path(f"outputs/{args.table_name}/aux").mkdir(parents=True, exist_ok=True)
    with open(f"outputs/{args.table_name}/aux/{col_name}.bin", "wb") as f:
        f.write(binary_repr)

    # Save the incorrect values
    incorrects = []
    for is_correct, value in zip(binary_repr, target):
        if not is_correct:
            incorrects.append(value)

    torch.save(
        torch.tensor(incorrects, dtype=torch.float32),
        f"outputs/{args.table_name}/aux/{col_name}.pt",
    )


def create_quantized_structure(
    args, output: torch.Tensor, target: torch.Tensor, err_bound: float, col_name: str
):
    pass


if __name__ == "__main__":
    a = torch.tensor([2, 2, 3], dtype=torch.float32)
    b = torch.tensor([1, 2, 3], dtype=torch.float32)
    create_auxiliary_structure(a, b, 0.1)
