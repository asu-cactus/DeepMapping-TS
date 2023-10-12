import importlib
import argparse


def main():
    parser = argparse.ArgumentParser(description="DeepMapping-TS")
    parser.add_argument("--table_name", type=str, default="ethylene_methane")
    parser.add_argument("--partition_size", type=int, default=4178505)
    # parser.add_argument("--partition_size", type=int, default=10000)
    parser.add_argument("--mode", type=str, default="from_another")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--share_model", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:1")

    args = parser.parse_args()

    run_func = getattr(importlib.import_module("exp.tcn"), "run")
    run_func(args)


if __name__ == "__main__":
    main()
