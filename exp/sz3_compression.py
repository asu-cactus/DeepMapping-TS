from pathlib import Path
import argparse
from math import ceil
from time import time
import numpy as np

from exp.utils.query_generator import generate_query
from exp.utils.data_utils import load_data, get_ts_length
from exp.sz3.pysz import (
    SZ,
)  # remember to include "third_party" folder in LD_LIBRARY_PATH


def sz_compress(args, df, error_bounds, partition_id):
    sz = SZ("exp/sz3/libSZ3c.so")
    for error_bound, col_name in zip(error_bounds, df.columns):
        column_data = df[col_name].to_numpy().astype(np.float32).reshape(-1)
        data_cmpr, cmpr_ratio = sz.compress(
            column_data,
            0,
            error_bound,
            0,
            0,
        )
        # folder = folder_name[table_name]
        save_dir = f"outputs/sz_{args.table_name}/{args.partition_size}/{col_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/{partition_id}.sz", "wb") as f:
            np.save(f, data_cmpr)


def compress(args):
    partitions, err_bound, _ = load_data(args.table_name, args.partition_size)
    for partition_id, partition in enumerate(partitions):
        sz_compress(args, partition, err_bound, partition_id)


def run_queries(args, query_range):
    n_cols = {"ethylene_CO": 16, "ethylene_methane": 16, "heavy_drinking": 1}
    time_elapsed = {
        "load_file": 0,
        "decompress": 0,
        "combine": 0,
    }
    sz = SZ("exp/sz3/libSZ3c.so")
    n_partitions = ceil(query_range / args.partition_size)
    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)
        first_partition = query[0] // args.partition_size
        last_partition = ceil(query[1] / args.partition_size)

        for col_id in range(n_cols[args.table_name]):
            data_dec_list = []
            for partition_id in range(first_partition, last_partition):
                start = time()
                with open(
                    f"outputs/sz_{args.table_name}/{args.partition_size}/{col_id}/{partition_id}.sz",
                    "rb",
                ) as f:
                    data_cmpr = np.load(f)
                time_elapsed["load_file"] += time() - start

                partition_size = (
                    args.partition_size
                    if partition_id < n_partitions - 1
                    else query_range % args.partition_size
                )

                data_dec = sz.decompress(
                    data_cmpr, (partition_size,), np.float32, time_elapsed
                )
                data_dec_list.append(data_dec)

            start = time()
            data_dec = np.concatenate(data_dec_list)
            base = first_partition * args.partition_size
            data_dec = data_dec[query[0] - base : query[1] + 1 - base]
            time_elapsed["combine"] += time() - start
    print(f"Time elapsed: {time_elapsed}")
    print(f"Total time: {sum(time_elapsed.values())}")


def run():
    parser = argparse.ArgumentParser(
        description="DeepMapping-TS baseline: sz compression"
    )
    parser.add_argument("--task", type=str, default="query")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")
    parser.add_argument("--partition_size", type=int, default=100000)
    parser.add_argument("--query_type", type=str, default="short")
    parser.add_argument("--query_size", type=int, default=1000)

    args = parser.parse_args()

    if args.task == "compress":
        compress(args)
    elif args.task == "query":
        query_range = get_ts_length(args.table_name)
        run_queries(args, query_range)
    else:
        raise ValueError(f"No task named {args.task}")
