import torch
from exp.models.univariate import DilatedConvEncoder
from exp.utils.data_utils import load_data
from exp.query_generator import generate_query
from bitarray import bitarray
import argparse
from time import time


def load_model(args, id):
    model = DilatedConvEncoder(
        args,
    )
    model.load_state_dict(
        torch.load(
            f"saved_models/{args.table_name}/{id}.pt",
        )
    )
    model.eval()

    return model


def get_partition_size(table_name):
    if table_name == "ethylene_CO":
        return 4208262
    elif table_name == "ethylene_methane":
        return 4178505
    else:
        raise ValueError(f"Unknown table name: {table_name}")


def inference_with_aux(
    model, input_ts, query, aux_data, existence_bitarray, elapsed_time
):
    # TODO: Consider no padding for input tensor
    start = time()
    model_output = model(input_ts[query[0] : query[1]].unsqueeze(0))
    elapsed_time["inference"] += time() - start

    start = time()
    start_idx = existence_bitarray.count(0, 0, query[0])
    length = existence_bitarray.count(0, query[0], query[1])
    selected_aux_data = (value for value in aux_data[start_idx : start_idx + length])
    selected_bits = existence_bitarray[query[0] : query[1]]
    aux_data = torch.tensor(
        [0.0 if bit else next(selected_aux_data) for bit in selected_bits],
        dtype=torch.float32,
    )
    elapsed_time["decode"] += time() - start

    start = time()
    final_output = model_output + aux_data
    elapsed_time["add"] += time() - start


def inference_with_quantized_code():
    pass


def run_queries(args, input_ts, id, query_range):
    model = load_model(args, id)
    aux_data = torch.load(f"outputs/{args.table_name}/aux/{id + 1}.pt")
    existence_bitarray = bitarray()
    with open(f"outputs/{args.table_name}/aux/{id + 1}.bin", "rb") as f:
        existence_bitarray.fromfile(f)

    time_elapsed = {"inference": 0, "decode": 0, "add": 0}
    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)
        inference_with_aux(
            model, input_ts, query, aux_data, existence_bitarray, time_elapsed
        )

    print(f"Time elapsed: {time_elapsed}")
    print(f"Total time: {sum(time_elapsed.values())}")


def run():
    parser = argparse.ArgumentParser(description="DeepMapping-TS inference")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")
    parser.add_argument("--query_type", type=str, default="medium")
    parser.add_argument("--query_size", type=int, default=10000)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--activation", type=str, default="identity")
    args = parser.parse_args()
    # Load data
    paritition_size = get_partition_size(args.table_name)
    partitions, err_bounds, corr_dep = load_data(args.table_name, paritition_size)
    print("Number of partitions:", len(partitions))

    # Convert partitions to tensors
    data = torch.tensor(partitions[0].to_numpy().transpose(), dtype=torch.float32)

    for ts_id in range(len(err_bounds)):
        input_ts = data[corr_dep[ts_id]]
        run_queries(args, input_ts, ts_id, paritition_size)
