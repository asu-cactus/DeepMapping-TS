import torch
from scipy.sparse import load_npz
from exp.models.univariate import DilatedConvEncoder
from exp.utils.data_utils import load_data
from exp.query_generator import generate_query
from exp.utils.compress_utils import decode_quantized_values
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


def inference_with_bitarray(
    model, input_ts, query, existence_bitarray, aux_data, elapsed_time, **kwargs
):
    # TODO: Consider no padding for input tensor
    start = time()
    model_output = model(input_ts[query[0] : query[1]].unsqueeze(0)).squeeze()
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


def inference_with_quantized_code(
    model,
    input_ts,
    query,
    quantized_code,
    unpredictables,
    elapsed_time,
    err_bound=None,
):
    # TODO: Consider no padding for input tensor
    start = time()
    model_output = model(input_ts[query[0] : query[1]].unsqueeze(0)).squeeze()
    elapsed_time["inference"] += time() - start

    start = time()

    unpredictable_tensor = torch.from_numpy(
        unpredictables[0, query[0] : query[1]].todense()
    ).squeeze()
    dequantized_values = decode_quantized_values(
        quantized_code[query[0] : query[1]], err_bound
    )
    elapsed_time["decode"] += time() - start

    start = time()
    # import pdb

    # pdb.set_trace()
    final_output = model_output + dequantized_values + unpredictable_tensor
    elapsed_time["add"] += time() - start


def run_queries(args, input_ts, id, query_range, err_bound):
    model = load_model(args, id)
    if args.inference_method == "bitarray":
        aux_data = torch.load(f"outputs/{args.table_name}/aux/{id}.pt")
        existence_bitarray = bitarray()
        with open(f"outputs/{args.table_name}/aux/{id}.bin", "rb") as f:
            existence_bitarray.fromfile(f)
        inference_func = inference_with_bitarray
        input_data1 = existence_bitarray
        input_data2 = aux_data
    elif args.inference_method == "quantized":
        quantized_code = torch.load(
            f"outputs/{args.table_name}/quantized_aux/{id}_quantized.pt"
        )
        unpredictables = load_npz(
            f"outputs/{args.table_name}/quantized_aux/{id}_unpredictables.npz"
        )
        inference_func = inference_with_quantized_code
        input_data1 = quantized_code
        input_data2 = unpredictables

    elapsed_time = {"inference": 0, "decode": 0, "add": 0}
    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)
        inference_func(
            model,
            input_ts,
            query,
            input_data1,
            input_data2,
            elapsed_time,
            err_bound=err_bound,
        )

    print(f"Time elapsed: {elapsed_time}")
    print(f"Total time: {sum(elapsed_time.values())}")


def run():
    parser = argparse.ArgumentParser(description="DeepMapping-TS inference")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")
    parser.add_argument("--query_type", type=str, default="medium")
    parser.add_argument("--query_size", type=int, default=10000)
    parser.add_argument("--inference_method", type=str, default="quantized")
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

    for ts_id, err_bound in enumerate(err_bounds):
        input_ts = data[corr_dep[ts_id]]
        run_queries(args, input_ts, ts_id + 1, paritition_size, err_bound)
