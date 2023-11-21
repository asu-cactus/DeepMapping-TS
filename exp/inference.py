from exp.models.univariate import DilatedConvEncoder, ConvLayer, WindowMLP, WindowRNN
from exp.utils.data_utils import load_data, get_ts_length
from exp.utils.query_generator import generate_query
from exp.sz3.pysz import SZ

from bitarray import bitarray
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz

import argparse
from time import time
import json
import pickle
from pathlib import Path


def load_model(args, id):
    # model = DilatedConvEncoder(
    #     args,
    # )
    if args.model_type == "tcn":
        model = ConvLayer(args)
    elif args.model_type == "mlp":
        model = WindowMLP(args)
    elif args.model_type == "rnn":
        model = WindowRNN(args)
    else:
        raise ValueError("Invalid model type")
    model.load_state_dict(
        torch.load(
            f"saved_models/{args.table_name}/{id}.pt",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    return model


def inference_with_bitarray(args, model, input_ts, aux_data, time_elapsed):
    # TODO: Consider no padding for input tensor

    if args.model_type == "rnn" or "mlp":
        # TODO: Correct the padding position
        original_length = len(input_ts)
        pad_size = args.window_size - len(input_ts) % args.window_size
        input_ts = F.pad(input_ts, (0, pad_size), "constant", 0)

    start = time()
    with torch.no_grad():
        model_output = model(input_ts.unsqueeze(0)).squeeze()
    time_elapsed["inference"] += time() - start

    if args.model_type == "rnn" or "mlp":
        model_output = model_output[:original_length]

    start = time()
    final_output = model_output + aux_data
    time_elapsed["decode"] += time() - start

    return final_output


def inference_with_quantized_code(
    args,
    model,
    input_ts,
    quantized_code,
    unpredictables,
    time_elapsed,
    err_bound=None,
):
    # TODO: Consider no padding for input tensor
    if args.model_type == "rnn" or args.model_type == "mlp":
        # TODO: Correct the padding position
        original_length = len(input_ts)
        pad_size = args.window_size - len(input_ts) % args.window_size
        input_ts = F.pad(input_ts, (0, pad_size), "constant", 0)

    start = time()
    with torch.no_grad():
        model_output = model(input_ts.unsqueeze(0)).squeeze()
    time_elapsed["inference"] += time() - start

    if args.model_type == "rnn" or args.model_type == "mlp":
        model_output = model_output[:original_length]

    start = time()
    # Decode quantized code
    dequantized_values = (quantized_code - 128) * err_bound * 2
    # print(query)
    # print(model_output.shape)
    # print(dequantized_values.shape)
    # print(unpredictables.shape)
    final_output = model_output.numpy() + dequantized_values + unpredictables
    time_elapsed["decode"] += time() - start
    return final_output


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
    else:
        raise ValueError("Invalid inference method")

    elapsed_time = {"inference": 0, "decode": 0, "add": 0}
    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)
        inference_func(
            model,
            input_ts[query[0] : query[1] + 1],
            query,
            input_data1,
            input_data2,
            elapsed_time,
            err_bound=err_bound,
        )

    print(f"Time elapsed: {elapsed_time}")
    print(f"Total time: {sum(elapsed_time.values())}")


def run_queries_v2(args, models, start_column, corr_dep, time_elapsed):
    query_range = len(start_column)
    first_col_id = next(iter(corr_dep.values()))["dep"]
    start_column = torch.from_numpy(start_column).float()

    if args.inference_method == "quantized":
        save_dir = f"outputs/{args.table_name}/quantized_aux"
        start = time()
        unpredictabless = [
            load_npz(f"{save_dir}/{id}.npz") if id != first_col_id else None
            for id in range(len(corr_dep) + 1)
        ]
        time_elapsed["load_unpredictables"] += time() - start
        save_dir = f"{save_dir}/{args.aux_partition_size}"
    elif args.inference_method == "bitarray":
        save_dir = f"outputs/{args.table_name}/aux"
        existence_bitarrays = []
        start = time()
        for id in range(len(corr_dep) + 1):
            if id == first_col_id:
                existence_bitarrays.append(None)
                continue
            existence_bitarray = bitarray()
            with open(f"{save_dir}/{id}.bin", "rb") as f:
                existence_bitarray.fromfile(f)
            existence_bitarrays.append(existence_bitarray)
        time_elapsed["load_bitarray"] += time() - start
        save_dir = f"{save_dir}/{args.aux_partition_size}"
    else:
        raise ValueError("Invalid inference method")

    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)

        results = [None] * (len(corr_dep) + 1)
        results[first_col_id] = start_column[query[0] : query[1] + 1]
        for id, dep_and_eb in corr_dep.items():
            dep, err_bound = dep_and_eb["dep"], dep_and_eb["err_bound"]

            input_ts = results[dep]
            model = models[id]

            if args.inference_method == "bitarray":
                existence_bitarray = existence_bitarrays[id]
                start = time()
                start_idx = existence_bitarray.count(0, 0, query[0])
                length = existence_bitarray.count(0, query[0], query[1] + 1)
                end_idx = start_idx + length
                time_elapsed["convert_incorrect"] += time() - start

                if args.aux_partition_size == 0:
                    start = time()
                    aux_data = torch.load(f"{save_dir}/{id}.pt")
                    selected_aux_data = (value for value in aux_data[start_idx:end_idx])
                    time_elapsed["load_incorrect"] += time() - start

                    start = time()
                else:
                    partitions = []
                    start_partition = start_idx // args.aux_partition_size
                    end_partition = end_idx // args.aux_partition_size
                    start = time()
                    for idx in range(start_partition, end_partition + 1):
                        partition = torch.load(f"{save_dir}/{id}/{idx}.pt")
                        partitions.append(partition)
                    time_elapsed["load_incorrect"] += time() - start

                    start = time()
                    selected_aux_data = torch.cat(partitions)
                    based_idx = start_partition * args.aux_partition_size
                    selected_aux_data = selected_aux_data[
                        start_idx - based_idx : end_idx - based_idx
                    ]
                    selected_aux_data = (value for value in selected_aux_data)

                selected_bits = existence_bitarray[query[0] : query[1] + 1]
                aux_data = torch.tensor(
                    [0.0 if bit else next(selected_aux_data) for bit in selected_bits],
                    dtype=torch.float32,
                )

                time_elapsed["convert_incorrect"] += time() - start

                final_output = inference_with_bitarray(
                    args, model, input_ts, aux_data, time_elapsed
                )

            elif args.inference_method == "quantized":
                start = time()
                if args.aux_partition_size == 0:
                    start = time()
                    quantized_code = torch.load(f"{save_dir}/{id}.pt")
                    time_elapsed["load_quantized"] += time() - start
                    start = time()
                    quantized_code = quantized_code[query[0] : query[1] + 1]
                    time_elapsed["combine_index_quantized"] += time() - start
                else:
                    start_partition = query[0] // args.aux_partition_size
                    end_partition = query[1] // args.aux_partition_size
                    # partitions = []
                    start = time()
                    with open(f"{save_dir}/{id}/{start_partition}.npy", "rb") as f:
                        quantized_code = np.load(f)
                    time_elapsed["load_quantized"] += time() - start

                    if start_partition != end_partition:
                        for idx in range(start_partition + 1, end_partition + 1):
                            start = time()
                            with open(f"{save_dir}/{id}/{idx}.npy", "rb") as f:
                                quantized_code = np.concatenate(
                                    (quantized_code, np.load(f))
                                )
                            time_elapsed["load_quantized"] += time() - start
                    # for idx in range(start_partition, end_partition + 1):
                    #     with open(f"{save_dir}/{id}/{idx}.npy", "rb") as f:
                    #         partitions.append(np.load(f))

                    start = time()
                    # quantized_code = np.concatenate(partitions)

                    # Get quantized code in the range of query
                    based_idx = start_partition * args.aux_partition_size
                    quantized_code = quantized_code[
                        query[0] - based_idx : query[1] + 1 - based_idx
                    ]
                    time_elapsed["combine_index_quantized"] += time() - start

                start = time()
                # This is a sparse vector
                unpredictables = unpredictabless[id][0, query[0] : query[1] + 1]
                # unpredictable_tensor = torch.from_numpy(
                #     unpredictables[0, query[0] : query[1] + 1].todense()
                # ).squeeze()
                time_elapsed["convert_unpredictables"] += time() - start

                final_output = inference_with_quantized_code(
                    args,
                    model,
                    input_ts,
                    quantized_code,
                    unpredictables,
                    time_elapsed,
                    err_bound=err_bound,
                )
            else:
                raise ValueError("Invalid inference method")

            results[id] = torch.from_numpy(final_output).float()


def run_queries_for_pair_grouping(args, models, start_columns, corr_dep, time_elapsed):
    query_range = len(start_columns[0])
    # first_col_id = next(iter(corr_dep.values()))["dep"]
    # start_column = torch.from_numpy(start_column).float()
    start_columns = [
        torch.from_numpy(start_column).float() if start_column is not None else None
        for start_column in start_columns
    ]

    if args.inference_method == "quantized":
        unpredictabless = [None] * len(start_columns)
        save_dir = f"outputs/{args.table_name}/quantized_aux/{args.aux_partition_size}"
        start = time()
        for path in Path(save_dir).iterdir():
            if path.suffix == ".npz":
                unpredictabless[int(path.stem)] = load_npz(path)
        time_elapsed["load_file"] += time() - start
    elif args.inference_method == "bitarray":
        save_dir = f"outputs/{args.table_name}/aux/{args.aux_partition_size}"
        existence_bitarrays = [None] * len(start_columns)
        start = time()
        for path in Path(save_dir).iterdir():
            if path.suffix == ".bin":
                existence_bitarray = bitarray()
                with open(path, "rb") as f:
                    existence_bitarray.fromfile(f)
                existence_bitarrays[int(path.stem)] = existence_bitarray
        time_elapsed["load_bitarray"] += time() - start
    else:
        raise ValueError("Invalid inference method")

    for _ in range(args.query_size):
        query = generate_query(args.query_type, query_range)

        results = [
            start_column[query[0] : query[1] + 1] if start_column is not None else None
            for start_column in start_columns
        ]

        for id, dep_and_eb in corr_dep.items():
            dep, err_bound = dep_and_eb["dep"], dep_and_eb["err_bound"]

            input_ts = results[dep]
            model = models[id]

            if args.inference_method == "bitarray":
                existence_bitarray = existence_bitarrays[id]
                start = time()
                start_idx = existence_bitarray.count(0, 0, query[0])
                length = existence_bitarray.count(0, query[0], query[1])
                end_idx = start_idx + length

                if args.aux_partition_size == 0:
                    aux_data = torch.load(f"{save_dir}/{id}.pt")
                    selected_aux_data = (value for value in aux_data[start_idx:end_idx])

                else:
                    partitions = []
                    start_partition = start_idx // args.aux_partition_size
                    end_partition = end_idx // args.aux_partition_size
                    for idx in range(start_partition, end_partition + 1):
                        partition = torch.load(f"{save_dir}/{id}/{idx}.pt")
                        partitions.append(partition)
                    selected_aux_data = torch.cat(partitions)
                    based_idx = start_partition * args.aux_partition_size
                    selected_aux_data = selected_aux_data[
                        start_idx - based_idx : end_idx - based_idx
                    ]
                    selected_aux_data = (value for value in selected_aux_data)

                selected_bits = existence_bitarray[query[0] : query[1] + 1]
                aux_data = torch.tensor(
                    [0.0 if bit else next(selected_aux_data) for bit in selected_bits],
                    dtype=torch.float32,
                )

                time_elapsed["load_incorrect"] += time() - start

                final_output = inference_with_bitarray(
                    args, model, input_ts, aux_data, time_elapsed
                )

            elif args.inference_method == "quantized":
                start = time()
                if args.aux_partition_size == 0:
                    quantized_code = torch.load(f"{save_dir}/{id}.pt")[
                        query[0] : query[1] + 1
                    ]
                else:
                    partitions = []
                    start_partition = query[0] // args.aux_partition_size
                    end_partition = query[1] // args.aux_partition_size
                    for idx in range(start_partition, end_partition + 1):
                        partition = torch.load(f"{save_dir}/{id}/{idx}.pt")
                        partitions.append(partition)
                    quantized_code = torch.cat(partitions)
                    # Get quantized code in the range of query
                    based_idx = start_partition * args.aux_partition_size
                    quantized_code = quantized_code[
                        query[0] - based_idx : query[1] + 1 - based_idx
                    ]
                time_elapsed["load_quantized"] += time() - start
                start = time()

                unpredictables = unpredictabless[id]
                unpredictable_tensor = torch.from_numpy(
                    unpredictables[0, query[0] : query[1] + 1].todense()
                ).squeeze()
                time_elapsed["load_incorrect"] += time() - start
                final_output = inference_with_quantized_code(
                    args,
                    model,
                    input_ts,
                    quantized_code,
                    unpredictable_tensor,
                    time_elapsed,
                    err_bound=err_bound,
                )

            else:
                raise ValueError("Invalid inference method")

            results[id] = final_output


def run():
    parser = argparse.ArgumentParser(description="DeepMapping-TS inference")
    parser.add_argument("--table_name", type=str, default="ethylene_CO")
    parser.add_argument("--group_mode", type=str, default="start_one")
    parser.add_argument("--query_type", type=str, default="short")
    parser.add_argument("--query_size", type=int, default=1000)
    parser.add_argument("--model_type", type=str, default="tcn")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--inference_method", type=str, default="quantized")
    parser.add_argument("--aux_partition_size", type=int, default=100000)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--activation", type=str, default="identity")
    args = parser.parse_args()
    # # Load data
    # ts_length = get_ts_length(args.table_name)
    # partitions, err_bounds, corr_dep = load_data(args.table_name, ts_length)
    # print("Number of partitions:", len(partitions))

    # # Convert partitions to tensors
    # data = torch.tensor(partitions[0].to_numpy().transpose(), dtype=torch.float32)

    # for ts_id, err_bound in enumerate(err_bounds):
    #     input_ts = data[corr_dep[ts_id]]
    #     run_queries(args, input_ts, ts_id, ts_length, err_bound)

    time_elapsed = {
        "inference": 0,
        "decompress": 0,
        "load_file": 0,
        "load_unpredictables": 0,
        "load_bitarray": 0,
        "load_quantized": 0,
        "combine_index_quantized": 0,
        "decode": 0,
        "convert_unpredictables": 0,
        "load_incorrect": 0,
        "convert_incorrect": 0,
    }
    # Get correlation dependencies
    start = time()
    with open(f"outputs/{args.table_name}/aux/corr_dep.pkl", "rb") as f:
        corr_dep = pickle.load(f)
    time_elapsed["load_file"] += time() - start
    # corr_dep = json.load(open(f"outputs/{args.table_name}/aux/corr_dep.json"))

    # Decompress to get the start column
    sz = SZ("exp/sz3/libSZ3c.so")
    if args.group_mode == "start_one":
        start = time()
        with open(f"outputs/{args.table_name}/aux/start_column.sz", "rb") as f:
            data_cmpr = np.load(f)
        ts_length = get_ts_length(args.table_name)
        data_dec = sz.decompress(data_cmpr, (ts_length,), np.float32, time_elapsed)
        time_elapsed["decompress"] += time() - start

        # Load models
        start_column = next(iter(corr_dep.values()))["dep"]
        models = [
            load_model(args, id) if id != start_column else None
            for id in range(len(corr_dep) + 1)
        ]

        # Run query
        run_queries_v2(args, models, data_dec, corr_dep, time_elapsed)
    elif args.group_mode == "pair":
        start_columns = [None] * (len(corr_dep) * 2)
        start = time()
        for path in Path(f"outputs/{args.table_name}/aux/").iterdir():
            if path.suffix == ".sz":
                with open(path, "rb") as f:
                    data_cmpr = np.load(f)
                ts_length = get_ts_length(args.table_name)
                data_dec = sz.decompress(
                    data_cmpr, (ts_length,), np.float32, time_elapsed
                )
                start_columns[int(path.stem)] = data_dec
        time_elapsed["decompress"] += time() - start

        # Load models

        models = [None] * (len(corr_dep) * 2)
        for path in Path(f"saved_models/{args.table_name}").iterdir():
            if path.is_dir():
                continue
            id = int(path.stem)
            start = time()
            models[id] = load_model(args, id)
            time_elapsed["load_file"] += time() - start

        # Run query
        run_queries_for_pair_grouping(
            args, models, start_columns, corr_dep, time_elapsed
        )
    else:
        raise ValueError("Invalid group mode")
    print(f"Time elapsed: {time_elapsed}")
    print(f"Total time: {sum(time_elapsed.values())}")
