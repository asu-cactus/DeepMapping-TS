import math
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet


def process_gas_data(file_name):
    with open(f"../datasets/gas_sensor/{file_name}.txt", "r") as f:
        lines = f.readlines()
    width = 16
    length = len(lines)
    data_array = np.empty((length, width), dtype=np.float32)
    for i, line in enumerate(lines[1:]):
        values = [float(v) for v in line.split()[3:]]
        assert len(values) == width
        data_array[i] = values
    df = pd.DataFrame(data_array, columns=[f"sensor_{i}" for i in range(1, width + 1)])
    # df["timestamp"] = range(0, length)
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Save as parquet
    table = pa.Table.from_pandas(df)
    # pdb.set_trace()
    parquet.write_table(table, f"../datasets/gas_sensor/{file_name}.parquet")


def process_drinking_data():
    df = pd.read_csv(
        "../datasets/heavy_drinking/all_accelerometer_data_pids_13.csv",
        usecols=[0, 2, 3, 4],
        header=0,
    )
    # df["time"] = pd.to_datetime(df["time"], unit="ms")
    # Save as parquet
    table = pa.Table.from_pandas(df)
    # pdb.set_trace()
    parquet.write_table(
        table, "../datasets/heavy_drinking/all_accelerometer_data_pids_13.parquet"
    )


def get_rel_ebs(table_name):
    if table_name == "ethylene_methane":
        rel_ebs = [0.001] * 16
        parquet_file_or_folder = "datasets/gas_sensor/ethylene_methane.parquet"
    elif table_name == "ethylene_CO":
        rel_ebs = [0.001] * 16
        parquet_file_or_folder = "datasets/gas_sensor/ethylene_CO.parquet"
    elif table_name == "heavy_drinking":
        rel_ebs = [0.001] * 3
        parquet_file_or_folder = (
            "datasets/heavy_drinking/all_accelerometer_data_pids_13.parquet"
        )
    else:
        raise ValueError(f"Unsupported Table Name: {table_name}")
    return rel_ebs, parquet_file_or_folder


def compute_err_bounds(df, rel_ebs):
    err_bounds = (df.max(axis=0) - df.min(axis=0)).abs() * rel_ebs
    return err_bounds


def load_table(table_name):
    rel_ebs, path = get_rel_ebs(table_name)
    arrow_table = parquet.read_table(path)

    return arrow_table, rel_ebs


def get_ts_length(table_name):
    if table_name == "ethylene_CO":
        return 4208262
    elif table_name == "ethylene_methane":
        return 4178505
    else:
        raise ValueError(f"Unknown table name: {table_name}")


def ts_distortion(ts1: np.array, ts2: np.array, mode="mse"):
    """ts1 is the original time series, ts2 is the reconstructed time series"""
    mse = np.mean((ts1 - ts2) ** 2)
    if mode == "mse":  # Mean Squared Error
        return mse
    elif mode == "rmse":  # Root Mean Squared Error
        return np.sqrt(mse)
    elif mode == "mae":  # Mean Absolute Error
        return np.mean(np.abs(ts1 - ts2))
    elif mode == "snr":  # Signal to Noise Ratio
        return np.mean(ts1**2) / mse
    elif mode == "psnr":  # Peak Signal to Noise Ratio
        return np.max(ts1) ** 2 / mse
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_data(args, truncate=True):
    # Load table
    arrow_table, rel_ebs = load_table(args.table_name)
    df = arrow_table.to_pandas()
    ts_length = len(df)
    print(f"Original time series length: {ts_length}")
    if args.table_name == "heavy_drinking":
        df = df.drop("time", axis=1)
    # Hard code selected columns
    df = df.iloc[:, [2, 3, 4, 5, 6, 7, 10, 11]]
    rel_ebs = rel_ebs[:8]

    # Truncate table
    if truncate:
        truncate_length = (len(df) // args.window_size) * args.window_size
        df = df.iloc[:truncate_length]
        ts_length = len(df)
        print(f"Truncated time series length: {ts_length}")

    # Compute error bounds
    err_bounds = compute_err_bounds(df, rel_ebs)

    # Return table and error bounds
    return df, err_bounds


def normalize_data(args, df):
    if args.norm_mode == "zscore":
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        df = (df - mean) / std
        return df, mean, std
    elif args.norm_mode == "minmax":
        min = df.min(axis=0)
        max = df.max(axis=0)
        df = (df - min) / (max - min)
        return df, min, max
    else:
        raise ValueError(f"Unknown normalization mode: {args.norm_mode}")


def denormalize_data(args, ndarray, param1, param2):
    if args.norm_mode == "zscore":
        mean, std = param1, param2
        ndarray = ndarray * std + mean
        return ndarray
    elif args.norm_mode == "minmax":
        min, max = param1, param2
        ndarray = ndarray * (max - min) + min
        return ndarray
    else:
        raise ValueError(f"Unknown normalization mode: {args.norm_mode}")


def preprocess(args, df):
    length = len(df) // args.reduce_factor
    width = len(df.columns)
    reduced_ndarray = np.empty((length, width), dtype=np.float32)
    if args.process_mode == "fourier":
        pass
    elif args.process_mode == "average":
        for i in range(length):
            start_idx = i * args.reduce_factor
            end_idx = (i + 1) * args.reduce_factor
            reduced_ndarray[i] = df.iloc[start_idx:end_idx].mean(axis=0).values
    else:
        raise NotImplementedError

    return reduced_ndarray


def postprocess(args, data):
    if args.process_mode == "fourier":
        pass
    elif args.process_mode == "average":
        pass
    else:
        raise NotImplementedError
