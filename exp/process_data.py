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


def partition_data(table, partition_size):
    if isinstance(table, pa.Table):
        table = table.to_pandas()

    if partition_size == 0:
        return [table]

    partitions = [
        table.iloc[i : i + partition_size] for i in range(0, len(table), partition_size)
    ]
    return partitions


def get_correlation_dependencies(df):
    df.rename(
        columns={col_name: i for i, col_name in enumerate(df.columns)},
        inplace=True,
        errors="raise",
    )
    corr_dep = {}
    corr_matrix = df.corr()
    # print(corr_matrix)
    # print(corr_matrix.columns)
    for col in corr_matrix.columns:
        corr_dep[col] = corr_matrix[col].abs().sort_values(ascending=False).index[1]
    # print(corr_dep)
    return corr_dep


def load_data(table_name, partition_size=0):
    # Load table
    arrow_table, rel_ebs = load_table(table_name)
    table = arrow_table.to_pandas()
    if table_name == "heavy_drinking":
        table = table.drop("time", axis=1)

    # Compute correlation dependencies
    corr_dep = get_correlation_dependencies(table)

    # Compute error bounds
    err_bounds = compute_err_bounds(table, rel_ebs)

    # Partition data
    partitions = partition_data(table, partition_size)

    # Return partitions and error bounds
    return partitions, err_bounds, corr_dep


if __name__ == "__main__":
    process_gas_data("ethylene_methane")
    # process_drinking_data()
