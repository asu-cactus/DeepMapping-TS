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


def get_correlation_dependencies(df, mode="start_one"):
    df.rename(
        columns={col_name: i for i, col_name in enumerate(df.columns)},
        inplace=True,
        errors="raise",
    )
    corr_dep = {}
    corr_matrix = df.corr()
    # print(corr_matrix)
    # print(corr_matrix.columns)
    if mode == "greedy":
        for col in corr_matrix.columns:
            corr_dep[col] = corr_matrix[col].abs().sort_values(ascending=False).index[1]

    elif mode == "start_one":
        # TODO: Determine whether to use absoulte correlation or not
        corr_matrix = corr_matrix.to_numpy()
        open_list = []
        closed_list = list(range(len(df.columns)))

        # Get the the index of max correlation
        for i in range(len(corr_matrix)):
            corr_matrix[i][i] = -1
        max_corr_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        # Get the first two columns
        corr_dep[max_corr_idx[1]] = max_corr_idx[0]
        open_list.append(max_corr_idx[0])
        open_list.append(max_corr_idx[1])
        closed_list.remove(max_corr_idx[0])
        closed_list.remove(max_corr_idx[1])

        # Get the rest of the columns
        while len(closed_list) > 0:
            max_corr = 0
            max_corr_idx = None
            for i in open_list:
                for j in closed_list:
                    if corr_matrix[i][j] > max_corr:
                        max_corr = corr_matrix[i][j]
                        max_corr_idx = (i, j)
            corr_dep[max_corr_idx[1]] = max_corr_idx[0]
            open_list.append(max_corr_idx[1])
            closed_list.remove(max_corr_idx[1])
    elif mode == "pair":
        corr_matrix = corr_matrix.to_numpy()
        closed_list = list(range(len(df.columns)))

        # Get the the index of max correlation
        for i in range(len(corr_matrix)):
            corr_matrix[i, i] = -1
        max_corr_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        # Get the first two columns
        corr_dep[max_corr_idx[1]] = max_corr_idx[0]
        closed_list.remove(max_corr_idx[0])
        closed_list.remove(max_corr_idx[1])

        while len(closed_list) > 1:
            max_corr = 0
            max_index = None
            for i in range(0, len(closed_list) - 1):
                for j in range(i, len(closed_list)):
                    first = closed_list[i]
                    second = closed_list[j]
                    if corr_matrix[first, second] > max_corr:
                        max_corr = corr_matrix[first, second]
                        max_index = (first, second)
            corr_dep[max_index[1]] = max_index[0]
            closed_list.remove(max_index[0])
            closed_list.remove(max_index[1])

    else:
        raise ValueError(f"Unknown mode: {mode}")
    corr_dep = {int(k): int(v) for k, v in corr_dep.items()}
    print(f"Correlation dependencies: {corr_dep}")
    return corr_dep


def load_data(table_name, partition_size=0, mode="start_one"):
    # Load table
    arrow_table, rel_ebs = load_table(table_name)
    table = arrow_table.to_pandas()
    ts_length = len(table)
    print(f"Time series length: {ts_length}")
    if table_name == "heavy_drinking":
        table = table.drop("time", axis=1)

    # Compute correlation dependencies
    corr_dep = get_correlation_dependencies(table, mode)

    # Compute error bounds
    err_bounds = compute_err_bounds(table, rel_ebs)

    # Partition data
    partitions = partition_data(table, partition_size)

    # Return partitions and error bounds
    return partitions, err_bounds, corr_dep


def get_ts_length(table_name):
    if table_name == "ethylene_CO":
        return 4208262
    elif table_name == "ethylene_methane":
        return 4178505
    else:
        raise ValueError(f"Unknown table name: {table_name}")
