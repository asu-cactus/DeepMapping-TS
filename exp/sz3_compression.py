from pathlib import Path
import numpy as np
from pyarrow import parquet

from exp.sz3.pysz import (
    SZ,
)  # remember to include "third_party" folder in LD_LIBRARY_PATH
from exp.process_data import load_data

folder_name = {
    "ethylene_methane": "gas_sensor",
    "ethylene_CO": "gas_sensor",
    "heavy_drinking": "heavy_drinking",
}


def sz_compress(df, error_bounds, table_name):
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
        folder = folder_name[table_name]
        Path(f"datasets/{folder}/sz_{table_name}").mkdir(parents=True, exist_ok=True)
        with open(f"datasets/{folder}/sz_{table_name}/{col_name}.sz", "wb") as f:
            np.save(f, data_cmpr)


# def load_data(table_name):
#     rel_ebs, path = get_rel_ebs(table_name)
#     arrow_table = parquet.read_table(path)
#     # if table_name == "heavy_drinking":
#     #     rel_ebs = [1.0] + rel_ebs
#     # else:
#     #     rel_ebs.append(1.0)
#     return arrow_table, rel_ebs
