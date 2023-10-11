from pathlib import Path
import numpy as np
from pyarrow import parquet

from sz3.pysz import (
    SZ,
)  # remember to include "third_party" folder in LD_LIBRARY_PATH
from process_data import get_rel_ebs

folder_name = {
    "ethylene_methane": "gas_sensor",
    "ethylene_CO": "gas_sensor",
    "heavy_drinking": "heavy_drinking",
}


def sz_compress(table, rel_ebs, table_name):
    sz = SZ("sz3/libSZ3c.so")
    for column, rel_eb, col_name in zip(table.columns, rel_ebs, table.column_names):
        column_data = column.to_numpy().astype(np.float32)
        data_cmpr, cmpr_ratio = sz.compress(
            column_data,
            1,
            0,
            rel_eb,
            0,
        )
        folder = folder_name[table_name]
        Path(f"../datasets/{folder}/sz_{table_name}").mkdir(parents=True, exist_ok=True)
        with open(f"../datasets/{folder}/sz_{table_name}/{col_name}.sz", "wb") as f:
            np.save(f, data_cmpr)


def load_data(table_name):
    rel_ebs, path = get_rel_ebs(table_name)
    arrow_table = parquet.read_table(path)
    if table_name == "heavy_drinking":
        rel_ebs = [1.0] + rel_ebs
    else:
        rel_ebs.append(1.0)
    rel_ebs = [rel_eb * 1e-2 for rel_eb in rel_ebs]
    return arrow_table, rel_ebs


if __name__ == "__main__":
    table_name = "ethylene_methane"
    arrow_table, rel_ebs = load_data(table_name)

    sz_compress(arrow_table, rel_ebs, table_name)
