from exp.process_data import load_data
from exp.sz3_compression import sz_compress


if __name__ == "__main__":
    table_name = "ethylene_CO"
    df, error_bounds = load_data(table_name)

    sz_compress(df, error_bounds, table_name)
