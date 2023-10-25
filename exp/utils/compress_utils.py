from dahuffman import HuffmanCodec
import zstd
from bitarray import bitarray
import numpy as np
import torch
from pathlib import Path
import struct
from math import floor, log2, ceil, copysign
from scipy.sparse import csc_array, save_npz


def compress_main(args, input_array, ref_array, err_bound, col_name):
    quantized_values, unpredictables = quantization(
        args, input_array, ref_array, err_bound
    )
    huffman_codes = get_huffman_coding(quantized_values)
    compr_data = compress_huffman_coding(huffman_codes)
    unpredictables = further_compress_unpredictables(unpredictables)
    # Concatenate compressed_huffman_codes and compressed_unpredictables and save
    save_dir = f"outputs/{args.table_name}/{args.mode}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_dir}/{col_name}.bin"
    print(f"{args.table_name} {col_name} compr_data length: {len(compr_data)}")
    print(f"{args.table_name} {col_name} unpredictables length: {len(unpredictables)}")

    with open(save_path, "wb") as f:
        f.write(compr_data + unpredictables)


def compute_real_rq_bits(nparray, error_bound):
    max_value = nparray.max()
    min_value = nparray.min()
    radius = abs(max_value - min_value)

    exp_radius = floor(log2(radius))
    exp_error_bound = floor(log2(error_bound))
    if exp_radius - exp_error_bound < 0:
        rq_mbits = 0
    elif exp_radius - exp_error_bound < 23:  # mantissa for float32 numbers
        rq_mbits = 23
    else:
        rq_mbits = exp_radius - exp_error_bound
    real_rq_bits = ceil((1 + 8 + rq_mbits) / 8) * 8
    return real_rq_bits


def quantization(
    args, input_array, ref_array, err_bound, is_compress_unpredictable=True
):
    # Compute some constants
    real_rq_bits = compute_real_rq_bits(ref_array, err_bound)
    middle_code = 2 ** (args.m - 1)

    # Prepare containers for results
    quantized_values = []
    unpredictables = bitarray() if is_compress_unpredictable else []
    for input, ref in zip(input_array, ref_array):
        # TODO: check if the following two lines are correct
        sign = copysign(1, ref - input)
        quantized_value = middle_code + sign * (abs(ref - input) // err_bound + 1) // 2

        if quantized_value <= 0 or quantized_value >= 2**args.m:
            # Cannot be predicted
            if is_compress_unpredictable:
                quantized_values.append(0)
                compress_unpredictable(input, unpredictables, real_rq_bits)
            else:
                quantized_values.append(middle_code)
                unpredictables.append(ref)
        else:
            # Can be predicted
            quantized_values.append(quantized_value)
            if not is_compress_unpredictable:
                unpredictables.append(0)

    return quantized_values, unpredictables


def get_huffman_coding(quantized_values) -> bytes:
    # TODO: progressively build the huffman coding dictionary
    codec = HuffmanCodec.from_data(quantized_values)
    return codec.encode(quantized_values)


def compress_huffman_coding(huffman_codes) -> bytes:
    return zstd.compress(huffman_codes, 22)


def compress_unpredictable(num, unpredictables, real_rq_bits):
    binary_value = "".join(format(c, "0>8b") for c in struct.pack("!f", num))
    unpredictables.extend(binary_value[:real_rq_bits])


def further_compress_unpredictables(unpredictables) -> bytes:
    def bitarray_to_numpy(bitarray):
        # Temporary solution: do not consider the last byte
        return np.frombuffer(bitarray.unpack(), dtype=bool)

    return zstd.compress(bitarray_to_numpy(unpredictables), 22)


def create_quantized_aux_structure(
    args, input_array, ref_array, err_bound, col_name, is_compress_unpredictable=False
):
    args.m = 8
    quantized_values, unpredictables = quantization(
        args, input_array, ref_array, err_bound, is_compress_unpredictable
    )
    assert len(quantized_values) == len(
        unpredictables
    ), f"len(quantized_values): {len(quantized_values)}, len(unpredictables): {len(unpredictables)}"

    print(f"Unpredictables: {np.count_nonzero(unpredictables)}")

    quantized_values = torch.tensor(quantized_values, dtype=torch.uint8)
    unpredictables = csc_array(np.array(unpredictables, dtype=np.float32))
    # Save quantized_values and unpredictables
    save_dir = f"outputs/{args.table_name}/quantized_aux"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        quantized_values,
        f"outputs/{args.table_name}/quantized_aux/{col_name}_quantized.pt",
    )
    save_npz(
        f"outputs/{args.table_name}/quantized_aux/{col_name}_unpredictables.npz",
        unpredictables,
    )
    return quantized_values, unpredictables


def decode_quantized_values(quantized_values: torch.Tensor, err_bound: float):
    return (quantized_values - 2**7) * err_bound * 2
