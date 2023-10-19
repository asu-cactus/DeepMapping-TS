from dahuffman import HuffmanCodec
import zstd
from bitarray import bitarray
import numpy as np
from pathlib import Path
import struct
from math import floor, log2, ceil, copysign


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


def quantization(args, input_array, ref_array, err_bound):
    # Compute some constants
    real_rq_bits = compute_real_rq_bits(ref_array, err_bound)
    middle_code = 2 ** (args.m - 1)

    # Prepare containers for results
    quantized_values = []
    unpredictables = bitarray()
    for input, ref in zip(input_array, ref_array):
        # TODO: check if the following two lines are correct
        sign = copysign(1, ref - input)
        quantized_value = middle_code + sign * (abs(ref - input) // err_bound + 1) // 2

        if quantized_value <= 0 or quantized_value >= 2**args.m:
            compress_unpredictable(input, unpredictables, real_rq_bits)
        else:
            quantized_values.append(quantized_value)

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
