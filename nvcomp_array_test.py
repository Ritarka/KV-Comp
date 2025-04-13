import torch
from nvidia import nvcomp

codec = nvcomp.Codec(algorithm="ANS", bitstream_kind=nvcomp.BitstreamKind.NVCOMP_NATIVE)

def quantize(array: torch.Tensor, num_bits: int = 2):
    # assert num_bits <= 8, "Just need to change the quantizes astype"
    
    num_levels = 2 ** num_bits
    min_val = array.min()
    max_val = array.max()
    scale = (max_val - min_val) / (num_levels - 1)
    offset = min_val

    quantized = torch.round((array - offset) / scale).to(torch.int8)
    return quantized, scale, offset

# @timer(1)
def dequantize(quantized: torch.Tensor, scale: float, offset: float):
    return quantized.float() * scale + offset

# @timer(2)
def compress_with_nvcomp(tensor: torch.Tensor, codec):
    if tensor.device.type != "cuda":
        tensor = tensor.to("cuda", non_blocking=True)

    # tensor = (tensor * 255).to(torch.uint8)  # maybe better compression for low-precision data
    nvarr_txt_d = nvcomp.as_array(tensor)
    return codec.encode(nvarr_txt_d), tensor.shape

# @timer(3)
def decompress_with_nvcomp(compressed_arr, codec, shape):
    decoded = codec.decode(compressed_arr)  # Avoid unnecessary numpy conversion
    return torch.as_tensor(decoded, dtype=torch.int8, device="cuda").view(shape)  # Direct to FP16

arr = torch.randn(10, 1).cuda()
q_array, scale, shift = quantize(arr, 4)
compressed_arr, original_shape = compress_with_nvcomp(q_array, codec)
decompressed_arr = decompress_with_nvcomp(compressed_arr, codec, original_shape)
dequant_arr = dequantize(decompressed_arr, scale, shift)

print("Original array:", arr.shape)
print("Quantized array:", q_array.shape)
print("Compressed array:", compressed_arr.shape)
print("Decompressed array:", decompressed_arr.shape)
print("Dequantized array:", dequant_arr.shape)
print("Is equal to original?", torch.allclose(arr, dequant_arr, atol=1))

# print(q_array)
# print(decompressed_arr)
print(q_array.dtype)
print(decompressed_arr.dtype)
print(torch.allclose(q_array, decompressed_arr, atol=1e-1))

# print(arr)
print(scale, shift)
print(torch.cat([arr, dequant_arr], dim=1))
print(torch.allclose(arr, dequant_arr, atol=1e-1))
