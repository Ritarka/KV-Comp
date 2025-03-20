import torch
import numpy as np
import matplotlib.pyplot as plt
from nvidia import nvcomp

def quantize(array: np.ndarray, num_bits: int) -> tuple[np.ndarray, float, float]:
    num_levels = 2 ** num_bits
    min_val, max_val = array.min(), array.max()
    scale = (max_val - min_val) / (num_levels - 1)
    offset = min_val
    quantized = np.round((array - offset) / scale).astype(int)
    return quantized, scale, offset

# Generate test data of size 10^6
size = 10**6
array = torch.randn(size).numpy()

# Quantize the array
q_array, scale, shift = quantize(array, 2)
nvarr_txt_h = nvcomp.as_array(q_array)
nvarr_txt_d = nvarr_txt_h.cuda()

# List of algorithms
algos = ["LZ4", "Snappy", "Bitcomp", "ANS", "Zstd", "Cascaded"]
bitstream_kind = nvcomp.BitstreamKind.NVCOMP_NATIVE  # Fixed for comparison

comp_times = []
decomp_times = []
comp_ratios = []

# Measure time for each algorithm
for algorithm in algos:
    codec = nvcomp.Codec(algorithm=algorithm, bitstream_kind=bitstream_kind)

    # Measure compression time
    torch.cuda.synchronize()
    start_comp = torch.cuda.Event(enable_timing=True)
    end_comp = torch.cuda.Event(enable_timing=True)
    
    start_comp.record()
    comp_arr = codec.encode(nvarr_txt_d)
    end_comp.record()
    
    torch.cuda.synchronize()
    comp_time = start_comp.elapsed_time(end_comp)  # Time in milliseconds
    comp_times.append(comp_time)

    # Measure decompression time
    torch.cuda.synchronize()
    start_decomp = torch.cuda.Event(enable_timing=True)
    end_decomp = torch.cuda.Event(enable_timing=True)

    start_decomp.record()
    decomp_array = codec.decode(comp_arr)
    end_decomp.record()

    torch.cuda.synchronize()
    decomp_time = start_decomp.elapsed_time(end_decomp)  # Time in milliseconds
    decomp_times.append(decomp_time)

    # Compute compression ratio
    comp_ratio = comp_arr.buffer_size / nvarr_txt_d.buffer_size
    comp_ratios.append(1 / comp_ratio)

    print(f"Algorithm: {algorithm}")
    print(f"  Compression Time: {comp_time:.3f} ms")
    print(f"  Decompression Time: {decomp_time:.3f} ms")
    print(f"  Compression Ratio: {1 / comp_ratio:.2f}")

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

ind = np.arange(len(algos))
width = 0.4

# Compression Time Plot
axs[0].bar(ind, comp_times, width, color='blue', label="Compression Time (ms)")
axs[0].set_ylabel("Time (ms)")
axs[0].set_title("Compression Time for Different Algorithms")
axs[0].set_xticks(ind)
axs[0].set_xticklabels(algos, rotation=45)
axs[0].legend()

# Decompression Time Plot
axs[1].bar(ind, decomp_times, width, color='green', label="Decompression Time (ms)")
axs[1].set_ylabel("Time (ms)")
axs[1].set_title("Decompression Time for Different Algorithms")
axs[1].set_xticks(ind)
axs[1].set_xticklabels(algos, rotation=45)
axs[1].legend()

plt.tight_layout()
plt.savefig("images/compression_decompression_times.png")
plt.show()
