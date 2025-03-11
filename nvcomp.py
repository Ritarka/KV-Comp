import torch
import cupy as cp
import numpy as np
import urllib.request
from rich import print
from nvidia import nvcomp
import matplotlib.pyplot as plt  
import GPUtil

print("nvcomp version:", nvcomp.__version__)
print("nvcomp cuda version:", nvcomp.__cuda_version__)

# urllib.request.urlretrieve("http://textfiles.com/etext/NONFICTION/locke-essay-113.txt", "locke-essay-113.txt")
# urllib.request.urlretrieve("http://textfiles.com/etext/FICTION/mobydick.txt", "mobydick.txt")



def get_gpu_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"  Total Memory: {gpu.memoryTotal} MB")
        print(f"  Free Memory: {gpu.memoryFree} MB")
        print(f"  Used Memory: {gpu.memoryUsed} MB")

get_gpu_memory()


def quantize(array: np.ndarray, num_bits: int) -> tuple[np.ndarray, float, float]:
    assert num_bits <= 8, "Just need to change the quantizes astype"
    
    num_levels = 2 ** num_bits
    
    min_val = array.min()
    max_val = array.max()
    scale = (max_val - min_val) / (num_levels - 1)
    offset = min_val
    
    quantized = np.round((array - offset) / scale).astype(int)
    return quantized, scale, offset


def dequantize(quantized: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return quantized * scale + offset

array = torch.randn(100, 100).numpy()
q_array, scale, shift = quantize(array, 2)
nvarr_txt_h = nvcomp.as_array(q_array)
nvarr_txt_d = nvarr_txt_h.cuda()


print("Uncompressed size is", nvarr_txt_d.buffer_size)
algos = ["LZ4", "Snappy", "Bitcomp", "ANS", "Zstd",  "Cascaded"]
bitstreams = [
    nvcomp.BitstreamKind.NVCOMP_NATIVE,
    nvcomp.BitstreamKind.RAW,
    nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE
]

data = [[], [], []]
index = 0
for bitstream_kind in bitstreams:
    for algorithm in algos:
        codec = nvcomp.Codec(algorithm=algorithm, bitstream_kind=bitstream_kind)
        comp_arr = codec.encode(nvarr_txt_d)
        comp_ratio = comp_arr.buffer_size/nvarr_txt_d.buffer_size
        print("Compressed size for", algorithm, "with bitstream", bitstream_kind, "is", comp_arr.buffer_size, "({:.1%})".format(comp_ratio))
        decomp_array = codec.decode(comp_arr)
        # print ("is equal to original? -", bytes(decomp_array.cpu()) ==  bytes(nvarr_txt_d.cpu()))
        data[index].append(round(1 / comp_ratio, 2))
    index += 1
    
# print(data)
        
        
ind = np.arange(len(algos))  
width = 0.25
  
bar1 = plt.bar(ind, data[0], width)
bar2 = plt.bar(ind+width, data[1], width) 
bar3 = plt.bar(ind+width*2, data[2], width) 
  
plt.xlabel("Algorithms") 
plt.ylabel("Compression ratio") 
plt.title("Comparing Bitstreams and algo types") 
  
plt.xticks(ind+width, algos) 
plt.legend((bar1, bar2, bar3), ('NVCOMP_NATIVE', 'RAW', 'WITH_UNCOMPRESSED_SIZE')) 
plt.show()
plt.savefig("images/bitstream_comparisons.png")

# get ready to plot more stuff
plt.clf()

array_sizes = [3, 4, 5, 6, 7]
data = np.zeros((6, 5))

index = 0
for algorithm in algos:
    sub_index = 0
    for size in array_sizes:
        minor_size = size // 2
        if size % 2 == 0:
            array = torch.randn(10 ** minor_size, 10 ** minor_size).numpy()
        else:
            array = torch.randn(10 ** minor_size, 10 ** (minor_size + 1)).numpy()

            
        q_array, scale, shift = quantize(array, 2)
        nvarr_txt_h = nvcomp.as_array(q_array)
        nvarr_txt_d = nvarr_txt_h.cuda()        
        
        codec = nvcomp.Codec(algorithm=algorithm, bitstream_kind=bitstream_kind)
        comp_arr = codec.encode(nvarr_txt_d)
        comp_ratio = comp_arr.buffer_size/nvarr_txt_d.buffer_size
        print("Compressed size for", algorithm, "with bitstream", bitstream_kind, "is", comp_arr.buffer_size, "({:.1%})".format(comp_ratio))
        decomp_array = codec.decode(comp_arr)
        # print ("is equal to original? -", bytes(decomp_array.cpu()) ==  bytes(nvarr_txt_d.cpu()))
        data[index][sub_index] = round(1 / comp_ratio, 2)
        
        sub_index += 1
    index += 1

print(data)

ind = np.arange(len(array_sizes))  
width = 0.1

bars = []
for indx, subdata in enumerate(data):
    bars.append(plt.bar(ind + width * indx, subdata, width))
    # bar1 = plt.bar(ind, data[0], width)
    # bar2 = plt.bar(ind+width, data[1], width) 
    # bar3 = plt.bar(ind+width*2, data[2], width) 
  
plt.xlabel("Compression Sizes (# of ints)") 
plt.ylabel("Compression Ratio") 
plt.title("Comparing Compression Ratio with Size") 

xticks = [f"10^{size}" for size in array_sizes]
  
plt.xticks(ind+width, xticks) 
plt.legend(bars, algos) 
plt.show()
plt.savefig("images/size_comparisons.png")