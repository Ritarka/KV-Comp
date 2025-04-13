from nvidia import nvcomp
import torch
import gc

codec = nvcomp.Codec(algorithm="LZ4", bitstream_kind=nvcomp.BitstreamKind.NVCOMP_NATIVE)
arr = torch.randn(100, 100).cuda()
arr = arr.cpu().numpy()
arr = nvcomp.as_array(arr)
arr = codec.encode(arr)
arr = codec.decode(arr)
print(arr)

torch.cuda.empty_cache()
gc.collect()

for i in range(1000):
    arr = torch.randn(100, 100).cuda()