# LLaMA model with KIVI
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
import time

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 12
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-2-7b-hf'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.use_flash = True
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

TARGET_GPU = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)  # Restrict PyTorch to only see this GPU

# Set device in PyTorch
device = torch.device(f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu")

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:1",
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:1",
    )

model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

model.cuda(1).eval()

batch_size = BATCH_SIZE
prompt_lenth = 160
output_length = 338
num_repeats = 3
context = []
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda:1')
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
torch.cuda.reset_peak_memory_stats(1)
with torch.no_grad():
    torch.cuda.synchronize(1)
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length)
    torch.cuda.synchronize(1)
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated(1)
    print(f'peak mem: {used_mem / 1024 ** 3} GB')



"""
KIVI-metrics
bs: 48, seqlen: 160+338
model:meta-llama/Llama-2-7b-hf
used time: 13870.40893236796 ms
peak mem: 20.762739658355713 GB

LLAMA regular metrics
bs: 12, seqlen: 160+338
model:meta-llama/Llama-2-7b-hf
used time: 8739.897648493448 ms
peak mem: 15.521936893463135 GB

KIVI-metrics
bs: 12, seqlen: 160+338
model:meta-llama/Llama-2-7b-hf
used time: 8939.444144566854 ms
peak mem: 14.685456275939941 GB
"""