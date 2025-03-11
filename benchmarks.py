import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
import time

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 24
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-2-7b-hf'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS
config.v_bits = V_BITS
config.use_flash = True
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

# Isolate to only one device to avoid error
TARGET_GPU = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)
device = torch.device(f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu")

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

model.cuda().eval()

context = []
batch_size = BATCH_SIZE
prompt_length = 160
output_length = 338
num_repeats = 3
for _ in range(batch_size):
    string = 't,' * (prompt_length // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']

total_tokens = batch_size * output_length  # Total tokens generated across all prompts
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length)
    torch.cuda.synchronize()
    total_time = (time.time() - st) / num_repeats
    
    print(f'Used time: {total_time * 1000:.2f} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {used_mem / 1024 ** 3:.2f} GB')
    
    # Throughput Calculations
    tokens_per_second = total_tokens / total_time
    prompts_per_second = batch_size / total_time
    
    print(f'Throughput: {tokens_per_second:.2f} tokens/sec')
    print(f'Throughput: {prompts_per_second:.2f} prompts/sec')

    # Cache Utilization Analysis
    cache_mem_used = torch.cuda.memory_allocated()
    cache_mem_free = torch.cuda.memory_reserved() - cache_mem_used
    cache_utilization = (cache_mem_used / torch.cuda.memory_reserved()) * 100 if torch.cuda.memory_reserved() > 0 else 0

    print(f'Cache Memory Used: {cache_mem_used / 1024 ** 3:.2f} GB')
    print(f'Cache Memory Free: {cache_mem_free / 1024 ** 3:.2f} GB')
    print(f'Cache Utilization: {cache_utilization:.2f}%')
