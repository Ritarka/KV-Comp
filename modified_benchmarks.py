from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset

import os
import random
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaConfig
from accelerate import infer_auto_device_map, dispatch_model

import matplotlib.pyplot as plt
import gc


def dataset_params(dataset_name):
    if 'wikitext2' in dataset_name:
        return 'wikitext', 'wikitext-2-raw-v1'
    elif 'c4' in dataset_name:
        raise NotImplementedError
    elif 'redpajama' in dataset_name:
        raise NotImplementedError
    else:
        raise NotImplementedError

@torch.no_grad()
def benchmark_model(model, tokenizer, dataset_name='wikitext2', max_batch_size=512, batch_step=16):
    """Runs benchmarks for runtime, peak memory, throughput, and cache utilization."""
    
    print("Starting benchmark...")

    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: "24GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)

    dataset_string, dataset_file = dataset_params(dataset_name)
    
    batch_sizes, runtimes, memory_usages, throughputs, cache_usages = [], [], [], [], []

    batch_size = batch_step
    prompt_length = 16
    output_length = 32

    use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()

    while batch_size <= max_batch_size:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            primary_device = next(model.parameters()).device
            data = load_dataset(dataset_string, dataset_file, split='test')
            context = data['text']
            batch = tokenizer(context[:prompt_length], return_tensors="pt", padding=True, truncation=True).to(primary_device)
            input_ids = batch['input_ids']

            start_time = time.time()
            torch.cuda.synchronize()
            outputs = model.generate(**batch, max_new_tokens=output_length)
            torch.cuda.synchronize()
            end_time = time.time()

            # Calculate the metrics
            runtime = (end_time - start_time) * 1000  # Convert to milliseconds
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            num_tokens = batch_size * output_length
            throughput = num_tokens / (runtime / 1000)

            # Store results
            batch_sizes.append(batch_size)
            runtimes.append(runtime)
            memory_usages.append(peak_memory)
            throughputs.append(throughput)
            cache_usages.append(peak_memory / (torch.cuda.get_device_properties(1).total_memory/ (1024 ** 3)) * 100)

            print(f"Batch Size: {batch_size}, Runtime: {runtime:.2f}ms, Peak Memory: {peak_memory:.2f}GB, Throughput: {throughput:.2f} tokens/s, Cache Utilization: {cache_usages[-1]:.2f}%")

            batch_size += batch_step

        except RuntimeError as e:
            print(f"CUDA memory error at batch size {batch_size}. Stopping.")
            print(e)
            break

    # Restore original model settings
    model.config.use_cache = use_cache
    
    torch.cuda.empty_cache()
    gc.collect()  # Python garbage collection
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats for next model


    
    return {
        "batch_sizes": batch_sizes,
        "runtimes": runtimes,
        "memory_usages": memory_usages,
        "throughputs": throughputs,
        "cache_usages": cache_usages,
    }

from models.llama_kivi import LlamaForCausalLM_KIVI

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
config.use_flash = False
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cuda:1"},
)
# model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama', 
    model_max_length=128
)
tokenizer.pad_token = tokenizer.eos_token
torch.backends.cudnn.benchmark = True

# Run the benchmark
benchmark_results_ours = benchmark_model(model, tokenizer)

# Run the benchmark with regular llama
from transformers import LlamaForCausalLM
regular_config = LlamaConfig.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    config=regular_config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cuda:1"},
)
benchmark_results_llama = benchmark_model(model, tokenizer)

config.k_bits = 2
config.v_bits = 2
config.use_flash = True
model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:1"},
    )
benchmark_results_kivi = benchmark_model(model, tokenizer)

KIVI_avg = sum(benchmark_results_kivi['runtimes'])/len(benchmark_results_kivi['runtimes'])
our_avg = sum(benchmark_results_ours['runtimes'])/len(benchmark_results_ours['runtimes'])
llama_avg = sum(benchmark_results_llama['runtimes'])/len(benchmark_results_llama['runtimes'])

speedup_ours_KIVI = KIVI_avg/our_avg
speedup_ours_llama = llama_avg/our_avg
print(speedup_ours_KIVI, speedup_ours_llama)

# Store all dictionaries in a list
datasets = [benchmark_results_ours, benchmark_results_kivi, benchmark_results_llama]
colors = ['r', 'g', 'b']  # Colors for each dataset
labels = ['Ours', 'KIVI', 'Default llama']  # Labels for legend




# Define metric names, titles, and file names

metrics = ['cache_usages', 'throughputs', 'memory_usages', 'runtimes']
titles = ['Cache Utilization vs Batch Size', 'Throughput vs Batch Size', 
          'Memory Usage vs Batch Size', 'Runtime vs Batch Size']
y_labels = ['Cache Utilization (%)', 'Throughput (tokens/s)', 
            'Memory Usage (GB)', 'Runtime (ms)']
file_names = ['cache_utilization.png', 'throughput.png', 
              'memory_usage.png', 'runtime.png']

# Loop through each metric and generate a separate plot
for i, metric in enumerate(metrics):
    plt.figure(figsize=(8, 6))  # Create a new figure for each plot
    for j, data in enumerate(datasets):
        plt.plot(data['batch_sizes'], data[metric], marker='o', linestyle='-', color=colors[j], label=labels[j])
    
    plt.title(titles[i])
    plt.xlabel('Batch Size')
    plt.ylabel(y_labels[i])
    plt.legend()
    plt.grid(True)

    # Save the figure instead of showing it
    plt.savefig(f"images/{file_names[i]}")
    plt.close()  # Close the figure to free up memory

print("Plots saved successfully!")  # Confirmation message
