import torch
import os
import argparse
import time
import matplotlib.pyplot as plt
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--use_kivi", type=int, choices=[0, 1], required=True, help="Use KIVI (1) or not (0)")
    return parser.parse_args()

def load_model(model_name, use_kivi, cache_dir="./cached_models"):
    config = LlamaConfig.from_pretrained(model_name)
    # Isolate to only one device to avoid error
    TARGET_GPU = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)
    device = torch.device(f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu")
    if use_kivi:
        config.k_bits = 2
        config.v_bits = 2
        config.use_flash = True
        config.group_size = 32
        config.residual_length = 128
        model = LlamaForCausalLM_KIVI.from_pretrained(
            model_name, config=config, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name, config=config, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto"
        )
    model.to(device)
    model.cuda().eval()
    return model, AutoTokenizer.from_pretrained(model_name, use_fast=False)

def benchmark(model, tokenizer, max_output_tokens=338):
    batch_size = 8
    prompt_length = 160
    batch_sizes, throughputs, memory_usages = [], [], []

    while True:
        try:
            context = ["t," * (prompt_length // 2) for _ in range(batch_size)]
            inputs = tokenizer(context, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            
            torch.cuda.synchronize()
            start_time = time.time()
            model.generate(**inputs, max_new_tokens=max_output_tokens)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time

            used_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            tokens_per_sec = (batch_size * max_output_tokens) / elapsed_time

            batch_sizes.append(batch_size)
            throughputs.append(tokens_per_sec)
            memory_usages.append(used_mem)
            print(f"Batch: {batch_size}, Throughput: {tokens_per_sec:.2f} tokens/sec, Memory: {used_mem:.2f} GB")
            
            batch_size += 8  # Increment batch_size
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Reached max batch size due to CUDA memory limit.")
                break
            else:
                raise e
    
    return batch_sizes, throughputs, memory_usages

def plot_results(batch_sizes, throughputs, memory_usages):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, throughputs, marker='o', label="Throughput (tokens/sec)")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, memory_usages, marker='o', color='r', label="Memory Usage (GB)")
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (GB)")
    plt.legend()
    
    plt.suptitle("LLM Performance Benchmark")
    plt.show()

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.use_kivi)
    batch_sizes, throughputs, memory_usages = benchmark(model, tokenizer)
    plot_results(batch_sizes, throughputs, memory_usages)

if __name__ == "__main__":
    main()
