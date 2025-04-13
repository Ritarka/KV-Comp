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
from models.llama_kivi import LlamaForCausalLM_KIVI
from models.llama_kvcomp import LlamaForCausalLM_KVCOMP
from models.comp_replace import convert_kvcache_llama_heavy_recent
import traceback


import wandb


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader

def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'c4' in name:
        return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError

gpu_num = 0
device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model,'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}: {ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results


@torch.no_grad()
def evaluate(model, tokenizer):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: "24GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    # we don't need c4 now, can add it in later
    datasets = ["wikitext2", 
                #"c4"
                ]
    
    ppl_seqlen = 2048
    ppl_results = test_ppl(model, tokenizer, datasets, ppl_seqlen)
    for dataset in ppl_results:
        logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    eval_batch_size = 16
    eval_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

    ## Uncomment later -- Ritarka
    
    # import lm_eval
    # from lm_eval.models.huggingface import HFLM
    # from lm_eval.utils import make_table
    
    # task_list = eval_tasks.split(',')
    # model = HFLM(pretrained=model, batch_size=eval_batch_size)
    # task_manager = lm_eval.tasks.TaskManager()
    # results = lm_eval.simple_evaluate(
    # model=model,
    # tasks=task_list,
    # num_fewshot=0,
    # task_manager=task_manager,
    # )
    # logger.info(make_table(results))
    # total_acc = 0
    # for task in task_list:
    #     total_acc += results['results'][task]['acc,none']
    # logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    
    return results

@torch.no_grad()
def benchmark_model(model, tokenizer, dataset_name='wikitext2', max_batch_size=512, batch_step=4, cache=None):
    """Runs benchmarks for runtime, peak memory, throughput, and cache utilization."""
    
    print("Starting benchmark...")
    # evaluate(model, tokenizer)
    testloader = get_loaders(dataset_name, tokenizer, seed=0, seqlen=128, test_only=True)
    testenc = testloader.input_ids if hasattr(testloader, 'input_ids') else testloader

    # block_class_name = model.model.layers[0].__class__.__name__
    # device_map = infer_auto_device_map(model, max_memory={gpu_num: "24GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    # model = dispatch_model(model, device_map=device_map)

    
    batch_sizes, runtimes, memory_usages, throughputs, cache_usages = [], [], [], [], []

    batch_size = batch_step
    seqlen = 128
    nsamples = testenc.numel() // seqlen

    model.to(device)
    # model.config.use_cache = True
    model.eval()
    
    assert cache is None

    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True
    # ) as prof:
    count = 0
    primary_device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats(gpu_num)
    torch.cuda.synchronize(gpu_num)
    
    while batch_size <= max_batch_size:
        try:
            # torch.cuda.empty_cache()

            context = ['t,' * (160 // 2)] * batch_size
            # context = []
            # for _ in range(batch_size):
            #     string = 't,' * (160 // 2)
            #     context.append(string[:-1])


            batch = testenc[:, :seqlen].repeat(batch_size, 1).to(device)
            # inputs = tokenizer(batch, return_tensors="pt").to(device)
            inputs = tokenizer(context, return_tensors="pt").to(device)
            

            start_time = time.time()
            
            outputs = model.generate(**inputs, max_new_tokens=338)

            end_time = time.time()            
            torch.cuda.synchronize(gpu_num)

            runtime = (end_time - start_time) * 1000  # Convert to milliseconds
            peak_memory = torch.cuda.max_memory_allocated(gpu_num) / (1024 ** 3)  # Convert to GB
            num_tokens = batch.shape[0] * 320
            throughput = num_tokens / (runtime / 1000)
            
            total_gpu_memory = torch.cuda.get_device_properties(gpu_num).total_memory / (1024 ** 3)
            cache_utilization = (peak_memory / total_gpu_memory) * 100  # Cache utilization %


            # Store results
            batch_sizes.append(batch_size)
            runtimes.append(runtime)
            memory_usages.append(peak_memory)
            throughputs.append(throughput)
            cache_usages.append(cache_utilization) # NEED TO DEBUG
            
            
            # wandb.log({
            #     "batch_size": batch_size,
            #     "runtime_ms": runtime,
            #     "peak_memory_GB": peak_memory,
            #     "throughput_tokens_per_s": throughput,
            #     "cache_utilization_percent": cache_utilization
            # })


            print(f"Batch Size: {batch_size}, Runtime: {runtime:.2f}ms, Peak Memory: {peak_memory:.2f}GB, Throughput: {throughput:.2f} tokens/s, Cache Utilization: {cache_usages[-1]:.2f}%")

            batch_size += batch_step
            # count += 1
            # if count == 5:
            #     break

        except RuntimeError as e:
            print(f"CUDA memory error at batch size {batch_size}. Stopping.")
            traceback.print_exc()
            print(e)
            break

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # print(runtimes)
    return {
        "batch_sizes": batch_sizes,
        "runtimes": runtimes,
        "memory_usages": memory_usages,
        "throughputs": throughputs,
        "cache_usages": cache_usages,
    }

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 24
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-2-7b-hf'
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

# torch.backends.cudnn.benchmark = True

from transformers import LlamaForCausalLM
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.cache_utils import QuantizedCacheConfig, DynamicCache, CacheConfig
# from transformers.utils import is_optimum_quanto_available, is_hqq_available

# wandb.init(project="test_kvcache")

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama'
)

# del model
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.synchronize(gpu_num)

# print(f"Allocated memory: {torch.cuda.memory_allocated(gpu_num) / (1024 ** 3)} GB")
# print(f"Max allocated memory: {torch.cuda.max_memory_allocated(gpu_num) / (1024 ** 3)} GB")
# print(f"Total GPU memory: {torch.cuda.get_device_properties(gpu_num).total_memory / (1024 ** 3)} GB")

def benchmark_kivi():
    ########### BENCHMARK KIVI ###############
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.k_bits = K_BITS
    config.v_bits = V_BITS
    config.use_flash = True
    config.group_size = GROUP_SIZE
    config.residual_length = RESIDUAL_LENGTH

    model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=config,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": f"cuda:{gpu_num}"},
        )
    benchmark_results_kivi = benchmark_model(model, tokenizer)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_num)
    
    return benchmark_results_kivi

def benchmark_llama():
    ########### BENCHMARK LLAMA ###############
    config = LlamaConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": f"cuda:{gpu_num}"},
        # attn_implementation="flash_attention_2"
    )
    benchmark_results_llama = benchmark_model(model, tokenizer)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_num)

    return benchmark_results_llama

def benchmark_kvcomp():
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.k_bits = K_BITS
    config.v_bits = V_BITS
    config.use_flash = True
    config.group_size = GROUP_SIZE
    config.residual_length = RESIDUAL_LENGTH
    
    
    model = LlamaForCausalLM_KVCOMP.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": f"cuda:{gpu_num}"},
    )
    benchmark_results_ours = benchmark_model(model, tokenizer)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_num)

    return benchmark_results_ours

benchmark_results_ours = benchmark_kvcomp()
benchmark_results_kivi = benchmark_kivi()
benchmark_results_llama = benchmark_llama()

KIVI_avg = sum(benchmark_results_kivi['runtimes'])/len(benchmark_results_kivi['runtimes'])
our_avg = sum(benchmark_results_ours['runtimes'])/len(benchmark_results_ours['runtimes'])
llama_avg = sum(benchmark_results_llama['runtimes'])/len(benchmark_results_llama['runtimes'])

speedup_ours_KIVI = KIVI_avg/our_avg
speedup_ours_llama = llama_avg/our_avg
print(speedup_ours_KIVI, speedup_ours_llama)

# Store all dictionaries in a list
datasets = [benchmark_results_ours, benchmark_results_kivi, benchmark_results_llama]
colors = ['r', 'g', 'b']  # Colors for each dataset
labels = ['KVComp', 'KIVI', 'Default llama']  # Labels for legend




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
    plt.savefig(f"images/{metrics[i]}.png")
    plt.close()  # Close the figure to free up memory

print("Plots saved successfully!")  # Confirmation message
