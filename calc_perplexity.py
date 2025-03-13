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
    # device_map = infer_auto_device_map(model, max_memory={i: "24GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    # model = dispatch_model(model, device_map=device_map)
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

# Isolate to only one device to avoid error
TARGET_GPU = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)
device = torch.device(f"cuda:{TARGET_GPU}" if torch.cuda.is_available() else "cpu")

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
# model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama'
)

torch.backends.cudnn.benchmark = True
results = evaluate(model, tokenizer)
print(results)