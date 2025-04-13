import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from nvidia import nvcomp
import time

from flash_attn import flash_attn_qkvpacked_func
from inspect import signature

_CONFIG_FOR_DOC = "LlamaConfig"

time_taken = [0 for i in range(4)]
codec = nvcomp.Codec(algorithm="ANS", bitstream_kind=nvcomp.BitstreamKind.NVCOMP_NATIVE)

def timer(argument):
    def timer_d(function):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            result = function(*args, **kwargs)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            time_taken[argument] += elapsed_time
            
            return result
        return wrapper
    return timer_d

# @timer(0)
def quantize(array: torch.Tensor, num_bits: int):
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
def compress_with_nvcomp(tensor: torch.Tensor):
    if tensor.device.type != "cuda":
        tensor = tensor.to("cuda", non_blocking=True)

    # tensor = (tensor * 255).to(torch.uint8)  # maybe better compression for low-precision data
    nvarr_txt_d = nvcomp.as_array(tensor)
    return codec.encode(nvarr_txt_d)

# @timer(3)
def decompress_with_nvcomp(compressed_arr):
    decoded = codec.decode(compressed_arr)  # Avoid unnecessary numpy conversion
    return torch.as_tensor(decoded, dtype=torch.float16, device="cuda")  # Direct to FP16


from transformers import Cache

class CompressedKVCache(Cache):
    def __init__(self, key=None, value=None, scale_k=None, scale_v=None, offset_k=None, offset_v=None):
        self.key = key
        self.value = value
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.offset_k = offset_k
        self.offset_v = offset_v

    def update(self, key_states, value_states):
        """ Update the cache with new KV states """
        q_k, scale_k, offset_k = quantize(key_states, num_bits=2)
        q_v, scale_v, offset_v = quantize(value_states, num_bits=2)
        compressed_k = compress_with_nvcomp(q_k)
        compressed_v = compress_with_nvcomp(q_v)

        return CompressedKVCache(compressed_k, compressed_v, scale_k, scale_v, offset_k, offset_v)


class LlamaAttention_Compression(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        # self.rotary_emb.to(device="cuda")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]        
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            # **Decompress and dequantize past key_value**
            past_k_quantized, past_v_quantized, past_scale_k, past_scale_v, past_offset_k, past_offset_v = past_key_value
            
            past_k = dequantize(decompress_with_nvcomp(past_k_quantized), past_scale_k, past_offset_k)
            past_v = dequantize(decompress_with_nvcomp(past_v_quantized), past_scale_v, past_offset_v)

            # Concatenate past and current key-value states
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        # **Quantize and compress KV cache for storage**
        q_k, scale_k, offset_k = quantize(key_states, num_bits=2)
        q_v, scale_v, offset_v = quantize(value_states, num_bits=2)
        compressed_k = compress_with_nvcomp(q_k)
        compressed_v = compress_with_nvcomp(q_v)

        past_key_value = (compressed_k, compressed_v, scale_k, scale_v, offset_k, offset_v) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
            
        # print(time_taken)

        return attn_output, attn_weights, past_key_value
    
    
class LlamaAttention_Compression_Flash(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False).half()
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False).half()
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False).half()
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False).half()
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        # self.rotary_emb.to(device="cuda")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CompressedKVCache] = None,  # Use new cache format
        output_attentions: bool = False,
        use_cache: bool = True,
        cache_position=None,
        position_embeddings=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[CompressedKVCache]]:
        print("Using Flash Attention with Compression")
        bsz, q_len, _ = hidden_states.size()
        
        print(f"hidden_states dtype: {hidden_states.dtype}")
        print(f"q_proj dtype: {self.q_proj.weight.dtype}")


        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k = dequantize(decompress_with_nvcomp(past_key_value.key), past_key_value.scale_k, past_key_value.offset_k)
            past_v = dequantize(decompress_with_nvcomp(past_key_value.value), past_key_value.scale_v, past_key_value.offset_v)

            key_states = torch.cat([past_k, key_states], dim=1)
            value_states = torch.cat([past_v, value_states], dim=1)

        qkv = torch.stack([query_states, key_states, value_states], dim=2)

        attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        print(f"past_key_value type: {type(past_key_value)}")
        print(f"Cache contents: {past_key_value}")
        print(dir(past_key_value))
        print(signature(past_key_value.update))

        # Store the new KV cache
        if use_cache:
            past_key_value = past_key_value.update(key_states, value_states) if past_key_value else CompressedKVCache().update(key_states, value_states)

        return attn_output, None, past_key_value

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        # self.rotary_emb.to(device="cuda")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class RotaryEmbedding(nn.Module):
    """ Applies rotary positional embeddings to queries and keys """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        q = self.apply_rotary(q, emb)
        k = self.apply_rotary(k, emb)
        return q, k

    @staticmethod
    def apply_rotary(x, emb):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        emb1, emb2 = emb[..., 0::2], emb[..., 1::2]
        x_rot = torch.cat([x1 * emb1.cos() - x2 * emb2.sin(),
                           x1 * emb1.sin() + x2 * emb2.cos()], dim=-1)
        return x_rot

def convert_kvcache_llama_heavy_recent(model, config):
    converts = 0
    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name], sub = convert_kvcache_llama_heavy_recent(module, config)
            converts += sub

        if isinstance(module, LlamaSdpaAttention):
            model._modules[name] = LlamaAttention_Compression_Flash(config)
            converts += 1
            # model._modules[name] = LlamaAttention(config)

    return model, converts