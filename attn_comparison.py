import math
import time

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from xformers.ops import fmha, LowerTriangularMask


def sdpa_benchmark(query, key, value, num_runs=10):
    """Benchmark torch sdpa attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            # Scaled Dot-Product Attention
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                is_causal=True,  # Enable internal causal masking
            )

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            # Scaled Dot-Product Attention
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                is_causal=True,  # Enable internal causal masking
            )

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return output, avg_time


def torch_benchmark(query, key, value, attn_mask, num_runs=10):
    """Benchmark plain torch attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                head_dim
            )
            attn_weights = attn_weights + attn_mask

            # upcast attention to fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            output = torch.matmul(attn_weights, value)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                head_dim
            )
            attn_weights = attn_weights + attn_mask

            # upcast attention to fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            output = torch.matmul(attn_weights, value)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return output, avg_time


def xformers_benchmark(query, key, value, attn_mask, num_runs=10):
    """Benchmark xformers attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            output = fmha.memory_efficient_attention_forward(
                query, key, value, attn_bias=attn_mask
            )

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            output = fmha.memory_efficient_attention_forward(
                query, key, value, attn_bias=attn_mask
            )

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return output, avg_time


def flash_attn_benchmark(query, key, value, num_runs=10):
    """Benchmark flash attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            output = flash_attn_func(q=query, k=key, v=value, causal=True)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            output = flash_attn_func(q=query, k=key, v=value, causal=True)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return output, avg_time


# Run the example
if __name__ == "__main__":
    # Input dimensions
    batch_size = 1
    seq_len = 8192
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads

    # Generate random inputs for query, key, and value
    query = torch.rand(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )
    key = torch.rand(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )
    value = torch.rand(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda"
    )

    # Attention mask (optional) - Mask future tokens in a causal setting
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.float16, device="cuda")
    temp_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda").tril(
        diagonal=0
    )
    attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_mask.to(torch.float16)

    # Runs of benchmark
    num_runs = 20

    sdpa_attn_output, sdpa_time = sdpa_benchmark(query, key, value, num_runs=num_runs)
    # print("SDPA Output:", sdpa_attn_output)
    print(f"SDPA Time: {sdpa_time * 1e3:.6f} ms")

    torch_attn_output, torch_time = torch_benchmark(
        query, key, value, attn_mask, num_runs=num_runs
    )
    torch.testing.assert_close(
        sdpa_attn_output, torch_attn_output, atol=1e-3, rtol=1e-3
    )
    # print("Torch Output:", torch_attn_output)
    print(f"Torch Time: {torch_time * 1e3:.6f} ms")

    # Arguments for xformers and flash attention:
    #     q: (batch_size, seqlen, nheads, headdim)
    #     k: (batch_size, seqlen, nheads_k, headdim)
    #     v: (batch_size, seqlen, nheads_k, headdim)_size, seqlen, nheads_k, headdim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    xformers_attn_output, xformers_time = xformers_benchmark(
        query, key, value, attn_mask=LowerTriangularMask(), num_runs=num_runs
    )
    torch.testing.assert_close(
        sdpa_attn_output.transpose(1, 2), xformers_attn_output, atol=1e-3, rtol=1e-3
    )
    # print("xformers Output:", xformers_attn_output)
    print(f"xformers Time: {xformers_time * 1e3:.6f} ms")

    flash_attn_output, flash_attn_time = flash_attn_benchmark(
        query, key, value, num_runs=num_runs
    )
    torch.testing.assert_close(
        sdpa_attn_output.transpose(1, 2), flash_attn_output, atol=1e-3, rtol=1e-3
    )
    # print("Flash Attention Output:", flash_attn_output)
    print(f"Flash Attention Time: {flash_attn_time * 1e3:.6f} ms")
