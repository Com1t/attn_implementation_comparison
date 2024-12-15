import time

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from flash_attn import flash_attn_func
from xformers.ops import fmha, LowerTriangularMask

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
)


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


def xformers_benchmark(query, key, value, num_runs=10):
    """Benchmark xformers attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            output = fmha.memory_efficient_attention_forward(
                query, key, value, attn_bias=LowerTriangularMask()
            )

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            output = fmha.memory_efficient_attention_forward(
                query, key, value, attn_bias=LowerTriangularMask()
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


def dist_weight_init(local_weight, master_weight=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        master_weight_list = torch.split(
            master_weight, master_weight.shape[2] // world_size, dim=2
        )
        scatter_list = [t.contiguous() for t in master_weight_list]
    else:
        scatter_list = None
    dist.scatter(local_weight, scatter_list, src=0)


def verify_correctness(gather_list, partial_attn_output, master_attn_output=None):
    """Verify the correctness of the gathered attention output"""
    rank = dist.get_rank()
    dist.gather(partial_attn_output, gather_list, dst=0)
    if rank == 0:
        final_attn_output = torch.cat(gather_list, dim=2)
        if torch.allclose(final_attn_output, master_attn_output, atol=1e-3, rtol=1e-3):
            print(
                "Verification passed: Parallel and local Attention outputs are close."
            )
        else:
            print("Verification failed: Outputs differ significantly.")


# Run the example
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    initialize_model_parallel(world_size)

    device = torch.device(f"cuda:{rank}")

    # Input dimensions
    batch_size = 1
    seq_len = 4096
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads

    query = torch.zeros(
        batch_size,
        seq_len,
        num_heads // world_size,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    key = torch.zeros(
        batch_size,
        seq_len,
        num_heads // world_size,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value = torch.zeros(
        batch_size,
        seq_len,
        num_heads // world_size,
        head_dim,
        dtype=torch.float16,
        device=device,
    )

    # Generate random inputs for query, key, and value
    if rank == 0:
        master_query = torch.rand(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device
        )
        master_key = torch.rand(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device
        )
        master_value = torch.rand(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device
        )
        dist_weight_init(query, master_query)
        dist_weight_init(key, master_key)
        dist_weight_init(value, master_value)
    else:
        dist_weight_init(query)
        dist_weight_init(key)
        dist_weight_init(value)

    # Runs of benchmark
    num_runs = 20

    xformers_attn_output, xformers_time = xformers_benchmark(
        query, key, value, num_runs=num_runs
    )
    print(f"[Rank {rank}] xformers Time: {xformers_time * 1e3:.6f} ms")

    flash_attn_output, flash_attn_time = flash_attn_benchmark(
        query, key, value, num_runs=num_runs
    )
    print(f"[Rank {rank}] Flash Attention Time: {flash_attn_time * 1e3:.6f} ms")

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    sdpa_attn_output, sdpa_time = sdpa_benchmark(query, key, value, num_runs=num_runs)
    print(f"[Rank {rank}] SDPA Time: {sdpa_time * 1e3:.6f} ms")

    sdpa_attn_output = sdpa_attn_output.transpose(1, 2).contiguous()

    if rank == 0:
        gather_list = [
            torch.zeros(
                batch_size,
                seq_len,
                num_heads // world_size,
                head_dim,
                dtype=torch.float16,
                device=device,
            )
            for i in range(world_size)
        ]

        master_query = master_query.transpose(1, 2).contiguous()
        master_key = master_key.transpose(1, 2).contiguous()
        master_value = master_value.transpose(1, 2).contiguous()
        master_attn_output, _ = sdpa_benchmark(master_query, master_key, master_value)
        master_attn_output = master_attn_output.transpose(1, 2).contiguous()
        verify_correctness(gather_list, sdpa_attn_output, master_attn_output)
        verify_correctness(gather_list, xformers_attn_output, master_attn_output)
        verify_correctness(gather_list, flash_attn_output, master_attn_output)
    else:
        gather_list = None
        verify_correctness(gather_list, sdpa_attn_output)
        verify_correctness(gather_list, xformers_attn_output)
        verify_correctness(gather_list, flash_attn_output)
