import math
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from xformers.ops import fmha, LowerTriangularMask


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def sdpa_adpater(query, key, value):
    """Adapter for torch sdpa attention module"""
    output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=True,  # Enable causal masking explicitly
    )

    return output


@torch.compile
def torch_sdpa(query, key, value, attn_mask=None, is_causal=False):
    """Perform sdpa attention with given inputs"""
    L, S = query.size(-2), key.size(-2)
    head_dim = query.size(-1)

    # Scaled Dot-Product Attention
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    # This one is faster in some cases
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weights += attn_bias

    # Softmax with numerical stability
    max_vals = torch.max(attn_weights, dim=-1, keepdim=True).values
    exp_weights = torch.exp(attn_weights - max_vals)
    sum_exp_weights = torch.sum(exp_weights, dim=-1, keepdim=True)
    lse = torch.log(sum_exp_weights)
    attn_weights = exp_weights / sum_exp_weights

    output = torch.matmul(attn_weights, value)
    return output, lse


def torch_sdpa_adapter(query, key, value):
    """Adapter for torch sdpa attention module"""
    # Attention mask (optional) - Mask future tokens in a causal setting
    # seq_len = query.size(-2)
    # attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.float16, device="cuda")
    # temp_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda").tril(
    #     diagonal=0
    # )
    # attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    # attn_mask.to(torch.float16)

    output, _ = torch_sdpa(query=query, key=key, value=value, is_causal=True)

    return output


def xformers_adpater(query, key, value):
    """Adapter for xformers attention module"""
    output = fmha.memory_efficient_attention_forward(
        query=query, key=key, value=value, attn_bias=LowerTriangularMask()
    )
    return output


def flash_attn_adpater(query, key, value):
    """Adapter for flash attention module"""
    output = flash_attn_func(q=query, k=key, v=value, causal=True)
    return output


def benchmark(attn_fn, query, key, value, warmup_runs=3, num_runs=10):
    """Benchmark torch sdpa attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            output = attn_fn(
                query=query,
                key=key,
                value=value,
            )

    total_time = 0
    # Benchmark
    with torch.no_grad():
        for _ in range(num_runs):
            output, elapsed_time = timed(
                lambda: attn_fn(
                    query=query,
                    key=key,
                    value=value,
                )
            )
            total_time += elapsed_time

    avg_time = total_time / num_runs
    return output, avg_time


def main():
    # Input dimensions
    batch_size = 1
    seq_len = 8192
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads

    # Runs of benchmark
    num_runs = 20
    warmup_runs = 3

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

    sdpa_attn_output, sdpa_time = benchmark(
        sdpa_adpater, query, key, value, warmup_runs=warmup_runs, num_runs=num_runs
    )
    # print("SDPA Output:", sdpa_attn_output)
    print(f"SDPA Time: {sdpa_time:.6f} ms")

    golden_output = sdpa_attn_output

    torch_attn_output, torch_time = benchmark(
        torch_sdpa_adapter,
        query,
        key,
        value,
        warmup_runs=warmup_runs,
        num_runs=num_runs,
    )
    torch.testing.assert_close(golden_output, torch_attn_output, atol=1e-3, rtol=1e-3)
    # print("Torch Output:", torch_attn_output)
    print(f"Torch Time: {torch_time:.6f} ms")

    # Arguments for xformers and flash attention:
    #     q: (batch_size, seqlen, nheads, headdim)
    #     k: (batch_size, seqlen, nheads_k, headdim)
    #     v: (batch_size, seqlen, nheads_k, headdim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    golden_output = golden_output.transpose(1, 2)

    attn_module = {"xformers": xformers_adpater, "flash_attn": flash_attn_adpater}
    for name, attn_fn in attn_module.items():
        attn_output, attn_time = benchmark(
            attn_fn, query, key, value, warmup_runs=warmup_runs, num_runs=num_runs
        )
        torch.testing.assert_close(golden_output, attn_output, atol=1e-3, rtol=1e-3)
        print(f"{name} Time: {attn_time:.6f} ms")


# Run the example
if __name__ == "__main__":
    main()
