import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from xformers.ops import fmha, LowerTriangularMask


def attn_comparison(query,
                    key,
                    value,
                    attn_mask):

    # Scaled Dot-Product Attention
    sdpa_attn_output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=False  # Enable causal masking explicitly
    )
    # print("SDPA Output:", attn_output)

    # hand craft
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = attn_weights + attn_mask

    # upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    hc_attn_output = torch.matmul(attn_weights, value)
    
    # print("Hand Craft Output:", hc_attn_output)
    torch.testing.assert_close(sdpa_attn_output, hc_attn_output, atol=1e-3, rtol=1e-3)

    # Arguments for xformers and flash attention:
    #     q: (batch_size, seqlen, nheads, headdim)
    #     k: (batch_size, seqlen, nheads_k, headdim)
    #     v: (batch_size, seqlen, nheads_k, headdim)_size, seqlen, nheads_k, headdim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    xformers_attn_output = fmha.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=LowerTriangularMask()
    )
    # print("xformers Output:", xformers_attn_output)
    torch.testing.assert_close(sdpa_attn_output.transpose(1, 2), xformers_attn_output, atol=1e-3, rtol=1e-3)

    flash_attn_output = flash_attn_func(
        q=query,
        k=key,
        v=value,
        causal=True
    )
    # print("Flash Attention Output:", flash_attn_output)
    torch.testing.assert_close(sdpa_attn_output.transpose(1, 2), flash_attn_output, atol=1e-3, rtol=1e-3)


# Run the example
if __name__ == "__main__":
    # Input dimensions
    batch_size = 1
    seq_len = 128
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads

    # Generate random inputs for query, key, and value
    query = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device="cuda")
    key = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device="cuda")
    value = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device="cuda")

    # Attention mask (optional) - Mask future tokens in a causal setting
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.float16, device="cuda")
    temp_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda").tril(diagonal=0)
    attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_mask.to(torch.float16)

    attn_comparison(query,
                    key,
                    value,
                    attn_mask)
