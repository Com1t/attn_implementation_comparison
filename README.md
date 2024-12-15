# Attention Implementation Comparison

This repository provides a benchmark and comparison of different attention implementations in PyTorch, including:

- **Torch Scaled Dot-Product Attention (SDPA)**
- **Hand-crafted Attention**
- **xFormers Memory-Efficient Attention**
- **FlashAttention**

The benchmarks evaluate runtime performance and correctness of each implementation.

---

## Features

1. **Performance Benchmarking**:
   - Measures the average runtime of each attention implementation over multiple runs.
   
2. **Correctness Validation**:
   - Ensures that all implementations produce similar outputs within a specified tolerance.

3. **Customizable Input Dimensions**:
   - Easily adjust batch size, sequence length, embedding dimensions, and more.

4. **Tensor Parallel**:
   - Implements tensor parallelism by distributing attention computation across GPUs, partitioning the query, key, and value tensors along the head dimension, and performing localized computations. This enables efficient scaling for large sequence lengths and embedding dimensions while maintaining model accuracy.

---

## Requirements

- Python 3.8 or higher
- PyTorch 2.5 or higher
- Additional dependencies:
  - `flash_attn` (`pip install flash-attn`)
  - `xformers` (`pip install xformers`)

---

## Usage

**Run the Benchmark**
   ```bash
   git clone https://github.com/YourUsername/attention-comparison.git
   cd attention-comparison
   python attn_comparison.py
   torchrun --nproc-per-node 4 ./dist_attn_comparison.py
   ```
**Example Output**
Local
```
SDPA Time: 30.217230 ms
Torch Time: 81.202161 ms
xformers Time: 7.641697 ms
Flash Attention Time: 7.457602 ms
```

Distributed (Check the longest)
```
[Rank 3] xformers Time: 0.184095 ms
[Rank 2] xformers Time: 0.184822 ms
[Rank 3] Flash Attention Time: 0.096738 ms
[Rank 2] Flash Attention Time: 0.097036 ms
[Rank 1] xformers Time: 0.405443 ms
[Rank 3] SDPA Time: 0.020730 ms
[Rank 2] SDPA Time: 0.020969 ms
[Rank 1] Flash Attention Time: 0.220585 ms
[Rank 0] xformers Time: 0.580966 ms
[Rank 1] SDPA Time: 0.038075 ms
[Rank 0] Flash Attention Time: 0.306368 ms
[Rank 0] SDPA Time: 0.324106 ms
```

---

**Adjust Benchmark Parameters**
Modify the script to change parameters like 
- batch size
- sequence length
- embedding dimensions
- number of heads
- number of runs

---

## Benchmark Methodology

1. **Warm-up**:
   - Each implementation runs a few iterations to ensure the GPU is warmed up and to eliminate cold-start effects.

2. **Execution Time Measurement**:
   - Synchronization points are added to ensure accurate timing of each implementation's runtime. The average time is computed over multiple runs.

3. **Correctness Check**:
   - The outputs of all implementations are compared against PyTorch's Scaled Dot-Product Attention (SDPA) as the baseline. The comparison ensures numerical consistency within a reasonable tolerance.

---

## Code Overview

### Key Functions

- **`sdpa_benchmark`**:
  Benchmarks PyTorch's Scaled Dot-Product Attention using `torch.nn.functional.scaled_dot_product_attention`.

- **`torch_benchmark`**:
  Benchmarks a manually implemented attention mechanism using matrix multiplications, scaling, and softmax.

- **`xformers_benchmark`**:
  Benchmarks the xFormers library's memory-efficient attention implementation.

- **`flash_attn_benchmark`**:
  Benchmarks FlashAttention, designed for high performance on long sequences and large batch sizes.

---

## Results Summary
The following benchmarks were conducted on an NVIDIA A100 (80 GB) GPU with the following parameters:
```
batch_size = 1
seq_len = 8192
embed_dim = 4096
num_heads = 32
head_dim = embed_dim // num_heads
```

| Attention Implementation | Average Runtime (ms) | Correctness Verified |
|---------------------------|----------------------|-----------------------|
| **Torch SDPA**            | ~3.0                 | ✅                   |
| **Hand-crafted**          | ~44.4                | ✅                   |
| **xFormers**              | ~3.0                 | ✅                   |
| **FlashAttention**        | ~2.9                 | ✅                   |

---

## Contributions

Feel free to contribute by:
- Adding new attention implementations.
- Improving benchmark accuracy and methodology.
- Reporting issues or suggesting enhancements.

---

## Acknowledgments

- **[FlashAttention](https://github.com/HazyResearch/flash-attention)**
- **[xFormers](https://github.com/facebookresearch/xformers)**
- The PyTorch community for the Scaled Dot-Product Attention implementation.
