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
   ```
**Example Output**
```
SDPA Time: 30.217230 ms
Torch Time: 81.202161 ms
xformers Time: 7.641697 ms
Flash Attention Time: 7.457602 ms
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
The following benchmarks were conducted on an NVIDIA GeForce RTX 3050 (8 GB) GPU with the following parameters:
```
batch_size = 1
seq_len = 4096
embed_dim = 4096
num_heads = 32
head_dim = embed_dim // num_heads
```

| Attention Implementation | Average Runtime (ms) | Correctness Verified |
|---------------------------|----------------------|-----------------------|
| **Torch SDPA**            | ~30.2                | ✅                   |
| **Hand-crafted**          | ~81.2                | ✅                   |
| **xFormers**              | ~7.6                 | ✅                   |
| **FlashAttention**        | ~7.5                 | ✅                   |

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
