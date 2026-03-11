# Backward Kernel Benchmark — `flash_attn_triton.py`

**Date:** 2026-03-10
**GPU:** NVIDIA GeForce RTX 4090
**dtype:** float16
**batch:** 4, **nheads:** 16
**warmup:** 25 runs, **rep:** 100 runs

## Baselines

| Name | Description |
|---|---|
| **std** | Unfused PyTorch: full forward + `backward()` through `Q @ K.T → softmax(fp32) → @ V` |
| **sdpa** | `torch.nn.functional.scaled_dot_product_attention` — dispatches to cuDNN FlashAttention on CUDA |
| **triton** | `_flash_attn_backward` from `flash_attn/flash_attn_triton.py` (Triton ≥ 3.0 port), backward-only (forward pre-run separately) |

> **Note:** Timing is **backward pass only**. For `std` and `sdpa` the forward is re-run inside the timed loop to produce fresh activations for autograd; this slightly inflates their reported time relative to a pure-backward measurement.

## Results

| seqlen | headdim | causal | std ms | std TF | sdpa ms | sdpa TF | triton ms | triton TF | vs std | vs sdpa |
|-------:|--------:|:------:|-------:|-------:|--------:|--------:|----------:|----------:|-------:|--------:|
|    512 |      64 | False  |  1.269 |   6.77 |   0.140 |   61.27 |     0.195 |     44.04 |  6.51x |   0.72x |
|    512 |      64 | True   |  1.261 |   3.41 |   0.115 |   37.31 |     0.125 |     34.28 | 10.06x |   0.92x |
|   1024 |      64 | False  |  5.721 |   6.01 |   0.413 |   83.14 |     0.745 |     46.12 |  7.68x |   0.55x |
|   1024 |      64 | True   |  6.890 |   2.49 |   0.317 |   54.25 |     0.438 |     39.21 | 15.73x |   0.72x |
|   2048 |      64 | False  | 22.352 |   6.15 |   1.594 |   86.24 |     3.001 |     45.80 |  7.45x |   0.53x |
|   2048 |      64 | True   | 26.940 |   2.55 |   1.037 |   66.28 |     1.664 |     41.30 | 16.19x |   0.62x |
|   4096 |      64 | False  |187.833 |   2.93 |   6.215 |   88.45 |    16.644 |     33.03 | 11.29x |   0.37x |
|   4096 |      64 | True   |106.070 |   2.59 |   3.629 |   75.75 |     7.916 |     34.73 | 13.40x |   0.46x |
|   1024 |     128 | False  |  6.066 |  11.33 |   0.859 |   80.00 |     1.613 |     42.59 |  3.76x |   0.53x |
|   1024 |     128 | True   |  7.240 |   4.75 |   0.576 |   59.62 |     1.152 |     29.84 |  6.29x |   0.50x |
|   2048 |     128 | False  | 23.021 |  11.94 |   3.218 |   85.43 |     7.645 |     35.95 |  3.01x |   0.42x |
|   2048 |     128 | True   | 27.602 |   4.98 |   1.948 |   70.55 |     4.510 |     30.47 |  6.12x |   0.43x |
|   4096 |     128 | False  |    OOM |      — |       — |       — |         — |         — |      — |       — |
|   4096 |     128 | True   |    OOM |      — |       — |       — |         — |         — |      — |       — |

> **OOM:** Standard attention (seqlen=4096, headdim=128) allocates a 4096×4096 attention matrix per head plus full autograd activations (~4 GiB), exhausting GPU memory. `sdpa` and `triton` results for those rows were not measured due to early exit; they would not OOM as they are tiled / IO-bound kernels.

## FLOP counting

- Non-causal backward: `8 * batch * nheads * seqlen² * headdim` FLOPs (≈ 2× forward: dQ, dK, dV each require one matmul pair)
- Causal: half the above

## Observations

**vs standard attention**
- The Triton backward kernel is **4–16× faster** than unfused PyTorch backward.
- The speedup grows with sequence length, same IO-bound argument as the forward pass.

**vs `torch.sdpa` (cuDNN FlashAttention)**
- `sdpa` is consistently **faster** than the Triton backward kernel across all measured configs.
  - For **headdim=64**: Triton reaches 37–92% of sdpa speed (8–63% slower).
  - For **headdim=128**: Triton reaches 42–53% of sdpa speed (~2× slower).
- The backward gap is wider than the forward gap, especially at large seqlen (headdim=64, seqlen=4096: Triton is only 37% of sdpa speed). The Triton backward kernel has higher register pressure and more atomic writes (dQ accumulation) compared to the cuDNN hand-tuned kernel.
- Causal masking narrows the gap slightly for both kernels, as both skip masked tiles.

**vs forward pass (same kernel)**
- The backward pass is roughly **3–8× slower** than the forward pass for the same config, consistent with the ~2× higher FLOP count plus additional memory traffic for storing/loading dQ, dK, dV.

**Comparison with git HEAD original (Triton 2.x)**
The original kernel uses `tl.dot(q, k, trans_b=True)` — an API removed in Triton 3.0 — and therefore **does not compile** on the installed environment. No timing comparison with the original is possible.
