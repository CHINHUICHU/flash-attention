# Changes from the original script:
#   - Replaced `triton.ops.flash_attention` with our local `flash_attn_triton_og` (renamed to TritonOG).
#     TritonOG only supports causal=True, so non-causal configs now record NaN instead of being attempted.
#     The old Triton block ran both sequence_parallel={False,True} and picked the faster backward;
#     the new block simply calls attention_triton_og with a single scale argument.
#   - Removed unused imports (pickle, torch.nn, repeat, benchmark_all, benchmark_forward, benchmark_backward).
#   - Added explanatory comments inside flops() for the FLOP formula and backward multipliers.
#   - Print format now shows latency in ms alongside TFLOPs/s (was TFLOPs/s only).
#   - Removed the dead pickle dump at the end.
#
# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import math
import torch
import torch.nn.functional as F

from einops import rearrange

from flash_attn.utils.benchmark import benchmark_fwd_bwd

from flash_attn import flash_attn_qkvpacked_func

try:
    from flash_attn.flash_attn_triton_og import attention as attention_triton_og
except ImportError:
    attention_triton_og = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    # 2 matmuls (QK^T and PV), each 2*B*S^2*H*D FLOPs => 4*B*S^2*H*D total.
    # Softmax FLOPs (3*B*H*S^2) are excluded — negligible vs matmuls at large S and D.
    # Causal masking halves the work (roughly half the attention matrix is computed).
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    # 2.5x and 3.5x are rough empirical multipliers, not exact derivations.
    # Backward requires dQ, dK, dV (each ~1 fwd matmul) plus softmax backward overhead,
    # which lands around 2.5x fwd in practice.
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # Adding is faster than masked_fill_
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = (["Flash2", "Pytorch"]
           + (["TritonOG"] if attention_triton_og is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else []))

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim

            # FlashAttention 2
            if "Flash2" in methods:
                qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim,
                                  device=device, dtype=dtype, requires_grad=True)
                f, b = time_fwd_bwd(
                    flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal,
                    repeats=repeats, verbose=False
                )
                time_f[config, "Flash2"] = f
                time_b[config, "Flash2"] = b

            # PyTorch baseline
            if "Pytorch" in methods:
                try:
                    # fresh tensor avoids grad-history reuse issues
                    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim,
                                      device=device, dtype=dtype, requires_grad=True)
                    f, b = time_fwd_bwd(
                        attention_pytorch, qkv, dropout_p, causal=causal,
                        repeats=repeats, verbose=False
                    )
                except Exception:
                    f, b = float('nan'), float('nan')
                time_f[config, "Pytorch"] = f
                time_b[config, "Pytorch"] = b

            # Triton OG (always causal — skip non-causal configs)
            if "TritonOG" in methods and attention_triton_og is not None:
                if not causal:
                    time_f[config, "TritonOG"] = float('nan')
                    time_b[config, "TritonOG"] = float('nan')
                else:
                    q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim,
                                           device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
                    try:
                        f, b = time_fwd_bwd(
                            attention_triton_og, q, k, v, headdim**(-0.5),
                            repeats=repeats, verbose=False
                        )
                    except Exception:
                        f, b = float('nan'), float('nan')
                    time_f[config, "TritonOG"] = f
                    time_b[config, "TritonOG"] = b

            # xFormers CUTLASS
            if "xformers.c" in methods and xops is not None:
                q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim,
                                       device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
                )
                time_f[config, "xformers.c"] = f
                time_b[config, "xformers.c"] = b

            # xFormers Flash
            if "xformers.f" in methods and xops is not None:
                q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim,
                                       device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                )
                time_f[config, "xformers.f"] = f
                time_b[config, "xformers.f"] = b

            # Report
            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                if (config, method) not in time_f or (config, method) not in time_b:
                    continue
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} "
                    f"fwd: {time_f[config, method]*1e3:.3f}ms ({speed_f[config, method]:.2f} TFLOPs/s), "
                    f"bwd: {time_b[config, method]*1e3:.3f}ms ({speed_b[config, method]:.2f} TFLOPs/s), "
                    f"fwd+bwd: {time_f_b[config, method]*1e3:.3f}ms ({speed_f_b[config, method]:.2f} TFLOPs/s)"
                )
