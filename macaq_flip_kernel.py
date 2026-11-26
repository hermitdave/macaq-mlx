"""Fused MACaQ Flip-Refinement step on MLX Metal.

Quantization-time only. Adapted from `rgarcia-fx/macaq-mlx/macaq_mlx.py`
(`macaq_flip_step`), which is based on Huawei CSL MACaQ (Apache-2.0).
Kept separate from the runtime checkpoint shim: no inference path imports this.

STATUS: unvalidated draft. Do not enable in `macaq_hadamard.py` until the
code/objective equivalence test against `macaq_refine.flip_refine` passes.

Semantics match `macaq_refine.flip_refine`: per output row, choose the best
single E2M1 level flip under the rotated Hessian and update the gradient.
"""

from __future__ import annotations

import mlx.core as mx

from macaq_refine import SG, dequant, scale_elem

BLOCK = 32
_FLIP_TG = 256

_FLIP_HEADER = """
constant float MACAQ_SGV[15] = {
    -6.f, -4.f, -3.f, -2.f, -1.5f, -1.f, -0.5f, 0.f,
     0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f};
"""

_FLIP_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const uint TGS = threads_per_threadgroup.x;
    const int n = params[0];
    const float min_gain = min_gainp[0];
    const device uchar* Lr = L + (size_t)row * n;
    const device float* gr = g + (size_t)row * n;
    const device float* sr = sb + (size_t)row * (n / 32);

    float best = -INFINITY; int bj = 0x7fffffff; float bd = 0.f; int bdir = 0;
    for (int c = tid; c < n; c += TGS) {
        int l = Lr[c];
        float s = sr[c >> 5], gc = gr[c], h = hd[c];
        float up = (l < 14) ? (MACAQ_SGV[min(l + 1, 14)] - MACAQ_SGV[l]) * s : 0.f;
        float dn = (l > 0)  ? (MACAQ_SGV[max(l - 1, 0)] - MACAQ_SGV[l]) * s : 0.f;
        float gu = 2.f * up * gc - up * up * h;
        float gd = 2.f * dn * gc - dn * dn * h;
        bool use_up = gu >= gd;
        float gain = use_up ? gu : gd;
        if (gain > best) { best = gain; bj = c; bd = use_up ? up : dn; bdir = use_up ? 1 : -1; }
    }
    threadgroup float sg[256]; threadgroup int sj[256];
    threadgroup float sd[256]; threadgroup int sdir[256];
    sg[tid] = best; sj[tid] = bj; sd[tid] = bd; sdir[tid] = bdir;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = TGS / 2; w > 0; w >>= 1) {
        if (tid < w) {
            float og = sg[tid + w]; int oj = sj[tid + w];
            if (og > sg[tid] || (og == sg[tid] && oj < sj[tid])) {
                sg[tid] = og; sj[tid] = oj; sd[tid] = sd[tid + w]; sdir[tid] = sdir[tid + w];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int j = sj[0]; float delta = sd[0]; int dir = sdir[0];
    bool accept = (sg[0] > min_gain) && (delta != 0.f);
    if (!accept) { delta = 0.f; dir = 0; }
    device uchar* Lo = L_out + (size_t)row * n;
    device float* go = g_out + (size_t)row * n;
    const device float* Hj = Hf + (size_t)j * n;
    for (int c = tid; c < n; c += TGS) {
        go[c] = accept ? gr[c] - delta * Hj[c] : gr[c];
        Lo[c] = (uchar)((int)Lr[c] + ((c == j) ? dir : 0));
    }
    if (tid == 0) ok[row] = accept ? 1 : 0;
"""

_kernel = None


def _flip_step(L8, g, sb, hd, Hf, min_gain=1e-12):
    """One fused greedy flip sweep. Inputs are 2-D [rows, width]."""
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="macaq_flip_step",
            input_names=["L", "g", "sb", "hd", "Hf", "params", "min_gainp"],
            output_names=["L_out", "g_out", "ok"],
            header=_FLIP_HEADER,
            source=_FLIP_SRC,
        )
    m, n = g.shape
    return _kernel(
        inputs=[
            L8, g, sb, hd, Hf,
            mx.array([n], dtype=mx.int32),
            mx.array([min_gain], dtype=mx.float32),
        ],
        template=[],
        grid=(m * _FLIP_TG, 1, 1),
        threadgroup=(_FLIP_TG, 1, 1),
        output_shapes=[L8.shape, g.shape, (m,)],
        output_dtypes=[mx.uint8, mx.float32, mx.uint8],
    )


def flip_refine_metal(sidx, log2s, W, Hf, n_iters, min_gain=1e-12,
                      recompute_every=0):
    """Drop-in for `macaq_refine.flip_refine` on 2-D dense layers.

    MoE experts (3-D) are not supported yet; run them per expert or use the
    pure-MLX path. `min_gain` matches the current absolute stop threshold.
    """
    if sidx.ndim != 2 or W.ndim != 2:
        raise NotImplementedError("fused flip step supports 2-D dense layers only")

    n = W.shape[1]
    sb = mx.power(2.0, log2s.astype(mx.float32))
    hd = mx.diagonal(Hf)
    L8 = sidx.astype(mx.uint8)
    g = (W - dequant(sidx, log2s, n)) @ Hf
    mx.eval(L8, g, sb)

    sweeps = 0
    for k in range(n_iters):
        L8, g, ok = _flip_step(L8, g, sb, hd, Hf, min_gain)
        if recompute_every and (k + 1) % recompute_every == 0:
            g = (W - dequant(L8.astype(mx.int32), log2s, n)) @ Hf
        mx.eval(L8, g, ok)
        sweeps += 1
        if not ok.any().item():
            break
    return L8.astype(mx.int32), sweeps
