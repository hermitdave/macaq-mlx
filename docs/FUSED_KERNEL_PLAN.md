# Fused Hadamard kernel — working plan (Route A selected)

Status: **Route A implemented and validated; opt-in, not enabled on the
published checkpoint.** On branch `feat/hadamard`. This file is the cross-session
handoff; update it as each step lands.

## Decision

Implement **Route A first**: fuse `x * t` and the 32-block orthonormal Walsh-
Hadamard transform into one `mx.fast.metal_kernel`, keep stock
`mx.quantized_matmul(mode="mxfp4", group_size=32, bits=4)`. Route B (full
scale+Hadamard+GEMM) stays deferred until Route A is measured.

## Integration point

No oMLX source changes. oMLX's LLM engine honours `config.json -> model_file`
with `trust_remote_code: true` and `model_type_override: "llm"`; the VLM engine
ignores `model_file`. The kernel must therefore live **inside `macaq_model.py`**,
which is loaded by `spec_from_file_location` with no package and no `sys.modules`
entry. Constraints: one self-contained module, inline Metal source string, no
relative imports, no `from __future__ import annotations`, no `mlx_lm.layers.*`.

`MACaQLinear.__call__` becomes:

```
if fused and kernel_ok:  x = fused_scale_hadamard(x, t)
else:                    x = x * t; x = mx.hadamard_transform(...)
return mx.quantized_matmul(...)
```

`config["macaq"]["fused"] = "rotate" | "none"` (default `"none"` so existing
checkpoints are unchanged). Weight format and `FORMAT_VERSION` stay the same.

## Kernel contract

- inputs: `x [M,K] bf16` (flattened from leading dims), `t [K] bf16`
- output: `[M,K] bf16`, same layout
- one threadgroup per row, 32 lanes per 32-block group
- per group: load `x*t` into fp32, butterfly `s = 1,2,4,8,16`, multiply by
  `1/sqrt(32)`, cast to bf16
- template ints: `T` (dtype), `K_DIM`
- grid `(32, M, 1)`, threadgroup `(32, 1, 1)`
- reference: `bench_online_transform.py::had_butterfly`, mflux
  `models/common/quantization/w8a8.py::_ROTATE_QUANTIZE`

## Steps / checklist

- [x] All prior work committed on `feat/hadamard` (03200a3 … 6998e76).
- [x] Add fused kernel + wrapper to `macaq_model.py`; add `fused` flag to
      `MACaQLinear` and `Model` factory; fallback to current path on compile error.
- [x] Unit test: kernel vs `mx.hadamard_transform(x*t)` on random bf16; max abs
      diff 0.03125, relative error ~0.0038 (tests/test_fused_kernel.py).
- [x] End-to-end layer test: fused vs stock layer output relative error < 0.01.
- [x] Bench sweep M = 1,4,16,64,256,1024,4096: fastest at M=1 (1.4×) and
      M=16–64 (1.2×); parity/slightly slower at M≥256. Full-model decode
      51.2 vs 50.0 tok/s — within noise.
- [x] Load check locally with `mlx_lm.load(..., trust_remote_code=True)` and
      generate; oMLX reload/load and chat completion also verified (53.0 tok/s).
- [x] Quality: same PPL 8.7089 ± 0.1829 on 32×512 WikiText for fused and stock.
- [ ] Decide whether to enable `macaq.fused = "rotate"` on a published variant.
      Current recommendation: leave default `"none"` until a larger win is
      demonstrated; Route B is the real speed project.
- [ ] Publish only after that decision: update `macaq_model.py` and `config.json`
      on the HF repo (shards byte-identical, Xet dedups). If materially different
      artifact, publish a new variant and link from the existing card.

## Deferred (Route B)

Full fused GEMV/GEMM with in-register E2M1 decode and E8M0 exponent. Likely needs
M-based dispatch (fused GEMV for M<=8, Route A + stock GEMM for prefill), mirroring
oMLX `oqh` (`qmv` decode vs `quantized_matmul` prefill). Risk: Python-defined
kernel unlikely to beat MLX steel GEMM at large M; `header` may not expose MLX's
internal `quantized_utils.h`. Estimate 1–2 weeks.

## Reference numbers to beat

- Local mlx-lm decode (988-token prompt, 256 tokens): this quant 50.0 tok/s;
  RCO 4.63 64.2 tok/s; unrotated MACaQ (earlier session) 70.7 tok/s.
- oMLX MTP-off decode: 47.9 tok/s (grafted-MTP sweep) / 41–49 tok/s (smoke).
- Standalone rotation overhead measured earlier: 0.246 ms at M=1, 6.785 ms at
  M=16384 on M3 Max.
