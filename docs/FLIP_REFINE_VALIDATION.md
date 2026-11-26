# Fused flip refinement validation

Source of fused kernel: `rgarcia-fx/macaq-mlx` (`macaq_flip_step`, Apache-2.0),
adapted into `macaq_flip_kernel.py`. Algorithm credit: Huawei CSL MACaQ.

## Why this model

`XHToken/Spark-X2.6-1.7B` is a fresh BF16 Transformers checkpoint (28 layers,
hidden 2048, intermediate 6656). Its `spark2_5` architecture is not supported by
our installed `mlx-lm`, so an end-to-end PPL/KLD run was not possible. We
validated on **real weight matrices** from the downloaded checkpoint with a
synthetic PSD Hessian: this tests equivalence and speed of the refinement step,
not downstream model quality.

Download: `hf download XHToken/Spark-X2.6-1.7B --local-dir ~/Projects/ml/models/Spark-X2.6-1.7B-BF16`.
All 20 files matched the Hub tree sizes exactly.

## Setup

For each target linear:

- `W` taken from the real safetensors shard and cast to fp32
- `t = column RMS(W)` (our `alpha=1.0` D1 scale)
- `(W/t)B` rotated with our 32-block Hadamard path
- `H` built from 512-token random activations, then rotated with `rotate_gram`
- `mx.quantize(..., mode="mxfp4")`, then our `screen_exponents`
- 32 refinement iterations in both arms (plus one 256-iteration case)

## Results

32 iterations:

| Layer | Shape | Sweeps | Different codes | Hessian objective delta | Pure time | Metal time | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| layers.0.mlp.up_proj | 6656×2048 | 32/32 | 44 / 13,631,488 | -1.7e-4 | 0.411 s | 0.040 s | **10.2×** |
| layers.0.mlp.down_proj | 2048×6656 | 32/32 | 62 / 13,631,488 | -2.1e-4 | 0.846 s | 0.056 s | **15.2×** |
| layers.1.mlp.up_proj | 6656×2048 | 32/32 | 66 / 13,631,488 | +1.5e-5 | 0.435 s | 0.045 s | **9.7×** |

256 iterations on `layers.0.mlp.up_proj`:

- different codes: 8,403 / 13,631,488 (0.062%)
- objective delta: +1.7e-3 on 172.95 (relative ~1e-5)
- speedup: **13.0×** (3.412 s → 0.263 s)

The tiny code differences are tie-breaking/numerical, not a systematic quality
loss: the Hessian objective is unchanged to within ~1e-5 relative and is
occasionally slightly better for the fused kernel.

## Integration

- `macaq_flip_kernel.py` provides `flip_refine_metal`.
- `macaq_hadamard.py` exposes `--flip-impl {mlx,metal}`, default `mlx`.
- `tests/test_flip_kernel.py` passes with tolerances: code-diff fraction < 1e-4,
  objective relative delta < 1e-5, same sweep count.
- MoE experts (3-D weights) still use the pure-MLX path: the fused kernel is
  2-D only.

## Decision

Adopt the fused kernel as an **opt-in** speedup for future W4A16 quantization
runs. Before publishing any checkpoint produced with it, run the usual full
PPL/KLD/top-1 gate against the pure-MLX path. Do not change the existing
published artifacts.
