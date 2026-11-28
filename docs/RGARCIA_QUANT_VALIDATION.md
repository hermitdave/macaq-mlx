# Validation of `rgarcia-fx/macaq-mlx` quantization improvements

Source checked: `https://github.com/rgarcia-fx/macaq-mlx` (cloned at `main`,
Apache-2.0). It is a separate native MLX implementation of MACaQ, not our
`hermitdave/macaq-mlx` fork.

## What it actually adds

- Core W4A16 algorithm: per-channel scale `t`, 32-block Hadamard, transformed
  Hessian, exponent screen, and Flip-Refinement.
- Inference shim (`macaq_model.py`): stock `mx.hadamard_transform` +
  `mx.quantized_matmul`. No custom inference/GEMV kernel.
- One custom kernel: `mx.fast.metal_kernel(name="macaq_flip_step")` in
  `macaq_mlx.py`. It fuses one greedy flip step (best code move + gradient
  update + accept flag) into one launch. README claims 5–10× over the pure
  reference during calibration.

## Runs performed

Environment: uv-synced Python 3.12.10, mlx 0.32.4, mlx-lm 0.32.2, offline
WikiText-2 cache. Hardware: Apple M3 Max 64 GB.

1. `python macaq_mlx.py --selftest` — passed:
   - flip Metal kernel vs reference: codes equal, 21 vs 21 iterations;
   - pack round-trip error 0;
   - `mlx` activation emulation exactly matches `mx.quantize`.
2. `python tests/test_roundtrip.py` — passed (`ok`).
3. Real layer A/B on `Qwen3.5-9B-BF16-text`, layer 3, W4A16, 4096 calibration
   tokens, 4 eval sequences of 256 positions, top-1 reference:

   | Method | KLD mean | p99 KLD | top-1 agree |
   |---|---:|---:|---:|
   | RTN MXFP4 | 1.1519e-3 | 1.1348e-2 | 0.9717 |
   | Their MACaQ default (`alpha=0.75`, 64 flips) | 7.4544e-4 | 7.5938e-3 | 0.9805 |
   | Their MACaQ with our config (`alpha=1.0`, 256 flips) | **5.5534e-4** | 7.2167e-3 | 0.9785 |

   Per-layer weight relative error also improved consistently: e.g.
   `mlp.up_proj` 1.14e-2 → 8.83e-3 (64 flips) and 1.148e-2 → 7.93e-3
   (256 flips). `self_attn.q_proj` went 1.79e-3 → 1.11e-3 with the default
   config, and 5.04e-3 → 2.71e-3 in the alpha=1.0 run (different scale `t`,
   so the RTN baselines are not directly comparable across those two runs).

4. Flip speed: their fused flip step took **4.5–8.7 ms/iteration** on real
   Qwen3.5-9B linears (12288×4096 and 4096×12288). Our equivalent pure-MLX
   `macaq_refine.flip_refine` on a comparable 12288×4096 shape took
   **78–127 ms/iteration** — roughly 10–20× slower.

## Assessment

- The reported quantization improvement is **real**: MACaQ beats RTN on this
  independent layer test, and our own config (`alpha=1.0`, 256 flips) gives the
  lowest KLD.
- Their algorithm is substantially the same as our `macaq_hadamard.py`
  pipeline. We already use `alpha=1.0` and 256 flips, so there is no reason to
  adopt their quantizer wholesale or replace our checkpoints.
- The one genuinely useful piece is their **fused flip kernel**: it could cut
  calibration/refinement time by roughly an order of magnitude. It does not
  affect inference speed or quality by itself.
- No custom inference kernel exists there, so it does not change the decision
  to stay with the rotated W4A16 path.

## Recommendation

Do not adopt the repo wholesale. If we need to re-quantize more models, port the
`macaq_flip_step` kernel into `macaq_refine.py`, adapt it to our scale layout,
require code/objective equality against the current reference, and re-run the
existing PPL/KLD/top-1 gate before publishing a new checkpoint. Keep the
Hadamard checkpoint decision unchanged.
