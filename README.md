# MACaQ-R for MLX

A weight-only port of [MACaQ](https://github.com/huawei-csl/MACaQ) (arXiv 2611.10649)
to Apple's MLX. It refines the E2M1 codes of an existing MLX `mxfp4` model against a
BF16 teacher, with **no runtime change, no format change, and no online transforms**.

The output is a standard MLX quantized model: same file layout, same
`{"mode": "mxfp4", "group_size": 32, "bits": 4}` config, loads with stock
`mlx_lm.generate`.

## Result

LFM3-1.7B, WikiText-2 test, 130,816 tokens, seq len 512:

| Model | Perplexity ↓ | vs BF16 |
| --- | --- | --- |
| BF16 | 30.84 ± 0.29 | — |
| affine-4 (group 64) | 42.38 ± 0.41 | +11.54 |
| mxfp4 RTN (MLX default) | 45.50 ± 0.45 | +14.66 |
| **mxfp4 + this refiner** | **35.47 ± 0.34** | **+4.63** |

Closes ~67% of the BF16→mxfp4 gap, and beats affine-4 at a lower bit width
(4.25 vs 4.50 bpw).

## Performance

M3 Max (64 GB unified memory), natural WikiText-2 prose prompts, 256 generated
tokens.

### Local (`mlx-lm`)

| Context | Model | Prompt TPS | Decode TPS | Peak memory |
| :--- | :--- | ---: | ---: | ---: |
| ~1K | BF16 | 2075 | 71.5 | 4.34 GB |
| ~1K | mxfp4 + refinement | 2623 | **150.9** | **2.22 GB** |
| ~8K | BF16 | 2903 | 70.3 | 4.95 GB |
| ~8K | mxfp4 + refinement | 2920 | **141.3** | **2.53 GB** |

### Served (oMLX)

Decode throughput read from the oMLX server log, 3 runs per cell:

| Context | BF16 | mxfp4 + refinement | Speedup |
| :--- | ---: | ---: | ---: |
| ~1K | 75.5 tok/s | 171.2 tok/s | **2.3×** |
| ~8K | 78.0 tok/s | 203.8 tok/s | **2.6×** |

Decode is 2.1–2.6× faster than BF16 and peak memory is roughly halved, because
weights are read at 4 bits instead of 16. Prefill is bandwidth-saturated at
these lengths, so prompt throughput is comparable.

The refined checkpoint was also loaded and validated as a served model in oMLX
(symlinked into `~/llm-models/`, settings `trust_remote_code: false`,
`mtp_enabled: false`). It is discoverable as
`LFM3-1.7B-MLX-MACaQ-R-4.25bpw` and generates coherent output.

## Tuning study

All runs: LFM3-1.7B, 200K calibration tokens unless noted, WikiText-2 train.

| Config | Perplexity ↓ | Hessian obj. reduction |
| --- | --- | --- |
| flips=64 | 36.140 | 30.9% |
| **flips=256** | **35.472** | 32.5% |
| flips=512 | 35.376 | 32.5% |
| calib=500K, flips=256 | 35.599 | 31.9% |
| calib=1M, flips=256 | 35.410 | 31.8% |
| ridge=0.5, flips=256 | 35.484 | 41.9% |
| ridge=2.0, flips=256 | 36.268 | 24.0% |
| screen offsets ±2, flips=256 | 35.486 | 32.5% |

Conclusions:

- **Flip budget saturates at ~256.** 64→256 gains 0.67 ppl; 256→512 gains 0.10 ppl
  and the Hessian objective is unchanged (1.0592e8 → 1.0589e8).
- **200K calibration tokens is enough.** 500K and 1M land within the ±0.34 noise band.
- **ridge λ=1.0 is at the flat optimum.** λ=0.5 gives an identical perplexity but a
  much larger Hessian reduction (41.9%) — a clean illustration that the layer
  objective is only a proxy for model quality. λ=2.0 is worse.
- **Screen range ±1 is enough.** Widening to ±2 changes nothing; MLX's own exponents
  are already near-optimal.

Recommended defaults: `--calib-tokens 200000 --flips 256 --ridge 1.0 --screen-offsets -1,0,1`.

## Usage

```bash
python macaq_refine.py \
    --model     LFM3-1.7B-MLX-mxfp4 \
    --bf16      LFM3-1.7B-MLX-BF16 \
    --output    LFM3-1.7B-MLX-mxfp4-f256 \
    --calib-tokens 200000 --flips 256 --group-size 24 \
    --hessian-cache hess_cache
```

`--hessian-cache` stores per-layer Hessians so hyper-parameter sweeps reuse them
(the 200K cache is ~8 GB; 1M is ~22 GB).

## How it works

For each quantized linear layer:

1. **Unpack** MLX's raw OCP E2M1 nibbles and E8M0 block scales into a signed-grid
   index `sidx ∈ [0,14]` and per-block `log2` scales.
2. **Capture the Hessian** `H = Σ xᵀx` from the BF16 teacher on calibration text,
   using the same forward-hook pattern as `mlx_lm/quant/gptq.py`.
3. **Screen block exponents** over offsets {-1,0,+1} under the diagonal Hessian
   metric `Σ_j H_jj (W_ij − Ŵ_ij)²`.
4. **Flip-Refinement**: greedy coordinate descent on the E2M1 codes with the
   closed-form gain `2Δg − Δ²H_jj`, exponents held fixed. Early-stops when no row
   has an improving move.
5. **Repack** into MLX's native layout and write back.

## Key MLX findings

These were all discovered empirically and matter for anyone doing MXFP4 work in MLX:

1. **MLX 0.32 has native MXFP4** — `mx.quantize(mode="mxfp4")` produces E2M1
   elements with an E8M0 scale per 32-element group, packed into uint32. This is
   exactly MACaQ's weight format.
2. **`mx.qqmm(mode="mxfp4")` performs real W4A4** (quantizes activations on the
   fly), despite the docstring listing only `nvfp4` and `mxfp8`. Verified against
   an emulated W4A4 reference (rel. err 3.3e-6).
3. **Custom codes are injectable.** The signed-grid index maps to the raw E2M1
   nibble as `nibble = sidx ≥ 7 ? sidx − 7 : 15 − sidx`. Packed output round-trips
   through `mx.dequantize` and both matmul kernels bit-exactly.
4. **MLX's exponent rule differs from MACaQ's RTN.** MLX uses
   `e = ceil(log2(amax/6))` — the finest power-of-two scale that does not clip the
   block maximum. MACaQ uses `floor(log2(amax)) − 2` and then screens offsets.
   So `mx.quantize` is not a drop-in RTN baseline.
5. **Packing gotcha (mine, not MLX's).** The packer multiplies nibbles by
   powers of two (1, 16, 256, …); unpacking must shift by the *exponents*
   (0, 4, 8, …). Using the powers as shift counts fails silently — a shift
   count ≥ 32 wraps modulo 32 on x86/ARM, so `x >> 256` evaluates as `x >> 0`
   and returns the unshifted value rather than zero. An earlier version of this
   refiner had exactly that bug and reported a bogus 99.6% Hessian reduction.

## Limitations

- **No Hadamard rotation.** Applying it requires an online activation transform,
  which is the part of MACaQ that does not port cleanly to MLX (see the analysis
  in the session notes). This costs the largest single ablation term in the paper.
- **W4A4 is not viable above decode on MLX.** `mx.qqmm` mxfp4 is only faster than
  bf16 at batch ≤ 4, and 2.5–3.8× slower at batch ≥ 16.
- **The layer objective is a proxy.** Lowering ridge inflates the Hessian gain
  without improving perplexity.
- **Single model validated.** LFM3-1.7B only; no MoE, no model > 2B.

## Files

| Path | Purpose |
| --- | --- |
| `macaq_refine.py` | the refiner |
| `eval_ppl.py` | WikiText-2 perplexity evaluator |
| `verify_pack.py` | pack/unpack verification against `mx.dequantize` |
| `poc_single_layer.py` | single-layer MACaQ proof-of-concept |
| `benchmarks/` | MLX MXFP4 kernel, local TPS, and oMLX served TPS benchmarks |
| `sweeps/` | tuning sweep drivers, raw logs, and `results.md` |

## Attribution

This port was developed with [Hermes Agent](https://hermes-agent.nousresearch.com/).

It builds directly on
[MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations](https://arxiv.org/abs/2611.10649)
by Lorenz K. Muller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang,
and Lukas Cavigelli, whose code is
[released under Apache-2.0](https://github.com/huawei-csl/MACaQ). The algorithm
(Flip-Refinement, Hessian-weighted exponent screening) and the MACaQ-R
weight-only refinement strategy follow their design; the MLX implementation,
packing layer, benchmarks, and tuning study here are new.

## License

Apache-2.0. See [LICENSE](LICENSE).
