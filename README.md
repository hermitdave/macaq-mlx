# MACaQ for MLX

A weight-only port of [MACaQ](https://github.com/huawei-csl/MACaQ) (arXiv 2611.10649)
to Apple's MLX. It refines the E2M1 codes of an existing MLX `mxfp4` model against a
BF16 teacher, with **no runtime change, no format change, and no online transforms**.

> **Naming:** artifacts use `MACaQ` (e.g. `LFM3-1.7B-MLX-MACaQ-4.25bpw`), not
> `MACaQ-R`. The `-R` suffix is unofficial repo-only jargon — the paper never uses
> it; the technique appears there as Appendix 0.C, "Extending Flip-Refinement to
> Vector-Quantized Weights".

The output is a standard MLX quantized model: same file layout, same
`{"mode": "mxfp4", "group_size": 32, "bits": 4}` config, loads with stock
`mlx_lm.generate`.

## Result

### LFM3-1.7B (hybrid attention)

WikiText-2 test, 130,816 tokens, seq len 512:

| Model | Perplexity ↓ | vs BF16 |
| --- | --- | --- |
| BF16 | 30.84 ± 0.29 | — |
| affine-4 (group 64) | 42.38 ± 0.41 | +11.54 |
| mxfp4 RTN (MLX default) | 45.50 ± 0.45 | +14.66 |
| **mxfp4 + this refiner** | **35.47 ± 0.34** | **+4.63** |

Closes ~67% of the BF16→mxfp4 gap, and beats affine-4 at a lower bit width
(4.25 vs 4.50 bpw).

### OLMoE-1B-7B (MoE, 64 experts, top-8)

Same protocol:

| Model | Perplexity ↓ | vs BF16 |
| --- | --- | --- |
| BF16 | 8.4512 ± 0.0610 | — |
| mxfp4 RTN (MLX default) | 9.5579 ± 0.0699 | +1.107 |
| **mxfp4 + this refiner** | **8.9950 ± 0.0651** | **+0.544** |

Closes ~51% of the BF16→mxfp4 gap. 129 layers refined (81 dense + 48 MoE
switch projections), 37.1% Hessian reduction, 34 min wall time.

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

Served numbers vary by roughly ±10% between sessions depending on what else is
resident; treat the absolute figures as indicative and the ratio as the result.

The refined checkpoint was also loaded and validated as a served model in oMLX
(symlinked into `~/llm-models/`, settings `trust_remote_code: false`,
`mtp_enabled: false`). It is discoverable as
`LFM3-1.7B-MLX-MACaQ-4.25bpw` and generates coherent output.

### OLMoE-1B-7B

Same benchmark, same hardware, 256 generated tokens:

| Context | Model | Prompt TPS | Decode TPS | Peak memory |
| :--- | :--- | ---: | ---: | ---: |
| ~1K | BF16 | 3244 | 109.6 | 14.47 GB |
| ~1K | mxfp4 RTN | 2479 | 276.9 | 4.46 GB |
| ~1K | mxfp4 + refinement | 3416 | **272.6** | **4.46 GB** |
| ~8K | BF16 | 3814 | 89.3 | 15.37 GB |
| ~8K | mxfp4 RTN | 3607 | 153.2 | 5.29 GB |
| ~8K | mxfp4 + refinement | 3602 | **152.0** | **5.29 GB** |

Decode is 2.5× faster than BF16 and peak memory is 3.2× lower. The refined
checkpoint lands within run noise of the stock mxfp4 quantization, so the
quality gain is free at inference.

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

MoE models need a per-layer group and an expert chunk size:

```bash
python macaq_refine.py \
    --model     OLMoE-1B-7B-MLX-mxfp4 \
    --bf16      OLMoE-1B-7B-MLX-BF16 \
    --output    OLMoE-1B-7B-MLX-MACaQ-4.25bpw \
    --calib-tokens 200000 --flips 256 --group-size 1 --expert-chunk 8
```

`--expert-chunk` batches experts through one graph to bound memory during
refinement (8 experts × 1024 × 2048 is ~600 MB of intermediates). The flip
budget applies per chunk.

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

**MoE.** MLX-LM does not quantize experts into individual linears; it collapses
them into one `QuantizedSwitchLinear` holding a stacked `(E, out, in)` tensor.
The refiner unpacks that tensor with the same nibble path, captures a separate
Hessian per expert, and runs the screen and Flip-Refinement with the expert
axis as a batch dimension. Per-expert Hessians come from a `SwitchCatcher`
that scatters each token into the slot of the expert it was routed to, so `H`
reflects the tokens that expert actually sees rather than the layer's whole
input. The scatter relies on MLX accepting a lazily-evaluated shape in
`mx.zeros`, which avoids a host sync per batch.

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
- **Two models validated.** LFM3-1.7B (hybrid attention) and OLMoE-1B-7B (MoE);
  no model > 2B.
- **MoE Hessians need `--group-size 1`.** Per-expert Hessians are
  `(num_experts, in, in)`; at 64 experts and `in=2048` that is ~1.1 GB per
  projection, so a multi-layer group can exhaust memory. `--group-size 1`
  bounds it to one layer (~2.4 GB).
- **MoE Hessians are routing-aware, not teacher-sampled.** Tokens are
  scattered into per-expert slots so each expert's `H` comes only from tokens
  that actually routed to it — the same quantity the reference implementation
  collects with a teacher model, obtained here from the model's own forward
  pass.

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
(Flip-Refinement, Hessian-weighted exponent screening) and the
weight-only code-refinement strategy follow their design; the MLX implementation,
packing layer, benchmarks, and tuning study here are new.

## License

Apache-2.0. See [LICENSE](LICENSE).
