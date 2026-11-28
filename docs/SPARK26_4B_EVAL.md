# Spark-X2.6-4B: MACaQ variants vs RCO

Same pipeline as `SPARK26_EVAL.md`, repeated on `XHToken/Spark-X2.6-4B`.
Unlike the 1.7B model, the Hadamard-rotated MACaQ path **does not regress** on
the 4B checkpoint.

## Variants

- **MACaQ Hadamard 4.25**: `macaq_hadamard.py`, alpha=1.0, 256 flips, fused
  flip kernel, per-channel `t` + 32-block Hadamard. Requires
  `trust_remote_code=True`.
- **MACaQ weight-only 4.25**: `macaq_refine.py` on an MLX `mxfp4` RTN base,
  exponent screen + flips, no rotation.
- **RCO 4.5 nominal**: `rco-mlx` dense/default pipeline, group-size 64,
  bits {2,3,4,6,8}, WikiText calibration + LSQ refit. Achieved 4.5000 nominal /
  3.9999 parameter-weighted.
- **RTN 4.25**: `mlx_lm.convert --quantize --q-mode mxfp4`.

## Zero-shot (`model-eval`, 400 items/task, Wilson 95% CI)

| Task | BF16 | MACaQ Hadamard | Δ vs BF16 | p (McNemar) | MACaQ WO | RCO |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | 0.3475 [0.303,0.395] | **0.3550** [0.310,0.403] | +0.0075 | 0.7359 | 0.3025 [0.260,0.349] | 0.3000 [0.257,0.347] |
| ARC-Easy | 0.5775 [0.529,0.625] | **0.5775** [0.529,0.625] | 0.0000 | 1.0000 | 0.5175 [0.469,0.566] | 0.5225 [0.474,0.571] |
| HellaSwag | 0.4000 [0.353,0.449] | 0.3800 [0.334,0.429] | −0.0200 | 0.0386 | **0.4000** [0.353,0.449] | 0.3800 [0.334,0.429] |
| WinoGrande | 0.5350 [0.486,0.583] | **0.5575** [0.509,0.605] | +0.0225 | 0.3425 | 0.5200 [0.471,0.569] | 0.5175 [0.469,0.566] |
| MMLU | 0.3225 [0.279,0.370] | 0.3050 [0.262,0.352] | −0.0175 | 0.3240 | 0.2825 [0.241,0.329] | 0.2700 [0.229,0.316] |

**Statistically lossless verdict:** yes under Bonferroni-corrected exact McNemar
across five tasks (α = 0.01). No task shows a significant degradation versus
BF16. HellaSwag is nominally p = 0.0386 but does not survive correction; under
an uncorrected α = 0.05 rule it would fail. Δ is Hadamard minus BF16 on the
same items.

Other paired McNemar comparisons:
- Hadamard vs weight-only: ARC-Challenge p = 0.0046 and ARC-Easy p = 0.0079
  favour Hadamard (both survive correction at 0.01).
- Hadamard vs RCO: ARC-Challenge p = 0.0038 and ARC-Easy p = 0.0103 favour
  Hadamard; ARC-Easy is borderline after Bonferroni.

## PPL and throughput

Pinned WikiText-2 test corpus (199,902 tokens, context 2048):

| Model | PPL ↓ | Decode TPS | Peak memory |
| :--- | ---: | ---: | ---: |
| BF16 | **23.14** | 41.3 | 8.98 GB |
| **MACaQ Hadamard 4.25** | 25.73 | 96.0 | 3.71 GB |
| MACaQ weight-only 4.25 | 25.78 | **122.2** | **3.24 GB** |
| RCO 4.5 nominal | 27.71 | 104.6 | 3.85 GB |
| RTN 4.25 | 129.63 | — | — |

Short-context PPL (64 × 512) gave the same ordering: BF16 49.69, Hadamard
57.61, weight-only 60.11, RCO 62.38, RTN 129.63.

## Decision

Promote **MACaQ Hadamard 4.25** for Spark-X2.6-4B: best zero-shot and PPL,
clearly ahead of RCO. Keep the weight-only variant as the faster fallback if
`trust_remote_code` or the online Hadamard transform is undesirable. RTN is
unusable. The 1.7B result did not generalise — the same architecture at 4B
benefits from the rotated basis.
