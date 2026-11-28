# Spark-X2.6-1.7B: MACaQ vs RCO

Both 4-bit variants were evaluated with the same harness on M3 Max.

- **MACaQ**: `Spark-X2.6-1.7B-MLX-MACaQ-4.25bpw`, weight-only MACaQ refinement
  of MLX `mxfp4` (exponent screen + Flip-Refinement).
- **RCO**: `Spark-X2.6-1.7B-MLX-RCO-4.5bpw`, built with `rco-mlx`
  `examples/rco_pipeline.py`, dense `default` adapter, target 4.5 bpw,
  group size 64, bits {2,3,4,6,8}, WikiText calibration and LSQ refit.
  Nominal achieved 4.5005 bpw; parameter-weighted allocation 4.0004 bpw.

## Zero-shot (400 items/task, Wilson 95% CI)

| Task | BF16 | MACaQ | RCO |
| :--- | ---: | ---: | ---: |
| ARC-Challenge | 0.3000 [0.257,0.347] | 0.2675 [0.227,0.313] | 0.2250 [0.187,0.268] |
| ARC-Easy | 0.5575 [0.509,0.605] | 0.5525 [0.504,0.601] | 0.4825 [0.434,0.531] |
| HellaSwag | 0.3475 [0.303,0.395] | 0.3325 [0.288,0.380] | 0.3200 [0.276,0.367] |
| WinoGrande | 0.5175 [0.469,0.566] | 0.5025 [0.454,0.551] | 0.5050 [0.456,0.554] |
| MMLU | 0.2625 [0.222,0.308] | 0.2875 [0.245,0.334] | 0.2525 [0.212,0.297] |

Paired McNemar on the same items:

- MACaQ vs BF16: no task significant (all p ≥ 0.104).
- RCO vs BF16: ARC-Challenge p = 0.0001, ARC-Easy p = 0.0005 — significant
  losses; other tasks not significant.
- MACaQ vs RCO: ARC-Easy p = 0.0018 (survives Bonferroni across 5 tasks),
  ARC-Challenge p = 0.0363 (trend only), other tasks not significant.

## PPL and throughput

Pinned WikiText-2 test corpus from `model-eval` (199,902 tokens, context 2048):

| Model | PPL ↓ | Decode TPS | Peak memory |
| :--- | ---: | ---: | ---: |
| BF16 | 78.35 | 86.8 | 5.97 GB |
| MACaQ 4.25 | **88.16** | **242.3** | **1.85 GB** |
| RCO 4.5 nominal | 103.39 | 191.7 | 2.29 GB |

decode TPS: M3 Max, natural WikiText-2 prompt, 256 generated tokens.

## Conclusion

MACaQ is the better 4-bit variant on this model: lower PPL by ~15%, better
decode throughput by ~26%, and no significant zero-shot drop versus BF16. RCO
loses significantly on both ARC tasks and runs slower. Keep MACaQ as the
Spark-X2.6 quantization; do not publish the RCO variant.
