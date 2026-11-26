# 32-block Hadamard MACaQ (W4A16) on MLX

This is an experimental extension of `macaq-mlx`. Unlike the default refiner,
it rotates both the weight and the activation before quantization:

```
y = ((x * t) B) @ ((W / t) B)^T
```

`t` is a per-input-channel D1 scale and `B` is the orthonormal Walsh-Hadamard
transform applied to independent 32-element blocks (the MXFP4 group size).
Only `W' = (W / t) B` is stored as MXFP4; `t` is stored in bf16 and `B` is
applied online by `macaq_model.py`.

## Why

On Apple Silicon there is no FP4 tensor core, so full W4A4 adds error without
a speedup. Rotating the weight spreads outlier channels across MXFP4 groups;
the third-party MACaQ W4A16 write-up reports removing the rotation costs
11-15% KLD on WikiText-2 and 13-20% on C4, more than removing the D1 scale.

## Build

```bash
python macaq_hadamard.py \
  --bf16 /path/to/Qwen3.5-9B-BF16-text \
  --output /path/to/Qwen3.5-9B-MLX-MACaQ-Hadamard-4.25bpw \
  --calib-tokens 65536 --seq-len 512 --batch-size 8 \
  --flips 256 --alpha 1.0 --group-size 2 \
  --hessian-cache /path/to/hess_cache
```

The script starts from BF16 weights (it does **not** refine existing MXFP4
codes), captures per-linear Hessians with module wrappers, rotates the
Hessian analytically (`H' = B D H D B`), then runs the existing exponent
screen and Flip-Refinement in the rotated basis.

## Loading

The checkpoint is **not** a plain MXFP4 model. It carries
`"model_file": "macaq_model.py"` and a `macaq` section in `config.json`, and
requires `trust_remote_code=True`:

```bash
mlx_lm.generate --model <out> --trust-remote-code --prompt "..."
python eval_ppl.py --model <out>
```

Loading without the shim either refuses or silently produces garbage; the
config marker makes plain loaders fail loudly.

### oMLX

Symlink the output into a `model_dirs` directory, then add a per-model entry:

```json
"Qwen3.5-9B-MLX-MACaQ-Hadamard-4.25bpw": {
  "trust_remote_code": true,
  "model_type_override": "llm",
  "mtp_enabled": false,
  "ttl_seconds": 3600,
  "enable_thinking": false
}
```

`model_type_override: "llm"` matters: Qwen3.5 checkpoints otherwise route to
the VLM engine, where `model_file` is not applied. Reload the registry with
`POST /admin/api/reload`, then load the model via
`POST /admin/api/models/<id>/load`.

## Validation (Qwen3.5-9B-text, 2026-11-26)

Built with `--calib-tokens 65536 --seq-len 512 --batch-size 8 --flips 256
--alpha 1.0 --ridge 1.0 --group-size 2`: 248 linears across 32 layers,
rotated Hessian objective 1.268e9 -> 7.935e8 (37.4% reduction), 4486 s wall
time. Output 7.2 GB.

| Check | Result |
|---|---|
| WikiText-2 PPL, 32×512 tokens | 8.7089 ± 0.1829 |
| BF16 teacher, same eval | 8.5607 ± 0.1799 (+1.73% for the quant) |
| `mlx_lm.generate --trust-remote-code` | coherent, 52.6 tok/s, 7.88 GB peak |
| oMLX load (model_type_override=llm) | `BatchedEngine loaded`, 7.44 GB actual |
| oMLX chat completions | coherent answers, 41–49 tok/s |

No same-recipe unrotated MACaQ checkpoint was rebuilt for this session, so the
quality comparison against the old quant is indirect: the per-layer objective
and the third-party ablation above. A historical session
(@session:helix/local-e09884e071ccfc2b94dfb73706b58a2845a412a4c6b6f3ef1d097697d80cd2b8)
recorded the unrotated 4.25 bpw MACaQ-Hessian build on this model: PPL 10.5345
vs BF16 10.2656 (+2.62%), KL 0.0388, top-1 88.82%, with 200k calibration
tokens. That used a different harness (`eval_quality_table`, 8×256 positions)
from the 32×512 `eval_ppl.py` run above, so the absolute PPLs are not
comparable; the relative gap to BF16 (+1.73% rotated vs +2.62% unrotated) is
only indicative. The old checkpoint has since been deleted locally, so a
same-harness rerun would require rebuilding it.

### Head-to-head with RCO 4.63 (same harness)

Re-run with `eval_quality_table.py` protocol (8×256 WikiText rows, BF16 text
teacher), after downloading `hermitdave/Qwen3.5-9B-MLX-RCO-4.63bpw-mtp4`:

| Model | PPL | dPPL | KL | top-1 |
|---|---:|---:|---:|---:|
| BF16 | 10.2617 | — | — | — |
| **MACaQ Hadamard 4.25** | **10.4865** | **+0.2247** | **0.0284** | **91.37%** |
| RCO 4.63 (mtp4) | 10.6043 | +0.3426 | 0.0397 | 89.46% |

Historical unrotated MACaQ-Hessian on the same harness: PPL 10.5345,
KL 0.0388, top-1 88.82%. So rotation moves every metric in the right direction.

Caveat: this is not equal parameter coverage. The rotated checkpoint quantizes
248 decoder linears to MXFP4 but keeps embeddings, lm_head, norms and vision
out of scope; the RCO checkpoint quantizes those too (its config has affine
4-bit `lm_head` and `embed_tokens`). The rotated artifact is also smaller
(4.25 vs 4.63 bpw overall). Treat this as a deployment-relevant comparison,
not a controlled study of the two quantizers on identical tensors.

### Throughput (local mlx-lm, M3 Max, 988-token prompt, 256 tokens)

| Model | prompt tok/s | decode tok/s | peak GB |
|---|---:|---:|---:|
| MACaQ Hadamard 4.25 | 665.2 | 50.0 | 9.04 |
| RCO 4.63 (mtp4) | 738.9 | **64.2** | 6.38 |

The quality advantage does not buy speed: RCO decodes ~28% faster and uses less
memory. This matches the earlier Qwen3.5-9B finding that MACaQ is a fidelity
play while RCO is the throughput choice. The Hadamard layer adds an online
scale + Hadamard transform per linear, and its embeddings/lm_head stay BF16.

### Grafted 4-bit MTP head — negative result

Grafted the 31 `mtp.*` tensors (0.137 GB) and 8 affine-4 MTP quantization
entries from RCO 4.63 into a copy of the Hadamard checkpoint
(`Qwen3.5-9B-MLX-MACaQ-Hadamard-4.25bpw-mtp4`). oMLX loads it and selects
`Lightning MTP (model_type=qwen3_5, active)`, but drafting is rejected:

| config | tok/s | vs off | acceptance |
|---|---:|---:|---|
| MTP off | 47.9 | 1.00× | — |
| depth 1 | 38.3 | 0.80× | 17/237 = **7.2%** (d2 0/17) |
| depth 3 | 25.0 | 0.52× | ~6.7% (d2 0/17, d3 0/0) |

This reproduces the earlier MACaQ/MTP finding (3.3–3.4% acceptance, 0.64×):
the rotated backbone's hidden states do not match a head trained against a
different quantization basis. `mtp_enabled` stays false; the grafted checkpoint
is kept locally as the documented negative result and is unloaded.

## Trade-offs

- Quality: rotated codes lower the Hessian-weighted layer error; expect a
  KLD improvement over the unrotated MACaQ quant, not a size change.
- Memory: weights stay at 4.25 bpw; each rotated linear adds a bf16 `t`
  vector (2 bytes per input channel).
- Speed: the online scale + Hadamard is extra work per linear. No FP4 matmul
  benefit on Apple GPUs, so decode is slower than the unrotated weight-only
  quant.
- Portability: oQh (oMLX's own rotated format) cannot load these checkpoints;
  different codebook and block size.
