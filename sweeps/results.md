# Tuning results

LFM3-1.7B, WikiText-2 train calibration, WikiText-2 test evaluation
(130,816 tokens, sequence length 512, batch size 8). All runs use the default
`ridge=1.0` and screen offsets `{-1,0,+1}` unless the column says otherwise.

## Baselines

| Model | Perplexity ↓ | vs BF16 |
| --- | --- | --- |
| BF16 | 30.839 ± 0.289 | — |
| affine-4 (group 64) | 42.380 ± 0.414 | +11.54 |
| mxfp4 RTN (MLX default) | 45.501 ± 0.446 | +14.66 |

## Sweep: flip budget (calib = 200K)

| flips | Perplexity ↓ | Hessian obj. ↓ | mean sweeps used |
| --- | --- | --- | --- |
| 64 | 36.140 ± 0.336 | 30.9% | 64.0 / 64 |
| **256** | **35.472 ± 0.336** | **32.5%** | 252.9 / 256 |
| 512 | 35.376 ± 0.335 | 32.5% | 388.2 / 512 |

The flip budget saturates at ~256. Beyond it the Hessian objective is flat
(1.0592e8 → 1.0589e8) and perplexity moves by 0.10.

## Sweep: calibration tokens (flips = 256)

| calib tokens | Perplexity ↓ | Hessian obj. ↓ |
| --- | --- | --- |
| 200 K | 35.472 ± 0.336 | 32.5% |
| 500 K | 35.599 ± 0.338 | 31.9% |
| 1 M | 35.410 ± 0.336 | 31.8% |

No significant effect; all three are inside the ±0.34 noise band.

## Sweep: ridge and screen range (calib = 200K, flips = 256)

| ridge | screen offsets | Perplexity ↓ | Hessian obj. ↓ |
| --- | --- | --- | --- |
| 0.5 | {-1,0,+1} | 35.484 ± 0.336 | 41.9% |
| **1.0** | **{-1,0,+1}** | **35.472 ± 0.336** | **32.5%** |
| 2.0 | {-1,0,+1} | 36.268 ± 0.345 | 24.0% |
| 1.0 | {-2..+2} | 35.486 ± 0.336 | 32.5% |

`ridge=0.5` inflates the Hessian reduction (41.9%) without improving perplexity —
the layer objective is a proxy for model quality, not a substitute for it.
`ridge=2.0` underfits and degrades. Widening the screen range changes nothing.

## Recommended configuration

```
--calib-tokens 200000 --flips 256 --ridge 1.0 --screen-offsets -1,0,1
```

**Result: 35.472 ± 0.336**, a 22.1% perplexity reduction versus the mxfp4 RTN
baseline, closing 67% of the BF16 → mxfp4 gap.

## Reproducing

```bash
bash sweeps/sweep_flips.sh 64 256 512
bash sweeps/sweep_calib.sh 256 500000 1000000
bash sweeps/sweep_extra.sh "0.5:-1,0,1" "2.0:-1,0,1" "1.0:-2,-1,0,1,2"
```

Raw logs are in `sweep_*.log`.
