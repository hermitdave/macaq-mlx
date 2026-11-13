#!/bin/bash
# Flip-count sweep at fixed calibration (200K tokens).
cd /Users/hermit.dave/Projects/ml/models || exit 1
export HF_HUB_OFFLINE=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
POC="$(cd "$ROOT/.." && pwd)"
CACHE="${HESS_CACHE:-$POC/hess_cache}"
LOG="$ROOT/sweep_flips.log"

echo "=== flip sweep (calib=200K, started $(date)) ===" > "$LOG"
for F in "$@"; do
  OUT="LFM3-1.7B-MLX-mxfp4-f$F"
  echo "" >> "$LOG"
  echo "--- flips=$F ---" >> "$LOG"
  python3 "$POC/macaq_refine.py" \
      --model LFM3-1.7B-MLX-mxfp4 --bf16 LFM3-1.7B-MLX-BF16 \
      --output "$OUT" --calib-tokens 200000 --flips "$F" --group-size 24 \
      --hessian-cache "$CACHE" >> "$LOG" 2>&1
  PPL=$(python3 "$POC/eval_ppl.py" --model "$OUT" --samples 256 --seq-len 512 \
        --batch-size 8 2>&1 | grep "^perplexity")
  OBJ=$(grep "^total Hessian" "$LOG" | tail -1)
  SW=$(grep "^flip sweeps" "$LOG" | tail -1)
  echo "RESULT flips=$F | $PPL | $OBJ | $SW" | tee -a "$LOG"
done
echo "" >> "$LOG"
echo "=== sweep done $(date) ===" >> "$LOG"
grep "^RESULT" "$LOG"
