#!/bin/bash
# Calibration-token sweep at a fixed flip budget.
# Usage: sweep_calib.sh <flips> <tokens1> [tokens2 ...]
cd /Users/hermit.dave/Projects/ml/models || exit 1
export HF_HUB_OFFLINE=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
POC="$(cd "$ROOT/.." && pwd)"
CACHE="${HESS_CACHE:-$POC/hess_cache}"
LOG="$ROOT/sweep_calib.log"
FLIPS="$1"; shift

echo "=== calib sweep (flips=$FLIPS, started $(date)) ===" > "$LOG"
for T in "$@"; do
  OUT="LFM3-1.7B-MLX-mxfp4-t$T-f$FLIPS"
  echo "" >> "$LOG"
  echo "--- calib=$T flips=$FLIPS ---" >> "$LOG"
  python3 "$POC/macaq_refine.py" \
      --model LFM3-1.7B-MLX-mxfp4 --bf16 LFM3-1.7B-MLX-BF16 \
      --output "$OUT" --calib-tokens "$T" --flips "$FLIPS" --group-size 24 \
      --hessian-cache "$CACHE" >> "$LOG" 2>&1
  PPL=$(python3 "$POC/eval_ppl.py" --model "$OUT" --samples 256 --seq-len 512 \
        --batch-size 8 2>&1 | grep "^perplexity")
  OBJ=$(grep "^total Hessian" "$LOG" | tail -1)
  echo "RESULT calib=$T flips=$FLIPS | $PPL | $OBJ" | tee -a "$LOG"
done
echo "" >> "$LOG"
echo "=== calib sweep done $(date) ===" >> "$LOG"
grep "^RESULT" "$LOG"
