#!/bin/bash
# Ridge / screen-offset sweep at fixed calib+flips.
# Usage: sweep_extra.sh "<ridge>:<offsets>" ...
#   e.g. sweep_extra.sh "0.5:-1,0,1" "2.0:-1,0,1" "1.0:-2,-1,0,1,2"
cd /Users/hermit.dave/Projects/ml/models || exit 1
export HF_HUB_OFFLINE=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
POC="$(cd "$ROOT/.." && pwd)"
CACHE="${HESS_CACHE:-$POC/hess_cache}"
LOG="$ROOT/sweep_extra.log"
CALIB=200000
FLIPS=256

echo "=== extra sweep (calib=$CALIB flips=$FLIPS, started $(date)) ===" > "$LOG"
for CFG in "$@"; do
  RIDGE="${CFG%%:*}"
  OFF="${CFG##*:}"
  TAG="r${RIDGE}-o${OFF}"
  OUT="LFM3-1.7B-MLX-mxfp4-$TAG"
  echo "" >> "$LOG"
  echo "--- ridge=$RIDGE offsets=$OFF ---" >> "$LOG"
  python3 "$POC/macaq_refine.py" \
      --model LFM3-1.7B-MLX-mxfp4 --bf16 LFM3-1.7B-MLX-BF16 \
      --output "$OUT" --calib-tokens "$CALIB" --flips "$FLIPS" --group-size 24 \
      --ridge "$RIDGE" --screen-offsets "$OFF" \
      --hessian-cache "$CACHE" >> "$LOG" 2>&1
  PPL=$(python3 "$POC/eval_ppl.py" --model "$OUT" --samples 256 --seq-len 512 \
        --batch-size 8 2>&1 | grep "^perplexity")
  OBJ=$(grep "^total Hessian" "$LOG" | tail -1)
  echo "RESULT ridge=$RIDGE offsets=$OFF | $PPL | $OBJ" | tee -a "$LOG"
done
echo "" >> "$LOG"
echo "=== extra sweep done $(date) ===" >> "$LOG"
grep "^RESULT" "$LOG"
