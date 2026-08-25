#!/usr/bin/env bash
# Full GSM8K run: baseline eval -> GRPO post-training -> final eval.
#
# The two evals are the point. A GRPO run's training reward always goes up
# (that is what it optimizes), so the only honest measure of whether anything
# was learned is held-out accuracy under identical decoding before and after.
#
#   scripts/train_gsm8k.sh [config] [output_dir]

set -euo pipefail

CONFIG="${1:-configs/qwen2.5-0.5b-gsm8k.yaml}"
OUT="${2:-outputs/gsm8k}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"

mkdir -p "$OUT"

echo "==> Baseline evaluation (before training)"
grpo eval --config "$CONFIG" \
  --split test --limit "$EVAL_LIMIT" \
  --output "$OUT/eval_before.json"

echo "==> GRPO post-training"
grpo train --config "$CONFIG" --set train.output_dir="$OUT"

echo "==> Final evaluation (same split, same decoding)"
grpo eval --config "$CONFIG" \
  --model "$OUT/final" \
  --split test --limit "$EVAL_LIMIT" \
  --output "$OUT/eval_after.json"

echo "==> Done"
echo "    before: $(cat "$OUT/eval_before.json")"
echo "    after:  $(cat "$OUT/eval_after.json")"
