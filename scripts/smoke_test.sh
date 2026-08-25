#!/usr/bin/env bash
# Ten steps on a 135M model: proves the loop runs end to end before you spend
# GPU-hours on a real config. It will not learn anything.
#
#   scripts/smoke_test.sh

set -euo pipefail

echo "==> Unit tests (no GPU, no network)"
python -m pytest -q

echo "==> Ten-step training loop"
grpo train --config configs/smoke.yaml

echo "==> Metrics written:"
tail -n 3 outputs/smoke/metrics.jsonl
