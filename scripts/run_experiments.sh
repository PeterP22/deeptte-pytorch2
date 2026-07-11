#!/usr/bin/env bash
# Sequential experiment sweep; each run early-stops. Results appended as they land.
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  name=$1; shift
  echo "=== $name ==="
  uv run python -m deeptte.train --run-name "$name" --epochs 60 --patience 8 "$@" 2>/dev/null
  echo "--- $name test metrics ---" >> results/experiments.log
  uv run python -m deeptte.evaluate --checkpoint "checkpoints/$name/best.pt" \
      --results-file "results/$name-predictions.txt" 2>/dev/null | tee -a results/experiments.log
}

mkdir -p results
run t1-a03
run t1-a01 --alpha 0.1
run t1-mask --masked-attention
run t1-buckets --dist-buckets 20
run t1-geo --geohash
echo "SWEEP COMPLETE" | tee -a results/experiments.log
