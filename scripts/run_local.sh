#!/usr/bin/env bash
# Run exactly one real dataset audio through ASR -> planner -> official simulator.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv-local/bin/python ]]; then
    echo 'Run bash scripts/setup_local.sh first.'
    exit 1
fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUTPUT="result/local_$(date +%Y%m%d_%H%M%S)_$$.jsonl"
.venv-local/bin/python -m baseline.run --config configs/local.json --split train \
    --sample-id "${1:-train_0004}" --output "$OUTPUT" --verbose
.venv-local/bin/python -m baseline.inspect_run "$OUTPUT" --require-complete
echo "Saved: $OUTPUT"
