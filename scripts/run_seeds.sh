#!/usr/bin/env bash
# Repeat a single-split comparison across several seeds.
#
# Measures how much of an apparent difference between two variants is real and
# how much is run-to-run noise. Seed 42 established the first result; this adds
# more seeds so the sign of the difference can be checked for consistency.
#
# Usage: bash scripts/run_seeds.sh 43 44
set -u

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(43 44)

cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

for seed in "${SEEDS[@]}"; do
  for variant in baseline clahe; do
    if [ "$variant" = "clahe" ]; then
      data_dir="data/processed_clahe"
    else
      data_dir="data/processed"
    fi

    echo ""
    echo "=================================================="
    echo "START  variant=$variant  seed=$seed"
    echo "=================================================="

    python -u scripts/train.py \
      --mode reg --model efficientnet_b0 --size 384 --batch 16 \
      --epochs 15 --patience 5 --lr 3e-4 --workers 4 \
      --seed "$seed" --data-dir "$data_dir" --variant "$variant" \
      --author senanur

    echo "DONE  variant=$variant  seed=$seed  (exit $?)"
  done
done

echo ""
echo "ALL SEED RUNS COMPLETE"
