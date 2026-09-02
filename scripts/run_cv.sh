#!/usr/bin/env bash
# Five-fold cross-validation for every preprocessing variant, in sequence.
#
# Run this and leave the machine alone: starting a second GPU job alongside it
# exhausts the Windows commit limit and kills the dataloader workers. Do not
# edit scripts/train_cv.py while it runs either - on Windows the dataloader
# workers re-import the main script by path.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

VARIANTS=${*:-"baseline clahe squash"}

for variant in $VARIANTS; do
  case "$variant" in
    baseline) dir="data/processed" ;;
    clahe)    dir="data/processed_clahe" ;;
    squash)   dir="data/processed_squash" ;;
    *) echo "unknown variant: $variant"; exit 1 ;;
  esac

  echo ""
  echo "=================================================="
  echo "CV START  variant=$variant  dir=$dir"
  echo "=================================================="
  python -u scripts/train_cv.py --folds 5 --model efficientnet_b0 --size 384 \
    --batch 16 --epochs 15 --patience 5 --lr 3e-4 --workers 4 --seed 42 \
    --data-dir "$dir" --variant "$variant" --exclude-leaked --author senanur
  echo "CV DONE  variant=$variant  (exit $?)"
done

echo ""
echo "ALL CV RUNS COMPLETE"
