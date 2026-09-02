#!/usr/bin/env bash
# Iki varyant icin 5 katli capraz dogrulama, sirayla.
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

for variant in baseline clahe; do
  if [ "$variant" = "clahe" ]; then dir="data/processed_clahe"; else dir="data/processed"; fi
  echo ""
  echo "=================================================="
  echo "CV BASLIYOR  variant=$variant"
  echo "=================================================="
  python -u scripts/train_cv.py --folds 5 --model efficientnet_b0 --size 384 \
    --batch 16 --epochs 15 --patience 5 --lr 3e-4 --workers 4 --seed 42 \
    --data-dir "$dir" --variant "$variant" --exclude-leaked --author senanur
  echo "CV BITTI  variant=$variant  (cikis $?)"
done
echo ""
echo "TUM CV KOSULARI TAMAMLANDI"
