#!/usr/bin/env bash
# Coklu seed karsilastirmasi: her varyanti ayni seed'lerle calistirir.
#
# Tek kosunun oynakligini olcmek icin. Seed 42 zaten calistirildi
# (run_id e41cb5fe = baseline, 26404d07 = clahe), burada 43 ve 44 eklenir.
#
# Kullanim: bash scripts/run_seeds.sh 43 44
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
    echo "BASLIYOR  variant=$variant  seed=$seed"
    echo "=================================================="

    python -u scripts/train.py \
      --mode reg --model efficientnet_b0 --size 384 --batch 16 \
      --epochs 15 --patience 5 --lr 3e-4 --workers 4 \
      --seed "$seed" --data-dir "$data_dir" --variant "$variant" \
      --author senanur

    echo "BITTI  variant=$variant  seed=$seed  (cikis $?)"
  done
done

echo ""
echo "TUM KOSULAR TAMAMLANDI"
