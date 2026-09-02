#!/usr/bin/env bash
# squash dogrulamasi: suren CV bitince ayni yontemle squash setini kosar.
#
# Karsilastirma tek degiskenli: ayni model, ayni seed, ayni 5 kat, ayni
# sizinti dislama. Tek fark kareye getirme yontemi (pad -> squash).
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore

CV_LOG="$1"
MANIFEST="data/processed_squash/_manifest.json"

echo "bekleniyor: squash goruntu seti..."
until [ -f "$MANIFEST" ]; do sleep 10; done
echo "  hazir: $(python -c "import json;m=json.load(open('$MANIFEST'));print(m['n_images'],'goruntu, kare=',m['square_mode'])")"

echo "bekleniyor: suren capraz dogrulama..."
until grep -q "TUM CV KOSULARI TAMAMLANDI" "$CV_LOG" 2>/dev/null; do sleep 30; done
echo "  CV bitti, GPU serbest"

echo ""
echo "=================================================="
echo "SQUASH DOGRULAMASI BASLIYOR"
echo "=================================================="
python -u scripts/train_cv.py --folds 5 --model efficientnet_b0 --size 384 \
  --batch 16 --epochs 15 --patience 5 --lr 3e-4 --workers 4 --seed 42 \
  --data-dir data/processed_squash --variant baseline-squash \
  --exclude-leaked --author senanur
echo "SQUASH DOGRULAMASI BITTI (cikis $?)"
