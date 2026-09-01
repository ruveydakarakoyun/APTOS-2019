"""APTOS goruntulerini egitime hazir hale getirir.

Ham PNG'ler ortalama 2 MB ve cogu retina disi siyah bosluk. Her epoch'ta bunlari
tekrar decode etmek egitimi I/O'ya baglar. Bu script bir kez calisir ve
scripts/preprocessing.py'deki ortak boru hattini her goruntuye uygular:

    Oku -> Kalite Kontrolu -> Auto-Crop -> CLAHE -> Kare -> Resize

Normalizasyon burada YAPILMAZ; egitimde torchvision donusumleri icinde yapilir
(train.py). Iki yerde birden normalize etmemek icin bilincli bir ayrim.

Girdi : data/images/{train_images/train_images, val_images/val_images,
                    test_images/test_images}/*.png
Cikti : data/processed/<split>/<id_code>.jpg          (--no-clahe ile)
        data/processed_clahe/<split>/<id_code>.jpg    (varsayilan)

Kullanim:
    python scripts/preprocess_images.py --size 512               # CLAHE'li
    python scripts/preprocess_images.py --size 512 --no-clahe    # kontrol grubu
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import preprocess  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Kaggle arsivinin klasor adlari tutarsiz: valid split'i "val_images" altinda.
SOURCE_DIRS = {
    "train": ROOT / "data" / "images" / "train_images" / "train_images",
    "valid": ROOT / "data" / "images" / "val_images" / "val_images",
    "test": ROOT / "data" / "images" / "test_images" / "test_images",
}


def process_one(args):
    src, dst, size, use_clahe, clip, square_mode = args
    img, info = preprocess(src, size=size, use_clahe=use_clahe,
                           normalize=False, clip_limit=clip,
                           quality_check=True, square_mode=square_mode)
    if img is None:
        return src.stem, info.get("error", "bilinmeyen hata")

    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return src.stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--square-mode", choices=["squash", "pad"], default="squash",
                    help="kareye getirme yontemi; squash siyah alani %28.5'ten "
                         "%13.4'e indirir ve doku kaybettirmez")
    ap.add_argument("--no-clahe", action="store_true",
                    help="CLAHE'siz uret - karsilastirmanin kontrol grubu")
    ap.add_argument("--out", default=None,
                    help="cikti klasoru; verilmezse CLAHE durumuna gore secilir")
    args = ap.parse_args()

    use_clahe = not args.no_clahe
    out_root = (ROOT / args.out if args.out else
                ROOT / "data" / ("processed_clahe" if use_clahe else "processed"))

    jobs = []
    for split, src_dir in SOURCE_DIRS.items():
        if not src_dir.exists():
            raise SystemExit(f"{src_dir} yok - once Kaggle indirmesini tamamlayin")
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(src_dir.glob("*.png")):
            jobs.append((png, out_dir / f"{png.stem}.jpg", args.size,
                         use_clahe, args.clip_limit, args.square_mode))

    print(f"{len(jobs)} goruntu, {args.size}x{args.size}, "
          f"CLAHE {'acik (clip=' + str(args.clip_limit) + ')' if use_clahe else 'KAPALI'}, "
          f"kare={args.square_mode}, {args.workers} surec")
    print(f"cikti: {out_root}")

    failures, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process_one, j) for j in jobs]):
            stem, err = fut.result()
            done += 1
            if err:
                failures.append((stem, err))
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")

    print(f"\nbitti: {done - len(failures)} basarili, {len(failures)} atlandi")
    for stem, err in failures[:20]:
        print(f"  {stem}: {err}")
    if len(failures) > 20:
        print(f"  ... ve {len(failures) - 20} tane daha")


if __name__ == "__main__":
    main()
