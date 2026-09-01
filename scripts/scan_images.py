"""Ham retina goruntulerinin ozelliklerini tarar ve BigQuery'ye yazar.

Tek gecise her seyi cikarir: boyut, cozunurluk, en-boy orani, renk modu, kanal
sayisi, parlaklik/kontrast istatistikleri ve algisal hash. Kalite raporu ve
duplicate analizi (quality_report.py) bu tablodan turer - 8 GB'lik veri ikinci
kez okunmaz.

Cikti: BigQuery `aptos_image_stats` + data/bq/image_stats.csv

Kullanim:
    python scripts/scan_images.py
    python scripts/scan_images.py --no-bq
"""
import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import auto_crop, dhash  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCE_DIRS = {
    "train": ROOT / "data" / "images" / "train_images" / "train_images",
    "valid": ROOT / "data" / "images" / "val_images" / "val_images",
    "test": ROOT / "data" / "images" / "test_images" / "test_images",
}

PROJECT_ID = "datascientis"
BQ_DATASET = "APTOS_2019"


def scan_one(args):
    path, split = args
    row = {"id_code": path.stem, "split": split, "readable": False}

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        row["error"] = "okunamadi"
        return row

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Renk modu: uc kanal esitse dosya RGB olsa da icerik gri tonlamadir
    b, g, r = cv2.split(img)
    is_gray = bool((b == g).all() and (g == r).all())

    cropped = auto_crop(img)
    ch, cw = cropped.shape[:2]

    row.update({
        "readable": True,
        "error": None,
        "width": int(w),
        "height": int(h),
        "megapixels": round(w * h / 1e6, 3),
        "aspect_ratio": round(w / h, 4),
        "channels": int(img.shape[2]),
        "color_mode": "Grayscale" if is_gray else "RGB",
        "file_kb": round(path.stat().st_size / 1024, 1),
        "brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "black_ratio": round(float((gray <= 7).mean()), 4),
        "crop_width": int(cw),
        "crop_height": int(ch),
        "crop_saving": round(1 - (cw * ch) / (w * h), 4),
        "dhash": dhash(img),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()

    jobs = []
    for split, d in SOURCE_DIRS.items():
        if not d.exists():
            raise SystemExit(f"{d} yok - once Kaggle indirmesini tamamlayin")
        jobs += [(p, split) for p in sorted(d.glob("*.png"))]

    print(f"{len(jobs)} goruntu taraniyor, {args.workers} surec...")

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(scan_one, j) for j in jobs]):
            rows.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")

    df = pd.DataFrame(rows).sort_values(["split", "id_code"]).reset_index(drop=True)

    out = ROOT / "data" / "bq" / "image_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} satir -> {out}")

    ok = df[df.readable]
    print(f"\nokunabilir: {len(ok)}/{len(df)}")
    print(f"renk modu : {ok.color_mode.value_counts().to_dict()}")
    print(f"kanal     : {ok.channels.value_counts().to_dict()}")
    print("\ncozunurluk:")
    print(f"  genislik  {ok.width.min()}-{ok.width.max()}  (medyan {int(ok.width.median())})")
    print(f"  yukseklik {ok.height.min()}-{ok.height.max()}  (medyan {int(ok.height.median())})")
    print(f"  megapiksel {ok.megapixels.min():.2f}-{ok.megapixels.max():.2f}")
    print(f"  en-boy     {ok.aspect_ratio.min():.3f}-{ok.aspect_ratio.max():.3f}")
    print(f"  farkli cozunurluk sayisi: {ok.groupby(['width','height']).ngroups}")
    print("\nen sik 5 cozunurluk:")
    for (w, h), n in ok.groupby(["width", "height"]).size().nlargest(5).items():
        print(f"  {w}x{h}  {n}")
    print(f"\nparlaklik : {ok.brightness.min():.1f}-{ok.brightness.max():.1f} "
          f"(ort {ok.brightness.mean():.1f})")
    print(f"kontrast  : {ok.contrast_std.min():.1f}-{ok.contrast_std.max():.1f} "
          f"(ort {ok.contrast_std.mean():.1f})")
    print(f"auto-crop ile kazanilan alan: ortalama %{ok.crop_saving.mean() * 100:.1f}")

    if args.no_bq:
        return

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.aptos_image_stats"
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        clustering_fields=["split"],
    )
    client.load_table_from_dataframe(df, table_id, job_config=cfg).result()
    print(f"\n{table_id} <- {len(df)} satir")


if __name__ == "__main__":
    main()
