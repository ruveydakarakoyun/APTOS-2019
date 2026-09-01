"""Veri kalite raporu ve problemli goruntu listesi.

scan_images.py'nin urettigi tablodan turer; goruntuler yeniden okunmaz.
Kontrol ettikleri:

  - okunamayan / bozuk goruntuler
  - CSV'de olup klasorde olmayan, klasorde olup CSV'de olmayan kayitlar
  - asiri karanlik ve asiri parlak goruntuler
  - duplicate goruntuler (algisal hash ile)
  - split'ler arasi duplicate - skoru sisirebilecegi icin ayrica isaretlenir

Cikti: reports/data_quality.md + reports/problem_images.csv

Kullanim:
    python scripts/quality_report.py
"""
import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "bq" / "image_stats.csv"
LABELS = ROOT / "data" / "bq" / "aptos_labels.csv"
REPORTS = ROOT / "reports"

PROCESSED = ROOT / "data" / "processed"


def section(title):
    return f"\n## {title}\n\n"


def _thumb(id_code, split, cache={}):
    """Islenmis 512px JPEG'den 128px gri kucultme - ham PNG'leri okumaktan hizli."""
    key = (id_code, split)
    if key in cache:
        return cache[key]

    path = PROCESSED / split / f"{id_code}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype("float32")
    cache[key] = img
    return img


def _same_image(a, b, min_corr=0.995, max_mae=3.0):
    """Iki kucultmenin gercekten ayni goruntu olup olmadigi.

    Esikler bilincli olarak siki: dogrulama sirasinda olculen gercek ciftler
    korelasyon 1.0000 / MAE 0.00 veriyor, farkli retinalar ise 0.97-0.985
    araliginda kaliyordu. Aradaki bosluga esik koyduk.
    """
    if a is None or b is None:
        return False
    return (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) >= min_corr
            and float(np.abs(a - b).mean()) <= max_mae)


def _verify_duplicates(candidates):
    """dHash adaylarini piksel duzeyinde dogrular, gercek gruplari doner."""
    groups = []
    for _, g in candidates.groupby("dhash"):
        members = list(zip(g.id_code, g.split))
        # grup icinde gercekten esit olan alt kumeleri bul
        remaining = list(members)
        while len(remaining) > 1:
            head, rest = remaining[0], remaining[1:]
            matched = [head]
            leftover = []
            for other in rest:
                if _same_image(_thumb(*head), _thumb(*other)):
                    matched.append(other)
                else:
                    leftover.append(other)
            if len(matched) > 1:
                groups.append(matched)
            remaining = leftover
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", type=float, default=12.0)
    ap.add_argument("--bright", type=float, default=240.0)
    args = ap.parse_args()

    if not STATS.exists():
        raise SystemExit(f"{STATS} yok - once scripts/scan_images.py calistirin")

    stats = pd.read_csv(STATS)
    labels = pd.read_csv(LABELS)
    REPORTS.mkdir(exist_ok=True)

    md = ["# Veri Kalite Raporu\n",
          f"\nKaynak: {len(stats)} taranan goruntu, {len(labels)} etiket satiri.\n"]
    problems = []

    # ---------------------------------------------------------- okunabilirlik
    unreadable = stats[~stats.readable]
    md.append(section("Bozuk / acilmayan goruntuler"))
    md.append(f"- Taranan: **{len(stats)}**\n")
    md.append(f"- Okunamayan: **{len(unreadable)}**\n")
    for _, r in unreadable.iterrows():
        problems.append({"id_code": r.id_code, "split": r.split,
                         "sorun": "okunamadi", "deger": r.get("error")})

    ok = stats[stats.readable].copy()

    # ------------------------------------------------- etiket <-> dosya eslesme
    md.append(section("Etiket ve dosya eslesmesi"))
    lab_ids, img_ids = set(labels.id_code), set(ok.id_code)
    only_lab, only_img = lab_ids - img_ids, img_ids - lab_ids
    md.append(f"- Etikette olup goruntusu olmayan: **{len(only_lab)}**\n")
    md.append(f"- Goruntusu olup etiketi olmayan: **{len(only_img)}**\n")
    for i in sorted(only_lab):
        problems.append({"id_code": i, "split": "", "sorun": "goruntu yok", "deger": ""})
    for i in sorted(only_img):
        problems.append({"id_code": i, "split": "", "sorun": "etiket yok", "deger": ""})

    # islenmis klasorle de karsilastir
    if PROCESSED.exists():
        proc = {p.stem for split in ("train", "valid", "test")
                for p in (PROCESSED / split).glob("*.jpg")}
        md.append(f"- Islenmis klasorde eksik: **{len(lab_ids - proc)}**\n")

    # ------------------------------------------------------- parlaklik uclari
    dark = ok[ok.brightness < args.dark]
    bright = ok[ok.brightness > args.bright]
    md.append(section("Parlaklik kontrolu"))
    md.append(f"- Esikler: karanlik `< {args.dark}`, parlak `> {args.bright}`\n")
    md.append(f"- Asiri karanlik: **{len(dark)}**\n")
    md.append(f"- Asiri parlak: **{len(bright)}**\n")
    md.append(f"- Parlaklik araligi: {ok.brightness.min():.1f} - {ok.brightness.max():.1f}"
              f" (ortalama {ok.brightness.mean():.1f})\n")
    md.append(f"- En dusuk %1: {ok.brightness.quantile(0.01):.1f}, "
              f"en yuksek %1: {ok.brightness.quantile(0.99):.1f}\n")
    for _, r in dark.iterrows():
        problems.append({"id_code": r.id_code, "split": r.split,
                         "sorun": "asiri karanlik", "deger": r.brightness})
    for _, r in bright.iterrows():
        problems.append({"id_code": r.id_code, "split": r.split,
                         "sorun": "asiri parlak", "deger": r.brightness})

    # düşük kontrast - CLAHE'nin en cok fayda saglayacagi goruntuler
    low_c = ok[ok.contrast_std < ok.contrast_std.quantile(0.02)]
    md.append(f"- En dusuk %2 kontrast ({len(low_c)} goruntu): "
              f"std < {ok.contrast_std.quantile(0.02):.1f} - CLAHE'nin en cok "
              f"fayda saglayacagi grup\n")

    # ------------------------------------------------------------- duplicate
    md.append(section("Duplicate goruntuler"))
    candidates = ok.groupby("dhash").filter(lambda g: len(g) > 1)
    n_cand = candidates.dhash.nunique() if len(candidates) else 0

    md.append("Iki asamali tespit: dHash ucuz bir aday ureteci, ardindan her aday "
              "cifti piksel duzeyinde dogrulaniyor.\n\n")
    md.append(f"- dHash adayi grup: **{n_cand}**\n")

    # dHash tek basina yeterli degil: fundus goruntulerinin hepsi siyah zeminde
    # parlak bir daire oldugu icin farkli retinalar ayni hash'i uretebiliyor.
    # Adaylari 128px gri kucultmeler uzerinden korelasyon + ortalama mutlak fark
    # ile dogruluyoruz.
    confirmed, cross = [], 0
    if n_cand:
        verified_groups = _verify_duplicates(candidates)
        for members in verified_groups:
            splits = sorted({m[1] for m in members})
            if len(splits) > 1:
                cross += 1
            confirmed.append((members, splits))
            for id_code, split in members:
                problems.append({"id_code": id_code, "split": split,
                                 "sorun": "duplicate", "deger": "+".join(
                                     m[0] for m in members)})

        md.append(f"- Piksel duzeyinde dogrulanan grup: **{len(confirmed)}**\n")
        md.append(f"- Yanlis pozitif elenen: **{n_cand - len(confirmed)}**\n")
        md.append(f"- Etkilenen goruntu: **{sum(len(m) for m, _ in confirmed)}**\n")
        md.append(f"- Split'ler arasi: **{cross}**\n")

        if confirmed:
            md.append("\n| goruntuler | split'ler |\n|---|---|\n")
            for members, splits in confirmed:
                md.append(f"| {', '.join(m[0] for m in members)} | "
                          f"{', '.join(splits)} |\n")
        if cross:
            md.append("\n> Split'ler arasi duplicate skoru sisirir: ayni goruntu hem "
                      "egitimde hem testte gorunuyorsa test sonucu iyimser cikar. "
                      "Bunlarin egitimden cikarilmasi gerekir.\n")
    else:
        md.append("\nAday bulunmadi.\n")
    n_groups = len(confirmed)

    # ------------------------------------------------------------------ ozet
    md.insert(2, section("Ozet"))
    md.insert(3,
              f"| kontrol | sonuc |\n|---|---|\n"
              f"| Okunamayan goruntu | {len(unreadable)} |\n"
              f"| Etiketi olmayan goruntu | {len(only_img)} |\n"
              f"| Goruntusu olmayan etiket | {len(only_lab)} |\n"
              f"| Asiri karanlik | {len(dark)} |\n"
              f"| Asiri parlak | {len(bright)} |\n"
              f"| Duplicate grup (dogrulanmis) | {n_groups} |\n"
              f"| Split'ler arasi duplicate | {cross} |\n")

    (REPORTS / "data_quality.md").write_text("".join(md), encoding="utf-8")

    prob_df = pd.DataFrame(problems, columns=["id_code", "split", "sorun", "deger"])
    prob_df.to_csv(REPORTS / "problem_images.csv", index=False)

    print("".join(md))
    print(f"\n-> {REPORTS / 'data_quality.md'}")
    print(f"-> {REPORTS / 'problem_images.csv'}  ({len(prob_df)} satir)")


if __name__ == "__main__":
    main()
