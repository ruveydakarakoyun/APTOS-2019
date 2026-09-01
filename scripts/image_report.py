"""Goruntu ozellikleri ozetini markdown rapor olarak yazar.

scan_images.py'nin urettigi tablodan turer; goruntuler yeniden okunmaz.
Cozunurluk, en-boy orani, renk modu ve piksel istatistiklerini derler.

Girdi : data/bq/image_stats.csv   (scripts/scan_images.py uretir)
Cikti : reports/image_properties.md

Kullanim:
    python scripts/image_report.py
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "bq" / "image_stats.csv"
REPORTS = ROOT / "reports"


def build(df):
    ok = df[df.readable]
    res = ok.groupby(["width", "height"]).size().sort_values(ascending=False)
    kare = int((ok.aspect_ratio.round(3) == 1.0).sum())

    md = []
    md.append("# Goruntu Ozellikleri Ozeti\n\n")
    md.append(f"Ham APTOS-2019 fundus goruntuleri, n={len(df)}.\n")

    md.append("\n## Okunabilirlik ve renk modu\n\n")
    md.append("| kontrol | sonuc |\n|---|---|\n")
    md.append(f"| Okunabilir | {len(ok)} / {len(df)} |\n")
    renk = ", ".join(f"{k}: {v}" for k, v in ok.color_mode.value_counts().items())
    kanal = ", ".join(f"{k}: {v}" for k, v in ok.channels.value_counts().items())
    md.append(f"| Renk modu | {renk} |\n")
    md.append(f"| Kanal sayisi | {kanal} |\n")
    md.append("\nRenk modu dosya bicimine degil icerige gore belirlendi: uc kanali\n"
              "birebir esit olan bir goruntu RGB kaydedilmis olsa da gri tonlamadir.\n")

    md.append("\n## Cozunurluk\n\n")
    md.append("| olcu | en kucuk | medyan | en buyuk |\n|---|---|---|---|\n")
    md.append(f"| Genislik | {ok.width.min()} | {int(ok.width.median())} | {ok.width.max()} |\n")
    md.append(f"| Yukseklik | {ok.height.min()} | {int(ok.height.median())} | {ok.height.max()} |\n")
    md.append(f"| Megapiksel | {ok.megapixels.min():.2f} | {ok.megapixels.median():.2f} "
              f"| {ok.megapixels.max():.2f} |\n")
    md.append(f"| En-boy orani | {ok.aspect_ratio.min():.3f} | {ok.aspect_ratio.median():.3f} "
              f"| {ok.aspect_ratio.max():.3f} |\n")

    md.append(f"\nToplam **{len(res)} farkli cozunurluk** var. En sik gorulenler:\n\n")
    md.append("| cozunurluk | adet | oran |\n|---|---|---|\n")
    for (w, h), n in res.head(8).items():
        md.append(f"| {w}x{h} | {n} | %{n / len(ok) * 100:.1f} |\n")

    md.append(f"\n{kare} goruntu (%{kare / len(ok) * 100:.1f}) zaten kare; geri kalani "
              "resize oncesi kareye tamamlanir. Cozunurlugun bu kadar degisken olmasi "
              "sabit boyuta getirmeyi zorunlu kiliyor.\n")

    md.append("\n## Piksel istatistikleri\n\n")
    md.append("| olcu | en dusuk | ortalama | en yuksek |\n|---|---|---|---|\n")
    md.append(f"| Parlaklik | {ok.brightness.min():.1f} | {ok.brightness.mean():.1f} "
              f"| {ok.brightness.max():.1f} |\n")
    md.append(f"| Kontrast (std) | {ok.contrast_std.min():.1f} | {ok.contrast_std.mean():.1f} "
              f"| {ok.contrast_std.max():.1f} |\n")
    md.append(f"| Siyah piksel orani | {ok.black_ratio.min():.3f} | {ok.black_ratio.mean():.3f} "
              f"| {ok.black_ratio.max():.3f} |\n")
    md.append(f"\nParlaklik hicbir goruntude 130'un uzerine cikmiyor; fundus fotograflari\n"
              f"dogasi geregi koyu. Sabit bir \"asiri parlak\" esigi (orn. 240) bu veri\n"
              f"setinde hicbir seyi elemez, yuzdelik tabanli esik daha anlamli.\n")

    md.append("\n## Auto-crop kazanci\n\n")
    md.append(f"Retina disindaki siyah cerceve kirpildiginda alanin ortalama "
              f"**%{ok.crop_saving.mean() * 100:.1f}**'i atiliyor "
              f"(en az %{ok.crop_saving.min() * 100:.1f}, "
              f"en cok %{ok.crop_saving.max() * 100:.1f}).\n")

    md.append("\n---\n\nGrafikler: `reports/figures/06_image_properties.png`. "
              "Goruntu basina ayrintili veri: BigQuery `aptos_image_stats`.\n")
    return "".join(md)


def main():
    if not STATS.exists():
        raise SystemExit(f"{STATS} yok - once scripts/scan_images.py calistirin")

    REPORTS.mkdir(exist_ok=True)
    text = build(pd.read_csv(STATS))
    (REPORTS / "image_properties.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {REPORTS / 'image_properties.md'}")


if __name__ == "__main__":
    main()
