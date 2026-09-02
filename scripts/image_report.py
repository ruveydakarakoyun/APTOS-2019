"""Write the image-properties summary as a markdown report.

Derives from the table produced by scan_images.py; images are not read again.
Covers resolution, aspect ratio, colour mode and pixel statistics.

Input : data/bq/image_stats.csv   (produced by scripts/scan_images.py)
Output: reports/image_properties.md

Usage:
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
    square = int((ok.aspect_ratio.round(3) == 1.0).sum())

    md = []
    md.append("# Image Properties Summary\n\n")
    md.append(f"Raw APTOS-2019 fundus photographs, n={len(df)}.\n")

    md.append("\n## Readability and colour mode\n\n")
    md.append("| check | result |\n|---|---|\n")
    md.append(f"| Readable | {len(ok)} / {len(df)} |\n")
    modes = ", ".join(f"{k}: {v}" for k, v in ok.color_mode.value_counts().items())
    channels = ", ".join(f"{k}: {v}" for k, v in ok.channels.value_counts().items())
    md.append(f"| Colour mode | {modes} |\n")
    md.append(f"| Channels | {channels} |\n")
    md.append("\nColour mode is judged by content rather than file format: an "
              "image whose three channels are identical is grayscale even if it "
              "was stored as RGB.\n")

    md.append("\n## Resolution\n\n")
    md.append("| measure | min | median | max |\n|---|---|---|---|\n")
    md.append(f"| Width | {ok.width.min()} | {int(ok.width.median())} | {ok.width.max()} |\n")
    md.append(f"| Height | {ok.height.min()} | {int(ok.height.median())} | {ok.height.max()} |\n")
    md.append(f"| Megapixels | {ok.megapixels.min():.2f} | {ok.megapixels.median():.2f} "
              f"| {ok.megapixels.max():.2f} |\n")
    md.append(f"| Aspect ratio | {ok.aspect_ratio.min():.3f} | {ok.aspect_ratio.median():.3f} "
              f"| {ok.aspect_ratio.max():.3f} |\n")

    md.append(f"\nThere are **{len(res)} distinct resolutions**. The most common:\n\n")
    md.append("| resolution | count | share |\n|---|---|---|\n")
    for (w, h), n in res.head(8).items():
        md.append(f"| {w}x{h} | {n} | {n / len(ok) * 100:.1f}% |\n")

    md.append(f"\n{square} images ({square / len(ok) * 100:.1f}%) are already "
              "square; the rest are brought to a square before resizing. This "
              "much variation in resolution makes a fixed input size mandatory.\n")

    md.append("\n## Pixel statistics\n\n")
    md.append("| measure | min | mean | max |\n|---|---|---|---|\n")
    md.append(f"| Brightness | {ok.brightness.min():.1f} | {ok.brightness.mean():.1f} "
              f"| {ok.brightness.max():.1f} |\n")
    md.append(f"| Contrast (std) | {ok.contrast_std.min():.1f} | {ok.contrast_std.mean():.1f} "
              f"| {ok.contrast_std.max():.1f} |\n")
    md.append(f"| Black pixel ratio | {ok.black_ratio.min():.3f} | {ok.black_ratio.mean():.3f} "
              f"| {ok.black_ratio.max():.3f} |\n")
    md.append("\nBrightness never exceeds 130 in this dataset; fundus photographs "
              "are inherently dark. A fixed \"too bright\" cutoff such as 240 "
              "would never fire here, which is why the quality report also uses "
              "a distribution-based outlier test.\n")

    md.append("\n## Auto-crop saving\n\n")
    md.append(f"Removing the black frame outside the retina discards "
              f"**{ok.crop_saving.mean() * 100:.1f}%** of the area on average "
              f"(min {ok.crop_saving.min() * 100:.1f}%, "
              f"max {ok.crop_saving.max() * 100:.1f}%).\n")

    md.append("\n---\n\nCharts: `reports/figures/06_image_properties.png`. "
              "Per-image detail: BigQuery `aptos_image_stats`.\n")
    return "".join(md)


def main():
    if not STATS.exists():
        raise SystemExit(f"{STATS} not found - run scripts/scan_images.py first")

    REPORTS.mkdir(exist_ok=True)
    text = build(pd.read_csv(STATS))
    (REPORTS / "image_properties.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {REPORTS / 'image_properties.md'}")


if __name__ == "__main__":
    main()
