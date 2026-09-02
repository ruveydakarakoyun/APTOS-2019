"""Rapor ve sunum icin gorselleri uretir.

Uretilenler (reports/figures/ altina):
  01_class_distribution.png    sinif dagilimi, uc split yan yana
  02_class_imbalance.png       dengesizlik orani ve yuzdeler
  03_autocrop_<evre>.png       auto-crop oncesi/sonrasi, her evreden bir ornek
  04_clahe_<evre>.png          CLAHE oncesi/sonrasi, her evreden bir ornek
  05_pipeline_stages.png       boru hattinin alti asamasi tek gorselde
  06_image_properties.png      cozunurluk, en-boy orani, parlaklik dagilimlari
  07_resolution_confound.png   cozunurluk-sinif kisayol iliskisi

Kullanim:
    python scripts/make_figures.py
"""
import pathlib
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocessing import apply_clahe, auto_crop, to_square  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
LABELS = ROOT / "data" / "bq" / "aptos_labels.csv"
STATS = ROOT / "data" / "bq" / "image_stats.csv"

SOURCE_DIRS = {
    "train": ROOT / "data" / "images" / "train_images" / "train_images",
    "valid": ROOT / "data" / "images" / "val_images" / "val_images",
    "test": ROOT / "data" / "images" / "test_images" / "test_images",
}

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
# ICDRSS siddet rampasi - raporda ve grafiklerde ayni renkler
COLORS = ["#15803D", "#A16207", "#C2410C", "#B91C1C", "#7B1D1D"]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.facecolor": "white",
})


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sample_per_grade(labels, split="train"):
    """Her evreden tekrarlanabilir bir ornek secer."""
    sub = labels[labels.split == split]
    out = {}
    for g in range(5):
        rows = sub[sub.diagnosis == g].sort_values("id_code")
        if len(rows):
            out[g] = rows.iloc[len(rows) // 2]["id_code"]
    return out


# ------------------------------------------------------------ 1. kisi: sinif

def fig_class_distribution(labels):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, split in zip(axes, ["train", "valid", "test"]):
        counts = (labels[labels.split == split].diagnosis
                  .value_counts().reindex(range(5), fill_value=0).sort_index())
        bars = ax.bar([f"{i}" for i in range(5)], counts.values, color=COLORS)
        total = counts.sum()
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.015,
                    f"{v}\n%{v / total * 100:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title(f"{split}  (n={total})")
        ax.set_xlabel("ICDRSS evresi")
        ax.margins(y=0.18)
    axes[0].set_ylabel("goruntu sayisi")
    fig.suptitle("Sinif dagilimi", fontsize=13, y=1.0)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS], GRADES,
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(FIG / "01_class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_imbalance(labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    splits = ["train", "valid", "test"]
    ratios = []
    for split in splits:
        c = labels[labels.split == split].diagnosis.value_counts()
        ratios.append(c.max() / c.min())
    bars = ax1.bar(splits, ratios, color="#0F766E", width=0.55)
    for bar, v in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.2f}x",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Sinif dengesizlik orani (en cok / en az)")
    ax1.set_ylabel("kat")
    ax1.margins(y=0.2)

    bottom = np.zeros(3)
    for g in range(5):
        vals = np.array([
            (labels[(labels.split == s) & (labels.diagnosis == g)].shape[0]
             / labels[labels.split == s].shape[0] * 100) for s in splits])
        ax2.barh(splits, vals, left=bottom, color=COLORS[g], label=f"{g} {GRADES[g]}")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax2.text(b + v / 2, i, f"%{v:.0f}", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
        bottom += vals
    ax2.set_title("Sinif yuzdeleri")
    ax2.set_xlabel("%")
    ax2.set_xlim(0, 100)
    ax2.grid(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
               fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG / "02_class_imbalance.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ 1. ve 2. kisi: once/sonra

def fig_autocrop(samples):
    for grade, id_code in samples.items():
        path = SOURCE_DIRS["train"] / f"{id_code}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        cropped = auto_crop(raw)
        saving = (1 - (cropped.shape[0] * cropped.shape[1])
                  / (raw.shape[0] * raw.shape[1])) * 100

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        for ax, im, t in [(axes[0], raw, f"Ham  {raw.shape[1]}x{raw.shape[0]}"),
                          (axes[1], cropped,
                           f"Auto-crop  {cropped.shape[1]}x{cropped.shape[0]}")]:
            ax.imshow(bgr2rgb(im))
            ax.set_title(t, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Auto-crop  |  Evre {grade} - {GRADES[grade]}  |  "
                     f"kazanilan alan %{saving:.1f}",
                     fontsize=11.5, color=COLORS[grade], y=1.0)
        fig.tight_layout()
        fig.savefig(FIG / f"03_autocrop_evre{grade}.png", bbox_inches="tight")
        plt.close(fig)


def fig_clahe(samples):
    for grade, id_code in samples.items():
        path = SOURCE_DIRS["train"] / f"{id_code}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        cropped = auto_crop(raw)
        enhanced = apply_clahe(cropped)

        s_before = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).std()
        s_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).std()

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        for ax, im, t in [(axes[0], cropped, f"Auto-crop sonrasi  (kontrast {s_before:.1f})"),
                          (axes[1], enhanced, f"CLAHE sonrasi  (kontrast {s_after:.1f})")]:
            ax.imshow(bgr2rgb(im))
            ax.set_title(t, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"CLAHE  |  Evre {grade} - {GRADES[grade]}", fontsize=11.5,
                     color=COLORS[grade], y=1.0)
        fig.tight_layout()
        fig.savefig(FIG / f"04_clahe_evre{grade}.png", bbox_inches="tight")
        plt.close(fig)


def fig_pipeline(samples):
    """Boru hattinin her asamasini tek gorselde gosterir."""
    grade = 2 if 2 in samples else next(iter(samples))
    path = SOURCE_DIRS["train"] / f"{samples[grade]}.png"
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        return

    def siyah_oran(im):
        return (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) <= 7).mean() * 100

    cropped = auto_crop(raw)
    enhanced = apply_clahe(cropped)
    squashed = to_square(enhanced, 512, mode="squash")
    padded = to_square(enhanced, 512, mode="pad")
    norm = (cv2.cvtColor(squashed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    stages = [
        (bgr2rgb(raw), f"1. Oku\n{raw.shape[1]}x{raw.shape[0]}"),
        (bgr2rgb(cropped), f"2. Kalite + auto-crop\n{cropped.shape[1]}x{cropped.shape[0]}"),
        (bgr2rgb(enhanced), "3. CLAHE\nLAB-L kanali"),
        (bgr2rgb(padded), f"(pad ile olsaydi)\nsiyah %{siyah_oran(padded):.0f}"),
        (bgr2rgb(squashed), f"4-5. Squash + resize\n512x512, siyah %{siyah_oran(squashed):.0f}"),
        (np.clip((norm - norm.min()) / (norm.max() - norm.min()), 0, 1),
         "6. Normalizasyon\nImageNet ort/std"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(17, 3.4))
    for ax, (im, title) in zip(axes, stages):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"On isleme boru hatti  |  Evre {grade} - {GRADES[grade]}",
                 fontsize=12.5, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "05_pipeline_stages.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 2. kisi: goruntu ozellikleri

def fig_image_properties(stats):
    ok = stats[stats.readable]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    axes[0, 0].hist(ok.width, bins=40, color="#0F766E", alpha=.85)
    axes[0, 0].set_title("Genislik dagilimi")
    axes[0, 0].set_xlabel("piksel")

    axes[0, 1].hist(ok.aspect_ratio, bins=40, color="#0F766E", alpha=.85)
    axes[0, 1].axvline(1.0, color="#B91C1C", ls="--", lw=1.2, label="kare (1.0)")
    axes[0, 1].set_title("En-boy orani")
    axes[0, 1].legend(frameon=False, fontsize=9)

    axes[1, 0].hist(ok.brightness, bins=40, color="#A16207", alpha=.85)
    axes[1, 0].set_title("Ortalama parlaklik")
    axes[1, 0].set_xlabel("0-255")

    axes[1, 1].hist(ok.contrast_std, bins=40, color="#A16207", alpha=.85)
    axes[1, 1].set_title("Kontrast (standart sapma)")
    axes[1, 1].set_xlabel("0-255")

    fig.suptitle(f"Ham goruntu ozellikleri  (n={len(ok)})", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "06_image_properties.png", bbox_inches="tight")
    plt.close(fig)


def fig_confound(stats, labels):
    """Cozunurluk ile sinif arasindaki kisayol iliskisi.

    Bu grafik projenin en onemli bulgusunu gosteriyor: 1050x1050 goruntuler
    neredeyse tamamen No DR. Model retinaya bakmadan bile bu iliskiden
    faydalanabilir.
    """
    m = stats[stats.readable].merge(labels[["id_code", "diagnosis"]], on="id_code")
    sq = m[(m.width == 1050) & (m.height == 1050)]
    rest = m[~((m.width == 1050) & (m.height == 1050))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    x = np.arange(5)
    w = 0.38
    a = [(sq.diagnosis == g).mean() * 100 for g in range(5)]
    b = [(rest.diagnosis == g).mean() * 100 for g in range(5)]
    ax1.bar(x - w / 2, a, w, label=f"1050x1050  (n={len(sq)})", color="#B91C1C")
    ax1.bar(x + w / 2, b, w, label=f"diger cozunurlukler  (n={len(rest)})", color="#0F766E")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax1.text(i - w / 2, va + 1.5, f"{va:.0f}", ha="center", fontsize=8.5)
        ax1.text(i + w / 2, vb + 1.5, f"{vb:.0f}", ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}\n{GRADES[i].split()[0]}" for i in range(5)], fontsize=8.5)
    ax1.set_ylabel("split icindeki oran (%)")
    ax1.set_title("Cozunurluk sinifi belli ediyor")
    ax1.legend(frameon=False, fontsize=9)
    ax1.margins(y=0.15)

    # megapiksel dagilimi, sinif bazinda
    data = [m[m.diagnosis == g].megapixels.values for g in range(5)]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.6,
                     medianprops=dict(color="white", linewidth=1.5))
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c)
    ax2.set_xticklabels([f"{i}" for i in range(5)])
    ax2.set_xlabel("ICDRSS evresi")
    ax2.set_ylabel("megapiksel")
    ax2.set_title("Goruntu boyutu evreye gore degisiyor")

    fig.suptitle("Meta-veri kisayolu: retinaya bakmadan QWK 0.652",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "07_resolution_confound.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    samples = sample_per_grade(labels)

    print("sinif grafikleri...")
    fig_class_distribution(labels)
    fig_imbalance(labels)

    print("auto-crop oncesi/sonrasi...")
    fig_autocrop(samples)

    print("CLAHE oncesi/sonrasi...")
    fig_clahe(samples)

    print("boru hatti asamalari...")
    fig_pipeline(samples)

    if STATS.exists():
        stats = pd.read_csv(STATS)
        print("goruntu ozellikleri...")
        fig_image_properties(stats)
        print("cozunurluk kisayolu...")
        fig_confound(stats, labels)
    else:
        print("image_stats.csv yok - 06 ve 07 atlandi (once scan_images.py)")

    files = sorted(FIG.glob("*.png"))
    print(f"\n{len(files)} gorsel -> {FIG}")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
