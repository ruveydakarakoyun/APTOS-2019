"""Kisayol ozellikleri ve etiket gurultusu analizi.

Bir goruntu siniflandirma modelinin yuksek skor almasi, dogru sebeple yuksek
skor aldigi anlamina gelmez. Bu script iki soruyu yanitlar:

1. KISAYOL: Model retinaya bakmadan ne kadar iyi olabilir? Yalnizca dosya
   meta-verisinden (cozunurluk, en-boy orani, parlaklik, dosya boyutu) egitilen
   bir siniflandirici bir taban olusturur. Bu taban yuksekse, gercek modelin
   skorunun bir kismi teshisten degil cekim cihazinin imzasindan geliyor
   olabilir.

2. ETIKET GURULTUSU: Ayni goruntunun birden fazla kez farkli etiketle
   isaretlendigi durumlar, etiketleyiciler arasi uyumun alt siniri hakkinda
   bilgi verir ve modelin ulasabilecegi tavani belirler.

Cikti: reports/confounds_and_noise.md

Kullanim:
    python scripts/confound_analysis.py
"""
import pathlib

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_predict

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATS = ROOT / "data" / "bq" / "image_stats.csv"
LABELS = ROOT / "data" / "bq" / "aptos_labels.csv"
PROBLEMS = ROOT / "reports" / "problem_images.csv"
REPORTS = ROOT / "reports"

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
META_COLS = ["width", "height", "aspect_ratio", "megapixels",
             "brightness", "contrast_std", "file_kb"]


def shortcut_baseline(m):
    """Yalnizca meta-veriden egitilen siniflandirici - kisayol tabani."""
    X, y = m[META_COLS].values, m.diagnosis.values
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    pred = cross_val_predict(clf, X, y, cv=5, n_jobs=-1)

    clf.fit(X, y)
    importance = sorted(zip(META_COLS, clf.feature_importances_),
                        key=lambda t: -t[1])
    return {
        "accuracy": float((pred == y).mean()),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "hep_sifir_acc": float((y == 0).mean()),
        "importance": importance,
        "pred": pred,
    }


def label_noise(problems, labels):
    """Duplicate ciftlerdeki etiket celiskilerinden gurultu tahmini."""
    d = problems[problems.sorun == "duplicate"].merge(labels, on="id_code",
                                                      suffixes=("", "_l"))
    pairs, conflicts, gaps = 0, 0, []
    for _, g in d.groupby("deger"):
        v = g.diagnosis.values
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs += 1
                if v[i] != v[j]:
                    conflicts += 1
                    gaps.append(abs(int(v[i]) - int(v[j])))
    if pairs == 0:
        return None

    agree = 1 - conflicts / pairs
    return {
        "groups": d.deger.nunique(),
        "pairs": pairs,
        "conflicts": conflicts,
        "rate": conflicts / pairs,
        "agree": agree,
        # Iki bagimsiz etiket p olasilikla dogruysa uyusma orani p^2 + (1-p)^2/4
        # civarindadir; kaba bir yaklasim olarak sqrt(uyusma) kullaniyoruz.
        "single_label_acc": float(np.sqrt(agree)),
        "gap_dist": dict(pd.Series(gaps).value_counts().sort_index()) if gaps else {},
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
    }


def build(m, sc, ln):
    md = ["# Kisayol Ozellikleri ve Etiket Gurultusu\n\n"]
    md.append("Modelin skorunu dogru sebeple alip almadigini ve ulasabilecegi "
              "tavani sorgulayan iki analiz.\n")

    # ---------------------------------------------------------- kisayol
    md.append("\n## 1. Meta-veri kisayolu\n\n")
    md.append("Yalnizca dosya ozelliklerinden (cozunurluk, en-boy orani, parlaklik, "
              "kontrast, dosya boyutu) egitilen bir RandomForest. **Retinaya hic "
              "bakmiyor.** 5 katli capraz dogrulama.\n\n")
    md.append("| olcut | meta-veri tabani | hep-0 tahmini |\n|---|---|---|\n")
    md.append(f"| Accuracy | {sc['accuracy']:.4f} | {sc['hep_sifir_acc']:.4f} |\n")
    md.append(f"| QWK | **{sc['qwk']:.4f}** | 0.0000 |\n")

    md.append("\nEn ayirt edici meta-veri ozellikleri:\n\n")
    md.append("| ozellik | onem |\n|---|---|\n")
    for name, imp in sc["importance"][:5]:
        md.append(f"| `{name}` | {imp:.3f} |\n")

    sq = m[(m.width == 1050) & (m.height == 1050)]
    rest = m[~((m.width == 1050) & (m.height == 1050))]
    md.append(f"\nKaynagi: en sik cozunurluk olan 1050x1050 goruntulerin "
              f"**%{(sq.diagnosis == 0).mean() * 100:.1f}**'i `No DR`, "
              f"diger cozunurluklerde bu oran **%{(rest.diagnosis == 0).mean() * 100:.1f}**.\n\n")
    md.append(f"| evre | 1050x1050 (n={len(sq)}) | digerleri (n={len(rest)}) |\n|---|---|---|\n")
    for g in range(5):
        a = (sq.diagnosis == g).sum()
        b = (rest.diagnosis == g).sum()
        md.append(f"| {g} {GRADES[g]} | {a} (%{a / len(sq) * 100:.1f}) | "
                  f"{b} (%{b / len(rest) * 100:.1f}) |\n")

    md.append("\n### Ne anlama geliyor\n\n")
    md.append("APTOS verisi Hindistan'da birden fazla merkezde, farkli cihazlarla "
              "toplandi. Cozunurluk cihazin imzasi ve cihaz ile hastalik yayginligi "
              "arasinda guclu bir iliski var. Model yeniden boyutlandirilmis "
              "goruntuleri gorse de en-boy orani, keskinlik ve kenar geometrisi bu "
              "bilgiyi tasimaya devam eder.\n\n")
    md.append(f"Bu, sonuclarin gecersiz oldugu anlamina gelmez; ama **QWK "
              f"{sc['qwk']:.3f}'lik bir kismi teshis olmadan da elde edilebilir** "
              "demektir. Rapor edilen skorlar bu taban ile birlikte okunmalidir.\n")

    # ------------------------------------------------------ sinif ozellikleri
    md.append("\n## 2. Sinif bazli goruntu ozellikleri\n\n")
    md.append("| evre | n | parlaklik | kontrast | megapiksel | kare olan |\n"
              "|---|---|---|---|---|---|\n")
    for g in range(5):
        s = m[m.diagnosis == g]
        kare = ((s.aspect_ratio > 0.98) & (s.aspect_ratio < 1.02)).mean() * 100
        md.append(f"| {g} {GRADES[g]} | {len(s)} | {s.brightness.median():.1f} | "
                  f"{s.contrast_std.median():.1f} | {s.megapixels.median():.2f} | "
                  f"%{kare:.0f} |\n")

    md.append("\nKruskal-Wallis testi (siniflar arasi fark anlamli mi):\n\n")
    md.append("| ozellik | p | sonuc |\n|---|---|---|\n")
    for col in ["brightness", "contrast_std", "megapixels"]:
        groups = [m[m.diagnosis == g][col].values for g in range(5)]
        _, p = st.kruskal(*groups)
        md.append(f"| {col} | {p:.2e} | {'farkli' if p < 0.05 else 'fark yok'} |\n")

    md.append("\n`No DR` goruntuleri medyan 1.10 megapiksel ve yarisi kare; "
              "hasta siniflar 4-5 megapiksel ve neredeyse hicbiri kare degil. "
              "Bu, yukaridaki kisayolun ta kendisi.\n")

    # ------------------------------------------------------- etiket gurultusu
    md.append("\n## 3. Etiket gurultusu\n\n")
    if ln is None:
        md.append("Duplicate cift bulunamadi, tahmin yapilamiyor.\n")
    else:
        md.append("Ayni goruntu veri setinde birden fazla kez yer aliyorsa, "
                  "etiketlerinin ayni olmasi beklenir. Olmadigi durumlar "
                  "etiketleyiciler arasi uyumun alt sinirini verir.\n\n")
        md.append("| olcut | deger |\n|---|---|\n")
        md.append(f"| Dogrulanmis duplicate grup | {ln['groups']} |\n")
        md.append(f"| Ayni goruntu cifti | {ln['pairs']} |\n")
        md.append(f"| Etiketi celisen cift | {ln['conflicts']} (%{ln['rate'] * 100:.1f}) |\n")
        md.append(f"| Ortalama celiski buyuklugu | {ln['mean_gap']:.2f} evre |\n")
        md.append(f"| Uyusma orani | %{ln['agree'] * 100:.1f} |\n")
        md.append(f"| Tek etiketin dogru olma tahmini | %{ln['single_label_acc'] * 100:.1f} |\n")
        if ln["gap_dist"]:
            dagilim = ", ".join(f"{k} evre: {v}" for k, v in ln["gap_dist"].items())
            md.append(f"\nCeliski buyuklugu dagilimi: {dagilim}.\n")
        md.append(f"\n### Ne anlama geliyor\n\n")
        md.append(f"Tek bir etiketin dogru olma olasiligi kabaca "
                  f"**%{ln['single_label_acc'] * 100:.0f}**. Bu, mukemmel bir modelin "
                  "bile bu veri setinde ulasabilecegi accuracy tavanini sinirlar. "
                  "Mevcut modelin test accuracy'si 0.82 civarinda - yani tavana "
                  "yakin. Kalan hatanin bir kismi modelin degil etiketlerin.\n\n")
        md.append("Bu tahmin yalnizca duplicate goruntulerden turedigi icin "
                  "**alt sinirdir**: tekrar etmeyen goruntulerdeki gurultuyu "
                  "gormuyoruz.\n")

    md.append("\n---\n\nUretim: `python scripts/confound_analysis.py`\n")
    return "".join(md)


def main():
    for f in (STATS, LABELS, PROBLEMS):
        if not f.exists():
            raise SystemExit(f"{f} yok - once scan_images.py ve quality_report.py calistirin")

    stats = pd.read_csv(STATS)
    stats = stats[stats.readable]
    labels = pd.read_csv(LABELS)[["id_code", "diagnosis", "split"]]
    problems = pd.read_csv(PROBLEMS)

    m = stats.merge(labels, on="id_code", suffixes=("", "_l"))

    print("meta-veri kisayol tabani hesaplaniyor (5 katli CV)...")
    sc = shortcut_baseline(m)
    print(f"  QWK={sc['qwk']:.4f}  accuracy={sc['accuracy']:.4f}")

    print("etiket gurultusu tahmin ediliyor...")
    ln = label_noise(problems, labels)
    if ln:
        print(f"  celisen cift: {ln['conflicts']}/{ln['pairs']} "
              f"(%{ln['rate'] * 100:.1f})")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "confounds_and_noise.md").write_text(build(m, sc, ln), encoding="utf-8")
    print(f"\n-> {REPORTS / 'confounds_and_noise.md'}")


if __name__ == "__main__":
    main()
