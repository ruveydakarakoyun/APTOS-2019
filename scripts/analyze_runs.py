"""Egitim kosularini BigQuery uzerinden karsilastirir ve hata analizi yapar.

train.py her kosuyu aptos_predictions / aptos_metrics tablolarina yazar. Bu script
o tablolari okur; boylece Colab'da ve yerelde yapilan kosular ayni yerden
karsilastirilir.

Kullanim:
    python scripts/analyze_runs.py                 # tum kosulari listele
    python scripts/analyze_runs.py --run 3f9a2b1c  # tek kosuyu incele
"""
import argparse

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "datascientis"
BQ_DATASET = "APTOS_2019"
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

client = bigquery.Client(project=PROJECT_ID)


def q(sql):
    return client.query(sql.format(p=PROJECT_ID, d=BQ_DATASET)).to_dataframe()


def leaderboard():
    """Tum kosular, valid QWK'ya gore sirali."""
    df = q("""
        SELECT run_id, author, model, mode, img_size, epochs, seed,
               MAX(IF(split='valid', qwk, NULL))      AS valid_qwk,
               MAX(IF(split='test',  qwk, NULL))      AS test_qwk,
               MAX(IF(split='valid', accuracy, NULL)) AS valid_acc,
               MAX(IF(split='valid', macro_f1, NULL)) AS valid_f1,
               MAX(created_at) AS created_at
        FROM `{p}.{d}.aptos_metrics`
        GROUP BY run_id, author, model, mode, img_size, epochs, seed
        ORDER BY valid_qwk DESC
    """)
    if df.empty:
        print("henuz kayitli kosu yok")
        return df
    print("=== KOSULAR (valid QWK'ya gore) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def confusion(run_id, split):
    df = q(f"""
        SELECT y_true, y_pred, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND split = '{split}'
        GROUP BY y_true, y_pred
    """)
    if df.empty:
        return None
    m = (df.pivot(index="y_true", columns="y_pred", values="n")
           .reindex(index=range(5), columns=range(5), fill_value=0)
           .fillna(0).astype(int))
    m.index = [f"gercek {g}" for g in GRADES]
    m.columns = [f"tah.{i}" for i in range(5)]
    return m


def per_class(run_id, split):
    """Sinif bazli duyarlilik ve en sik karistirildigi sinif."""
    df = q(f"""
        SELECT y_true, y_pred, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND split = '{split}'
        GROUP BY y_true, y_pred
    """)
    rows = []
    for grade in range(5):
        sub = df[df.y_true == grade]
        total = sub.n.sum()
        if total == 0:
            continue
        correct = sub[sub.y_pred == grade].n.sum()
        wrong = sub[sub.y_pred != grade]
        worst = wrong.loc[wrong.n.idxmax()] if not wrong.empty else None
        rows.append({
            "sinif": f"{grade} {GRADES[grade]}",
            "n": int(total),
            "duyarlilik": round(correct / total, 3),
            "en_cok_karisti": (f"{int(worst.y_pred)} ({int(worst.n)})"
                               if worst is not None else "-"),
        })
    return pd.DataFrame(rows)


def severe_missed(run_id):
    """Klinik olarak en pahali hata: ciddi DR'yi saglam sanmak."""
    df = q(f"""
        SELECT split, COUNT(*) n
        FROM `{{p}}.{{d}}.aptos_predictions`
        WHERE run_id = '{run_id}' AND y_true >= 3 AND y_pred <= 1
        GROUP BY split
    """)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run_id; verilmezse en iyi valid QWK'li kosu")
    args = ap.parse_args()

    board = leaderboard()
    if board.empty:
        return

    run_id = args.run or board.iloc[0]["run_id"]
    print(f"\n{'=' * 60}\nDETAY: run_id={run_id}\n{'=' * 60}")

    for split in ("valid", "test"):
        m = confusion(run_id, split)
        if m is None:
            continue
        print(f"\n--- {split.upper()} confusion matrix ---")
        print(m.to_string())
        print(f"\n--- {split.upper()} sinif bazli ---")
        print(per_class(run_id, split).to_string(index=False))

    missed = severe_missed(run_id)
    print("\n--- KACIRILAN CIDDI VAKA (gercek >=3, tahmin <=1) ---")
    print(missed.to_string(index=False) if not missed.empty
          else "  yok - hicbir ciddi vaka saglam olarak siniflandirilmadi")


if __name__ == "__main__":
    main()
