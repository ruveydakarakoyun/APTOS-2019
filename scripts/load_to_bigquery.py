"""Hazirlanan APTOS-2019 CSV'lerini BigQuery'ye yukler.

Onkosul:
    pip install google-cloud-bigquery
    python scripts/prepare_bq_csv.py
    Kimlik dogrulama: gcloud auth application-default login
                      veya  set GOOGLE_APPLICATION_CREDENTIALS=...\key.json

Kullanim:
    python scripts/load_to_bigquery.py --project PROJE_ID [--dataset aptos2019]
                                       [--location EU] [--prefix aptos_]
"""
import argparse
import pathlib

from google.cloud import bigquery

ROOT = pathlib.Path(__file__).resolve().parent.parent
BQ_DIR = ROOT / "data" / "bq"

SCHEMA = [
    bigquery.SchemaField("id_code", "STRING", mode="REQUIRED",
                         description="Fundus goruntusunun Kaggle kimligi (dosya adi = id_code.png)"),
    bigquery.SchemaField("diagnosis", "INT64", mode="REQUIRED",
                         description="ICDRSS DR siddet derecesi, 0-4"),
    bigquery.SchemaField("diagnosis_label", "STRING", mode="REQUIRED",
                         description="Derecenin metin karsiligi"),
    bigquery.SchemaField("is_referable", "BOOL", mode="REQUIRED",
                         description="diagnosis >= 2, sevk gerektiren DR"),
    bigquery.SchemaField("split", "STRING", mode="REQUIRED",
                         description="train / valid / test"),
    bigquery.SchemaField("image_file", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED",
                         description="Kaggle arsivi icindeki goreli goruntu yolu"),
]

# tablo adi -> kaynak CSV
TABLES = {
    "labels": "aptos_labels.csv",
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", default="aptos2019")
    ap.add_argument("--location", default="EU", help="EU, US veya europe-west1 gibi")
    ap.add_argument("--prefix", default="", help="tablo adi oneki, orn. aptos_")
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)

    ds_ref = bigquery.Dataset(f"{args.project}.{args.dataset}")
    ds_ref.location = args.location
    ds_ref.description = "APTOS-2019 Blindness Detection etiketleri (Kaggle: mariaherrerot/aptos2019)"
    dataset = client.create_dataset(ds_ref, exists_ok=True)
    print(f"dataset hazir: {dataset.full_dataset_id} ({dataset.location})")

    for table, csv_name in TABLES.items():
        path = BQ_DIR / csv_name
        if not path.exists():
            raise SystemExit(f"{path} yok - once scripts/prepare_bq_csv.py calistirin")

        table_id = f"{args.project}.{args.dataset}.{args.prefix}{table}"
        job_config = bigquery.LoadJobConfig(
            schema=SCHEMA,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # 3662 satir kucuk; kumeleme ileride goruntu tablolariyla JOIN'de ise yarar
            clustering_fields=["split", "diagnosis"],
        )
        with path.open("rb") as fh:
            client.load_table_from_file(fh, table_id, job_config=job_config).result()

        loaded = client.get_table(table_id)
        print(f"  {table_id}: {loaded.num_rows} satir")

    q = f"""
        SELECT split, diagnosis, diagnosis_label, COUNT(*) AS n
        FROM `{args.project}.{args.dataset}.{args.prefix}labels`
        GROUP BY split, diagnosis, diagnosis_label
        ORDER BY split, diagnosis
    """
    print("\ndogrulama sorgusu:")
    for row in client.query(q).result():
        print(f"  {row.split:<6} {row.diagnosis} {row.diagnosis_label:<17} {row.n}")


if __name__ == "__main__":
    main()
