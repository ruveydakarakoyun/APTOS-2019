"""Load the prepared APTOS-2019 label CSVs into BigQuery.

Prerequisites:
    pip install google-cloud-bigquery
    python scripts/prepare_bq_csv.py
    Authentication: gcloud auth application-default login
                    or  export GOOGLE_APPLICATION_CREDENTIALS=.../key.json

Usage:
    python scripts/load_to_bigquery.py --project PROJECT_ID [--dataset aptos2019]
                                       [--location EU] [--prefix aptos_]
"""
import argparse
import pathlib

from google.cloud import bigquery

ROOT = pathlib.Path(__file__).resolve().parent.parent
BQ_DIR = ROOT / "data" / "bq"

SCHEMA = [
    bigquery.SchemaField("id_code", "STRING", mode="REQUIRED",
                         description="Kaggle image id; the file is <id_code>.png"),
    bigquery.SchemaField("diagnosis", "INT64", mode="REQUIRED",
                         description="ICDRSS DR severity grade, 0-4"),
    bigquery.SchemaField("diagnosis_label", "STRING", mode="REQUIRED",
                         description="Human-readable grade"),
    bigquery.SchemaField("is_referable", "BOOL", mode="REQUIRED",
                         description="diagnosis >= 2, referable DR"),
    bigquery.SchemaField("split", "STRING", mode="REQUIRED",
                         description="train / valid / test"),
    bigquery.SchemaField("image_file", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("image_uri", "STRING", mode="REQUIRED",
                         description="Full gs:// path to the image"),
]

# table name -> source CSV
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
    ap.add_argument("--location", default="EU", help="EU, US, europe-west1, ...")
    ap.add_argument("--prefix", default="", help="table name prefix, e.g. aptos_")
    args = ap.parse_args()

    client = bigquery.Client(project=args.project)

    ds_ref = bigquery.Dataset(f"{args.project}.{args.dataset}")
    ds_ref.location = args.location
    ds_ref.description = ("APTOS-2019 Blindness Detection labels "
                          "(Kaggle: mariaherrerot/aptos2019)")
    dataset = client.create_dataset(ds_ref, exists_ok=True)
    print(f"dataset ready: {dataset.full_dataset_id} ({dataset.location})")

    for table, csv_name in TABLES.items():
        path = BQ_DIR / csv_name
        if not path.exists():
            raise SystemExit(f"{path} not found - run scripts/prepare_bq_csv.py first")

        table_id = f"{args.project}.{args.dataset}.{args.prefix}{table}"
        job_config = bigquery.LoadJobConfig(
            schema=SCHEMA,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # 3662 rows is small; clustering pays off later when joining
            # against prediction tables.
            clustering_fields=["split", "diagnosis"],
        )
        with path.open("rb") as fh:
            client.load_table_from_file(fh, table_id, job_config=job_config).result()

        loaded = client.get_table(table_id)
        print(f"  {table_id}: {loaded.num_rows} rows")

    q = f"""
        SELECT split, diagnosis, diagnosis_label, COUNT(*) AS n
        FROM `{args.project}.{args.dataset}.{args.prefix}labels`
        GROUP BY split, diagnosis, diagnosis_label
        ORDER BY split, diagnosis
    """
    print("\nverification query:")
    for row in client.query(q).result():
        print(f"  {row.split:<6} {row.diagnosis} {row.diagnosis_label:<17} {row.n}")


if __name__ == "__main__":
    main()
