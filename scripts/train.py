"""APTOS-2019 diyabetik retinopati siniflandirmasi.

Etiketler BigQuery'den, goruntuler data/processed/ altindaki 512px JPEG'lerden
okunur (once scripts/preprocess_images.py calistirin).

Iki mod:
  cls  5 sinifli softmax, sinif agirlikli cross-entropy
  reg  tek cikisli regresyon + valid uzerinde optimize edilen esikler

Neden QWK: veri %49 "No DR". Hicbir sey ogrenmeyen "hep 0 tahmin et" modeli
%49 accuracy alir ama QWK'da 0 alir. Yarismanin resmi metrigi de buydu.

Kullanim:
    python scripts/train.py --mode reg --epochs 30 --patience 5
    python scripts/train.py --data-dir data/processed_clahe --variant clahe
    python scripts/train.py --mode cls --model resnet50 --no-bq
"""
import argparse
import datetime as dt
import pathlib
import random
import uuid

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

PROJECT_ID = "datascientis"
BQ_DATASET = "APTOS_2019"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    """Kosuyu tekrarlanabilir yapar - ayni seed, ayni sonuc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------- veri

def load_labels():
    """Etiketleri BigQuery'den ceker; erisim yoksa yerel CSV'ye duser."""
    try:
        from google.cloud import bigquery
        q = f"""SELECT id_code, diagnosis, split
                FROM `{PROJECT_ID}.{BQ_DATASET}.aptos_labels`"""
        df = bigquery.Client(project=PROJECT_ID).query(q).to_dataframe()
        print(f"etiketler BigQuery'den alindi: {len(df)} satir")
    except Exception as e:
        print(f"BigQuery okunamadi ({type(e).__name__}), yerel CSV kullaniliyor")
        df = pd.read_csv(ROOT / "data" / "bq" / "aptos_labels.csv")
    return df


class APTOSDataset(Dataset):
    def __init__(self, df, split, transform, root=DEFAULT_PROCESSED):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.dir = pathlib.Path(root) / split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.dir / f"{row['id_code']}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(int(row["diagnosis"]))


def build_transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),          # fundus goruntusunun ust/alt yonu anlamli degil
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


# ------------------------------------------------------------- metrikler

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_thresholds(y_true, raw):
    """Regresyon ciktisini 0-4'e cevirecek esikleri valid uzerinde arar."""
    best = np.array([0.5, 1.5, 2.5, 3.5])
    best_score = qwk(y_true, np.digitize(raw, best))
    for _ in range(60):
        improved = False
        for i in range(4):
            for delta in (-0.12, -0.04, 0.04, 0.12):
                cand = best.copy()
                cand[i] += delta
                if not np.all(np.diff(cand) > 0.05):
                    continue
                score = qwk(y_true, np.digitize(raw, cand))
                if score > best_score:
                    best, best_score, improved = cand, score, True
        if not improved:
            break
    return best, best_score


# ---------------------------------------------------------------- egitim

def run_epoch(model, loader, criterion, device, mode, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, trues = 0.0, [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
            target = labels.float().unsqueeze(1) if mode == "reg" else labels

            with torch.autocast("cuda", enabled=scaler is not None):
                out = model(imgs)
                loss = criterion(out, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            raw = out.squeeze(1) if mode == "reg" else out.argmax(1)
            preds.append(raw.float().detach().cpu().numpy())
            trues.append(labels.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


# ------------------------------------------------------------- BigQuery

def write_to_bigquery(run, preds_df, metrics_df):
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

    for name, df in [("aptos_predictions", preds_df), ("aptos_metrics", metrics_df)]:
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{name}"
        client.load_table_from_dataframe(df, table_id, job_config=cfg).result()
        print(f"  {table_id} <- {len(df)} satir  (run_id={run})")


# ------------------------------------------------------------------ ana

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cls", "reg"], default="reg")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5,
                    help="valid QWK bu kadar epoch iyilesmezse dur; 0 = kapali")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="iyilesme sayilmasi icin gereken en kucuk QWK artisi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="split basina goruntu siniri - sadece hizli deneme icin")
    ap.add_argument("--data-dir", default=None,
                    help="islenmis goruntu klasoru; verilmezse data/processed")
    ap.add_argument("--exclude-leaked", action="store_true",
                    help="valid/test'te kopyasi olan egitim goruntulerini disla "
                         "(reports/leaked_train_ids.csv)")
    ap.add_argument("--variant", default=None,
                    help="kosu etiketi, orn. clahe / baseline")
    ap.add_argument("--author", default="senanur")
    ap.add_argument("--no-bq", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("UYARI: GPU bulunamadi, CPU ile calisacak (cok yavas)")
    else:
        print(f"cihaz: {torch.cuda.get_device_name(0)}")

    data_root = pathlib.Path(args.data_dir) if args.data_dir else DEFAULT_PROCESSED
    if not data_root.exists():
        raise SystemExit(f"{data_root} yok - once scripts/preprocess_images.py calistirin")
    variant = args.variant or ("clahe" if "clahe" in data_root.name else "baseline")
    print(f"veri: {data_root}  (variant={variant})")

    df = load_labels()

    # Split'ler arasi duplicate: ayni goruntu hem egitimde hem degerlendirmede
    # varsa test sonucu iyimser cikar. Egitim tarafini duseriz - degerlendirme
    # kumeleri boylece bozulmadan kalir ve kosular karsilastirilabilir olur.
    if args.exclude_leaked:
        leak_file = ROOT / "reports" / "leaked_train_ids.csv"
        if not leak_file.exists():
            raise SystemExit(f"{leak_file} yok - once scripts/quality_report.py calistirin")
        leaked = set(pd.read_csv(leak_file).id_code)
        before = len(df)
        df = df[~((df.split == "train") & (df.id_code.isin(leaked)))].reset_index(drop=True)
        print(f"sizinti temizligi: {before - len(df)} egitim goruntusu dislandi")

    if args.limit:
        df = (df.sample(frac=1, random_state=args.seed)
                .groupby("split", group_keys=False).head(args.limit).reset_index(drop=True))
        print(f"UYARI: --limit {args.limit} etkin, bu bir duman testidir, gercek sonuc degildir")

    train_tf, eval_tf = build_transforms(args.size)
    loaders = {}
    for split, tf, shuffle in [("train", train_tf, True),
                               ("valid", eval_tf, False),
                               ("test", eval_tf, False)]:
        ds = APTOSDataset(df, split, tf, root=data_root)
        loaders[split] = DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                                    num_workers=args.workers, pin_memory=True,
                                    persistent_workers=args.workers > 0)
        print(f"  {split}: {len(ds)} goruntu")

    n_out = 1 if args.mode == "reg" else 5
    model = timm.create_model(args.model, pretrained=True, num_classes=n_out).to(device)

    if args.mode == "reg":
        criterion = nn.MSELoss()
    else:
        counts = df[df.split == "train"].diagnosis.value_counts().sort_index().values
        weights = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32).to(device)
        print("sinif agirliklari:", np.round(weights.cpu().numpy(), 2))
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    MODELS.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    best_qwk, best_state, best_thr = -1.0, None, None
    best_epoch, sabirsiz, son_epoch = 0, 0, 0

    print(f"\nrun_id={run_id}  mod={args.mode}  model={args.model}  {args.size}px"
          f"  erken_durdurma={args.patience or 'kapali'}\n")
    for epoch in range(1, args.epochs + 1):
        son_epoch = epoch
        tr_loss, _, _ = run_epoch(model, loaders["train"], criterion, device,
                                  args.mode, optimizer, scaler)
        va_loss, va_raw, va_true = run_epoch(model, loaders["valid"], criterion,
                                             device, args.mode)

        if args.mode == "reg":
            thr, va_qwk = optimize_thresholds(va_true, va_raw)
        else:
            thr, va_qwk = None, qwk(va_true, va_raw.astype(int))

        scheduler.step()
        flag = ""
        if va_qwk > best_qwk + args.min_delta:
            best_qwk, best_thr, best_epoch = va_qwk, thr, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sabirsiz, flag = 0, "  <- en iyi"
        else:
            sabirsiz += 1
            if args.patience:
                flag = f"  ({sabirsiz}/{args.patience})"

        print(f"epoch {epoch:>2}/{args.epochs}  train_loss={tr_loss:.4f}  "
              f"valid_loss={va_loss:.4f}  valid_QWK={va_qwk:.4f}{flag}")

        # Erken durdurma: valid QWK patience epoch boyunca iyilesmezse kes.
        # Model doygunluga ulastiktan sonraki epoch'lar ezberlemeye gider ve
        # en iyi checkpoint zaten saklandigi icin devam etmenin faydasi yok.
        if args.patience and sabirsiz >= args.patience:
            print(f"\nerken durduruldu: valid QWK {args.patience} epoch boyunca "
                  f"iyilesmedi (en iyi epoch {best_epoch}, QWK {best_qwk:.4f})")
            break

    model.load_state_dict(best_state)
    ckpt = MODELS / f"{run_id}_{args.model}_{args.mode}.pt"
    torch.save({"state_dict": best_state, "thresholds": best_thr, "args": vars(args)}, ckpt)
    print(f"\nen iyi valid QWK: {best_qwk:.4f}  -> {ckpt.name}")

    # ------- son degerlendirme: valid ve test
    created = dt.datetime.now(dt.timezone.utc)
    pred_rows, metric_rows = [], []

    for split in ("valid", "test"):
        _, raw, true = run_epoch(model, loaders[split], criterion, device, args.mode)
        pred = np.digitize(raw, best_thr) if args.mode == "reg" else raw.astype(int)

        ids = df[df.split == split].reset_index(drop=True)["id_code"]
        pred_rows.append(pd.DataFrame({
            "run_id": run_id, "author": args.author, "model": args.model,
            "mode": args.mode, "split": split, "id_code": ids,
            "y_true": true.astype(int), "y_pred": pred.astype(int),
            "raw_score": raw.astype(float), "created_at": created,
        }))

        s_qwk = qwk(true, pred)
        metric_rows.append({
            "run_id": run_id, "author": args.author, "model": args.model,
            "mode": args.mode, "split": split, "qwk": s_qwk,
            "accuracy": float((true == pred).mean()),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "n": int(len(true)), "epochs": son_epoch, "best_epoch": best_epoch,
            "variant": variant, "leak_excluded": bool(args.exclude_leaked),
            "img_size": args.size,
            "seed": args.seed,
            "created_at": created,
        })
        print(f"\n{split.upper()}  QWK={s_qwk:.4f}  "
              f"acc={(true == pred).mean():.4f}  "
              f"macro_F1={f1_score(true, pred, average='macro', zero_division=0):.4f}")
        print(confusion_matrix(true, pred, labels=range(5)))

    # --limit bir duman testidir; sonuclari gercek kosularla ayni tabloya
    # yazmak karsilastirmalari kirletir. Yazmak icin --no-bq'yu kaldirmak
    # yetmez, --limit'i de kaldirmak gerekir.
    if args.limit:
        print("\n--limit etkin: sonuclar BigQuery'ye YAZILMADI "
              "(duman testi gercek kosularla ayni tabloya girmesin diye).")
    elif not args.no_bq:
        print("\nBigQuery'ye yaziliyor...")
        try:
            write_to_bigquery(run_id, pd.concat(pred_rows, ignore_index=True),
                              pd.DataFrame(metric_rows))
        except Exception as e:
            print(f"  yazilamadi: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
