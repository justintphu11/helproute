"""Fine-tune DistilRoBERTa for binary need detection without touching test data.

Scored with the same 5-fold grouped CV splits as train_m1.py (StratifiedGroupKFold,
shuffle, seed 42, groups = near-duplicate key), so M1 and M2 out-of-fold numbers are
directly comparable. Epochs are fixed (default 3, per the plan) rather than picked on
validation, so the CV numbers are not optimistic. A final model is then trained on all
labeled rows and saved.

Outputs: models/m2_distilroberta/, models/m2_threshold.json, reports/m2_cv.md,
         reports/m2_oof.parquet (post_id, y, prob)
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

MODEL = "distilroberta-base"


class Posts(Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {**{k: torch.tensor(v[i]) for k, v in self.enc.items()}, "labels": self.y[i]}


def threshold(y, p):
    """Highest threshold that still gives >=90% recall."""
    choices = np.unique(np.r_[0, p, 1]); ok = [t for t in choices if (p[y == 1] >= t).mean() >= .9]
    return float(max(ok)) if ok else 0.0


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def subset(enc, idx):
    return {k: np.asarray(v)[idx] for k, v in enc.items()}


def fit(enc, y, idx, epochs, device, seed=42):
    seed_all(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    pos = y[idx].sum()
    weights = torch.tensor([1, (len(idx) - pos) / pos], dtype=torch.float, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    data = Posts(subset(enc, idx), y[idx])
    for _ in range(epochs):
        model.train()
        for batch in DataLoader(data, batch_size=16, shuffle=True):
            labels = batch.pop("labels").to(device); batch = {k: v.to(device) for k, v in batch.items()}
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, labels, weight=weights)
            loss.backward(); opt.step(); opt.zero_grad()
    return model


def predict(model, enc, idx, device):
    model.eval(); probs = []
    data = Posts(subset(enc, idx), np.zeros(len(idx), dtype=int))
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=32):
            batch.pop("labels"); batch = {k: v.to(device) for k, v in batch.items()}
            probs.extend(torch.softmax(model(**batch).logits, 1)[:, 1].cpu().numpy())
    return np.array(probs)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=root / "data/processed/train_labels.parquet")
    ap.add_argument("--out", type=Path, default=root / "models/m2_distilroberta")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--silver", type=Path, default=None,
                    help="silver_labels.parquet: added to every TRAINING fold only; scoring stays gold-only")
    args = ap.parse_args()

    df = pd.read_parquet(args.data); n_gold = len(df)
    groups = df.dup_key.fillna(df.post_id).to_numpy()
    silver_idx = np.array([], dtype=int)
    if args.silver:
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_dataset import dup_key
        s = pd.read_parquet(args.silver)
        s = s[s["error"].isna()] if "error" in s else s
        s["post_id"] = s["post_id"].astype(str); s["dup_key"] = dup_key(s["text"])
        test_ids = set(pd.read_csv(root / "labeling/test_final.csv", usecols=["post_id"], dtype=str)["post_id"])
        clean = pd.read_parquet(root / "data/processed/reddit_clean.parquet", columns=["post_id", "text"])
        clean["post_id"] = clean["post_id"].astype(str)
        test_keys = set(dup_key(clean.loc[clean["post_id"].isin(test_ids), "text"]))
        bad = s["post_id"].isin(test_ids) | s["dup_key"].isin(test_keys) | s["post_id"].isin(set(df["post_id"].astype(str))) | s["dup_key"].isin(set(df["dup_key"]))
        print(f"silver: {len(s)} rows, dropped {int(bad.sum())} matching test or gold posts", flush=True)
        s = s[~bad]
        silver_idx = np.arange(n_gold, n_gold + len(s))
        df = pd.concat([df[["post_id", "text", "is_need"]], s[["post_id", "text", "is_need"]]], ignore_index=True)
    y = df.is_need.astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; gold={n_gold}; silver={len(silver_idx)}; folds={args.folds}; epochs={args.epochs}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(df.text.fillna("").tolist(), truncation=True, padding=True, max_length=256)

    # Same gold splits as train_m1.py: identical y, groups and CV settings. Silver joins each training fold.
    gy = y[:n_gold]
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros(n_gold)
    for k, (tr, va) in enumerate(cv.split(df["text"][:n_gold].fillna("").to_numpy(), gy, groups), 1):
        model = fit(enc, y, np.r_[tr, silver_idx], args.epochs, device)
        oof[va] = predict(model, enc, va, device)
        print(f"fold={k} val_n={len(va)} val_pr_auc={average_precision_score(y[va], oof[va]):.3f}", flush=True)
        del model; torch.cuda.empty_cache()

    # Threshold and metrics on gold only.
    t = threshold(gy, oof)
    prec, rec, f1, _ = precision_recall_fscore_support(gy, oof >= t, average="binary", zero_division=0)
    pr_auc = average_precision_score(gy, oof)
    print(f"oof (gold) precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} pr_auc={pr_auc:.3f} threshold={t:.4f}", flush=True)

    tag = "m2_silver" if args.silver else "m2"
    out = args.out if not args.silver else args.out.parent / "m2_silver_distilroberta"
    model = fit(enc, y, np.arange(len(df)), args.epochs, device)
    out.mkdir(parents=True, exist_ok=True); model.save_pretrained(out); tok.save_pretrained(out)
    out.parent.joinpath(f"{tag}_threshold.json").write_text(json.dumps({"threshold": t, "epochs": args.epochs, "silver_rows": int(len(silver_idx))}, indent=2))
    (root / "reports").mkdir(exist_ok=True)
    pd.DataFrame({"post_id": df.post_id[:n_gold], "y": gy, "prob": oof}).to_parquet(root / f"reports/{tag}_oof.parquet", index=False)
    (root / f"reports/{tag}_cv.md").write_text(
        f"# {tag} cross-validation\n\n{args.folds}-fold grouped CV on {n_gold} gold posts (same splits as M1), "
        f"{len(silver_idx)} silver posts added to every training fold, {args.epochs} epochs fixed.\n\n"
        f"Precision {prec:.3f}; recall {rec:.3f}; F1 {f1:.3f}; PR-AUC {pr_auc:.3f}; threshold {t:.4f}.\n")


if __name__ == "__main__":
    main()
