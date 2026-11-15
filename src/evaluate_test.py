"""Final, one-time evaluation on the locked test set (labeling/test_final.csv).

Nothing here is tuned on the test set: every threshold comes from training out-of-fold scores
(models/*_threshold.json; B2's from data/processed/zeroshot_train.parquet vs. gold labels).

Run once, after models are frozen:
    python src/zeroshot_screen.py --ids labeling/test_final.csv --out data/processed/zeroshot_test.parquet
    python src/evaluate_test.py
Outputs: reports/test_results.md, reports/test_predictions.parquet
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
P, M, R = ROOT / "data/processed", ROOT / "models", ROOT / "reports"
LOCKED_SHA = "c7b81f0fc5017462905f75dc50c9f1b47b92efdd808a29ddcacd7c49c9367708"


def hr_threshold(y, p):
    ok = [t for t in np.unique(np.r_[0, p, 1]) if (p[y == 1] >= t).mean() >= .9]
    return float(max(ok)) if ok else 0.0


def m2_probs(model_dir, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            out.extend(torch.softmax(model(**enc).logits, 1)[:, 1].cpu().numpy())
    return np.array(out)


def scores(y, pred, prob):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f, (average_precision_score(y, prob) if 0 < y.sum() < len(y) else np.nan)


def table(y, preds, probs, rng, B=1000):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    rows = []
    for m in preds:
        point = scores(y, preds[m], probs[m])
        bs = np.array([scores(y[i], preds[m][i], probs[m][i]) for i in idx])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        rows.append(f"| {m} | " + " | ".join(f"{v:.3f} [{a:.2f}, {b:.2f}]" for v, a, b in zip(point, lo, hi)) + " |")
    return ["| Model | Precision | Recall | F1 | PR-AUC |", "|---|---|---|---|---|", *rows]


def main():
    sha = hashlib.sha256((ROOT / "labeling/test_final.csv").read_bytes()).hexdigest()
    assert sha == LOCKED_SHA, "test_final.csv changed since it was locked"
    test = pd.read_csv(ROOT / "labeling/test_final.csv", dtype={"post_id": str})
    key = pd.read_csv(ROOT / "labeling/test_key.csv", dtype={"post_id": str})
    clean = pd.read_parquet(P / "reddit_clean.parquet", columns=["post_id", "text", "any_need", "n_words"])
    clean["post_id"] = clean["post_id"].astype(str)
    d = (test[["post_id", "is_need", "L1_is_need", "L2_is_need"]]
         .merge(key[["post_id", "sub_group", "bay", "sample_source"]], on="post_id", how="left")
         .merge(clean, on="post_id", how="left"))
    assert d["text"].notna().all() and len(d) == 200
    y = d["is_need"].astype(int).to_numpy()
    texts = d["text"].fillna("").tolist()

    # Thresholds: all from training data.
    gold = pd.read_parquet(P / "train_labels.parquet")
    zs_tr = pd.read_parquet(P / "zeroshot_train.parquet").merge(gold[["post_id", "is_need"]], on="post_id")
    th = {"B2": hr_threshold(zs_tr["is_need"].to_numpy(int), zs_tr["zs_need_score"].to_numpy()),
          "M1": json.loads((M / "m1_threshold.json").read_text())["threshold"],
          "M2": json.loads((M / "m2_threshold.json").read_text())["threshold"],
          "M2+silver": json.loads((M / "m2_silver_threshold.json").read_text())["threshold"]}

    zs = pd.read_parquet(P / "zeroshot_test.parquet")[["post_id", "zs_need_score"]]
    probs = {"B0": np.zeros(len(d)), "B1": d["any_need"].astype(float).to_numpy(),
             "B2": d[["post_id"]].merge(zs, on="post_id", how="left")["zs_need_score"].to_numpy(),
             "M1": joblib.load(M / "m1_is_need.joblib").predict_proba(texts)[:, 1],
             "M2": m2_probs(M / "m2_distilroberta", texts),
             "M2+silver": m2_probs(M / "m2_silver_distilroberta", texts)}
    assert not np.isnan(probs["B2"]).any(), "run zeroshot_screen.py on the test ids first"
    preds = {m: (probs[m] >= th.get(m, .5)).astype(int) for m in probs}

    rng = np.random.default_rng(0)
    agreed = (d["L1_is_need"] == d["L2_is_need"]).to_numpy()
    L = ["# Final test-set results (locked test set, evaluated once)", "",
         f"test_final.csv SHA-256 verified: {sha[:16]}…", "",
         f"Thresholds (from training data only): " + ", ".join(f"{k}={v:.4f}" for k, v in th.items()) + "; B0/B1 = 0.5.", "",
         f"## All 200 test posts (needs: {y.sum()}, union of two labelers)", "",
         *table(y, preds, probs, rng), "",
         f"## Posts both labelers agreed on ({agreed.sum()}; needs: {y[agreed].sum()})", "",
         *table(y[agreed], {m: v[agreed] for m, v in preds.items()}, {m: v[agreed] for m, v in probs.items()}, rng), "",
         "## Recall by subgroup (all 200)", "",
         "| Group | Posts | Needs | " + " | ".join(preds) + " |", "|---|---:|---:|" + "---:|" * len(preds)]
    groups = {**{f"sub_group={g}": d["sub_group"].eq(g) for g in ["bay_local", "housing_money", "work"]},
              "Bay Area": d["bay"].astype(bool), "national": ~d["bay"].astype(bool),
              "keyword hit": d["any_need"].astype(bool), "no keyword": ~d["any_need"].astype(bool),
              "short (<50 words)": d["n_words"] < 50, "long (>=50 words)": d["n_words"] >= 50}
    for g, mask in groups.items():
        mk = mask.to_numpy() & (y == 1)
        L.append(f"| {g} | {int(mask.sum())} | {int(mk.sum())} | " +
                 " | ".join(f"{preds[m][mk].mean():.2f}" if mk.any() else "–" for m in preds) + " |")
    nk = (~d["any_need"].astype(bool)).to_numpy() & (y == 1)
    L += ["", f"Needs with no keyword: {int(nk.sum())} of {int(y.sum())} ({nk.sum() / y.sum():.0%}). "
          f"The keyword rule (B1) misses all of them; M2+silver catches {preds['M2+silver'][nk].mean():.0%}."]
    R.mkdir(exist_ok=True)
    (R / "test_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame({"post_id": d["post_id"], "y": y, "agreed": agreed, "sub_group": d["sub_group"],
                  **{f"prob_{m}": v for m, v in probs.items()}, **{f"pred_{m}": v for m, v in preds.items()}}
                 ).to_parquet(R / "test_predictions.parquet", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
