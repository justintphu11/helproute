"""Train M1 and baselines using grouped cross-validation on labeled training data only."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

TYPE_COLUMNS = ["housing", "homelessness", "food", "medical", "job_income", "bills_debt", "animal"]


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def metrics(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "pr_auc": float(average_precision_score(y, p))}


def high_recall_threshold(y, p):
    candidates = np.unique(np.r_[0.0, p, 1.0])
    valid = [t for t in candidates if np.mean(np.asarray(p)[np.asarray(y) == 1] >= t) >= 0.90]
    return float(max(valid)) if valid else 0.0


def make_pipeline(c):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42)),
    ])


def main():
    project = root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=project / "data" / "processed" / "train_labels.parquet")
    ap.add_argument("--models", type=Path, default=project / "models")
    ap.add_argument("--report", type=Path, default=project / "reports" / "m1_cv.md")
    args = ap.parse_args()
    df = pd.read_parquet(args.data)
    y = df["is_need"].to_numpy(int)
    text = df["text"].fillna("").to_numpy()
    groups = df["dup_key"].fillna(df["post_id"]).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(text, y, groups))
    candidates = [0.3, 1, 3, 10]
    all_probs = {}
    for c in candidates:
        prob = np.zeros(len(df))
        for tr, va in splits:
            model = make_pipeline(c).fit(text[tr], y[tr])
            prob[va] = model.predict_proba(text[va])[:, 1]
        all_probs[c] = prob
    best_c = max(candidates, key=lambda c: average_precision_score(y, all_probs[c]))
    m1_prob = all_probs[best_c]
    threshold = high_recall_threshold(y, m1_prob)
    b0_prob = np.zeros(len(df))
    b1_prob = df["any_need"].astype(int).to_numpy()

    type_scores, type_models = {}, {}
    for col in TYPE_COLUMNS:
        positives = df[df["is_need"] == 1].copy()
        ty = positives[col].to_numpy(int)
        if ty.sum() < 5 or len(np.unique(ty)) < 2:
            type_scores[col] = {"status": "skipped", "positives": int(ty.sum())}
            continue
        n_splits = min(5, int(ty.sum()), int((1 - ty).sum()))
        tc = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        prob = np.zeros(len(positives))
        ptext = positives["text"].fillna("").to_numpy()
        pgroups = positives["dup_key"].fillna(positives["post_id"]).to_numpy()
        for tr, va in tc.split(ptext, ty, pgroups):
            model = make_pipeline(best_c).fit(ptext[tr], ty[tr])
            prob[va] = model.predict_proba(ptext[va])[:, 1]
        type_scores[col] = {"status": "ok", "positives": int(ty.sum()), "f1": float(f1_score(ty, prob >= 0.5, zero_division=0)), "pr_auc": float(average_precision_score(ty, prob))}
        type_models[col] = make_pipeline(best_c).fit(ptext, ty)

    subgroup = {}
    for name, mask in {"bay_local": df["sub_group"].eq("bay_local"), "housing_money": df["sub_group"].eq("housing_money"), "work": df["sub_group"].eq("work"), "bay": df["bay"].astype(bool), "national": ~df["bay"].astype(bool), "short": df["n_words"] < 50, "long": df["n_words"] >= 50}.items():
        if int(mask.sum()):
            subgroup[name] = metrics(y[mask], m1_prob[mask], threshold)["recall"]

    args.models.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(make_pipeline(best_c).fit(text, y), args.models / "m1_is_need.joblib")
    joblib.dump(type_models, args.models / "m1_types.joblib")
    (args.models / "m1_threshold.json").write_text(json.dumps({"threshold": threshold, "C": best_c}, indent=2))
    summary = {"B0": metrics(y, b0_prob, 0.5), "B1": metrics(y, b1_prob, 0.5), "M1": metrics(y, m1_prob, threshold)}
    lines = ["# M1 cross-validation", "", f"Best C: {best_c}; high-recall threshold: {threshold:.4f}.", "", "| Model | Precision | Recall | F1 | PR-AUC |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['pr_auc']:.3f} |" for k, v in summary.items()]
    lines += ["", "## Recall by subgroup", ""] + [f"- {k}: {v:.3f}" for k, v in subgroup.items()]
    lines += ["", "## Type models", ""] + [f"- {k}: {v}" for k, v in type_scores.items()]
    args.report.write_text("\n".join(lines) + "\n")
    pd.DataFrame({"post_id": df["post_id"], "y": y, "prob": m1_prob}).to_parquet(args.report.parent / "m1_oof.parquet", index=False)
    print(json.dumps({"best_C": best_c, "threshold": threshold, "metrics": summary, "recall_by_group": subgroup, "types": type_scores}, indent=2))


if __name__ == "__main__":
    main()
