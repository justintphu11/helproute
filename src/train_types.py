"""Need-type models (multi-label) trained on silver + gold needs. Cross-validated only; the
test set is already used and is not touched. Output: models/types_silver.joblib, reports/types_cv.md"""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "data/processed"
TYPES = ["housing", "homelessness", "food", "medical", "job_income", "bills_debt", "animal"]
gold = pd.read_parquet(P / "train_labels.parquet"); gold = gold[gold.is_need == 1]
silver = pd.read_parquet(P / "silver_labels.parquet"); silver = silver[silver.is_need.fillna(False).astype(bool)]
test_ids = set(pd.read_csv(ROOT / "labeling/test_final.csv", usecols=["post_id"], dtype=str).post_id)
silver = silver[~silver.post_id.astype(str).isin(test_ids)]
df = pd.concat([gold[["text", *TYPES]].assign(src="gold"), silver[["text", *TYPES]].assign(src="silver")], ignore_index=True)
df[TYPES] = df[TYPES].fillna(0).astype(int)
pipe = lambda: Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")),
                         ("clf", LogisticRegression(C=3, class_weight="balanced", max_iter=2000))])
lines = ["# Need-type models (silver + gold needs, 5-fold CV)", "", f"{len(df)} need posts ({(df.src=='gold').sum()} gold, {(df.src=='silver').sum()} silver).", "",
         "| Type | Positives | Precision | Recall | F1 | F1 on gold rows |", "|---|---:|---:|---:|---:|---:|"]
models = {}
for t in TYPES:
    y = df[t].to_numpy()
    p = cross_val_predict(pipe(), df.text.fillna(""), y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    g = (df.src == "gold").to_numpy()
    lines.append(f"| {t} | {y.sum()} | {precision_score(y, p >= .5, zero_division=0):.2f} | {recall_score(y, p >= .5):.2f} | {f1_score(y, p >= .5):.2f} | {f1_score(y[g], p[g] >= .5, zero_division=0):.2f} |")
    models[t] = pipe().fit(df.text.fillna(""), y)
joblib.dump(models, ROOT / "models/types_silver.joblib")
(ROOT / "reports/types_cv.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
