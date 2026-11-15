"""Build the training dataset while preserving the locked test-set boundary."""

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


LABEL_COLUMNS = [
    "is_need", "unsure", "crisis", "housing", "homelessness", "food",
    "medical", "job_income", "bills_debt", "animal",
]
TYPE_COLUMNS = LABEL_COLUMNS[3:]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_binary(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(str).str.strip().isin({"1", "1.0", "true", "True", "yes", "Yes"}).astype(int)


def dup_key(text: pd.Series) -> pd.Series:
    return (text.fillna("").str.lower()
            .str.replace(r"\s*(read more|leer más)\s*$", "", regex=True)
            .str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str[:300])


def main() -> None:
    root = default_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=root / "data" / "processed")
    ap.add_argument("--labels", type=Path, default=root / "labeling" / "done")
    ap.add_argument("--test", type=Path, default=root / "labeling" / "test_final.csv")
    ap.add_argument("--out", type=Path, default=root / "data" / "processed" / "train_labels.parquet")
    args = ap.parse_args()

    label_paths = sorted(args.labels.glob("*.xlsx"))
    if not label_paths:
        raise FileNotFoundError(f"No .xlsx label files in {args.labels}")

    frames = []
    for path in label_paths:
        labels = pd.read_excel(path, dtype={"post_id": str})
        labels["post_id"] = labels["post_id"].astype(str)
        for col in LABEL_COLUMNS:
            labels[col] = normalize_binary(labels[col]) if col in labels else 0
        labels["is_need"] = labels[["is_need", *TYPE_COLUMNS]].max(axis=1)
        labels["source"] = labels.get("source", pd.Series(path.stem, index=labels.index)).fillna(path.stem)
        labels["_file"] = path.name
        frames.append(labels)

    labels = pd.concat(frames, ignore_index=True)
    if "labeler" not in labels:
        labels["labeler"] = labels["_file"]
    duplicate_rows = labels[labels.duplicated("post_id", keep=False)].copy()

    # Inter-annotator agreement: one pair (first two distinct labelers) per overlapping post,
    # Cohen's kappa per label across all such posts. Computed BEFORE merging.
    pairs = []
    for post_id, group in duplicate_rows.groupby("post_id"):
        distinct = group.drop_duplicates("labeler")
        if len(distinct) >= 2:
            pairs.append((distinct.iloc[0], distinct.iloc[1]))
    if pairs:
        print(f"inter-annotator overlap: {len(pairs)} posts")
        for col in ["is_need", *TYPE_COLUMNS, "crisis"]:
            a = [int(p[0][col]) for p in pairs]; b = [int(p[1][col]) for p in pairs]
            agree = sum(x == y for x, y in zip(a, b)) / len(pairs)
            kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")
            print(f"  {col:13s} agreement={agree:.3f} kappa={kappa:.3f} (positives: {sum(a)} vs {sum(b)})")
    else:
        print("inter-annotator overlap: none found (no post labeled by two different labelers)")

    # Merge duplicates: two labelers -> union (max) of labels; same labeler -> keep newest row.
    for post_id, group in duplicate_rows.groupby("post_id"):
        if group["labeler"].nunique() >= 2:
            merged = group.iloc[-1].copy()
            for col in LABEL_COLUMNS:
                merged[col] = int(group[col].max())
            merged["labeler"] = " + ".join(map(str, group["labeler"].unique())) + " (union)"
            labels = labels[labels["post_id"] != post_id]
            labels = pd.concat([labels, pd.DataFrame([merged])], ignore_index=True)
        else:
            print(f"warning: duplicate {post_id}; keeping newest label-file row")
            labels = labels.drop(group.index[:-1])

    clean = pd.read_parquet(args.data / "reddit_clean.parquet")
    clean["post_id"] = clean["post_id"].astype(str)
    clean["dup_key"] = dup_key(clean["text"])
    if "n_words" not in clean:
        clean["n_words"] = clean["text"].fillna("").str.split().str.len()

    # Access the locked CSV only for identifiers. Test texts/labels remain unread before Phase 8.
    test_ids = set(pd.read_csv(args.test, usecols=["post_id"], dtype={"post_id": str})["post_id"])
    test_keys = set(clean.loc[clean["post_id"].isin(test_ids), "dup_key"])
    labels = labels.drop(columns=["text"], errors="ignore")
    joined = labels.merge(clean[["post_id", "text", "sub_group", "bay", "n_words", "dup_key", "needs", "any_need"]], on="post_id", how="inner")
    before = len(joined)
    joined = joined[~joined["dup_key"].isin(test_keys)].copy()
    print(f"dropped for test-set ID/near-duplicate exclusion: {before - len(joined)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.out, index=False)
    print(f"rows: {len(joined)}; is_need=1: {int(joined['is_need'].sum())}")
    print("by sub_group:\n" + joined.groupby("sub_group").size().to_string())
    print("by source:\n" + joined.groupby("source").size().to_string())
    print("by need type:\n" + joined[TYPE_COLUMNS].sum().to_string())
    print(f"crisis: {int(joined['crisis'].sum())}")


if __name__ == "__main__":
    main()
