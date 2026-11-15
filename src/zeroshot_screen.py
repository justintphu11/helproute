"""Zero-shot screen: is a keyword-flagged post a real need, or just a question/news/discussion?

Uses facebook/bart-large-mnli as-is (this is baseline B2, not our model).
Input:  data/processed/reddit_clean.parquet   (made by profile_reddit.ipynb)
Output: data/processed/zeroshot_flagged.parquet  (adds zs_need_score, 0-1)

Usage:  python src/zeroshot_screen.py            # all keyword-flagged posts
        python src/zeroshot_screen.py --limit 200 # quick sample
CPU is slow (30+ min for ~850 posts). A GPU (cuda) or Apple Silicon (mps) takes minutes.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--data", default="data/processed")
args = ap.parse_args()
D = Path(args.data)

df = pd.read_parquet(D / "reddit_clean.parquet")
n = df[df["any_need"]].copy()
if args.limit:
    n = n.sample(min(args.limit, len(n)), random_state=0)
texts = [" ".join(t.split()[:200]) for t in n["text"]]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"{len(texts)} posts on {device}", flush=True)
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
LABELS = ["someone who cannot afford or is losing housing, food, utilities, medical care, or income",
          "a general question, news, opinion, or discussion"]

t0 = time.time()
scores = []
for i in range(0, len(texts), 16):
    res = clf(texts[i:i + 16], candidate_labels=LABELS,
              hypothesis_template="This post is from {}.", truncation=True)
    scores += [dict(zip(r["labels"], r["scores"]))[LABELS[0]] for r in res]
    print(f"{len(scores)}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)

n["zs_need_score"] = scores
n[["post_id", "sub", "needs", "cities", "urgent_kw", "zs_need_score", "text"]].to_parquet(
    D / "zeroshot_flagged.parquet", index=False)
print(f"done in {time.time() - t0:.0f}s: {(n['zs_need_score'] > 0.5).sum()} of {len(n)} scored > 0.5 as a real need")
