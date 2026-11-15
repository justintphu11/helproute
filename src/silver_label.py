"""Silver labels: Claude labels Reddit posts with the HelpRoute labeling guide (JSON output).

Silver labels are for TRAINING ONLY. The test set stays human-labeled, and any post in
labeling/test_final.csv (or a near-duplicate of one) is never sent for labeling.

Steps
  1. Pilot on the human-labeled posts and compare with the human labels (synchronous, ~minutes):
       python src/silver_label.py pilot --n 100
  2. If agreement is good, label a sample with the Message Batches API (50% cheaper, async):
       python src/silver_label.py submit --n 5000      # prints a batch id, saves it to disk
       python src/silver_label.py collect              # waits for the batch, writes the parquet

Inputs : data/processed/reddit_clean.parquet, data/processed/train_labels.parquet,
         labeling/test_final.csv (excluded if present)
Outputs: data/processed/silver_pilot_<model>.parquet, reports/silver_pilot_<model>.md,
         data/processed/silver_labels.parquet
Needs  : ANTHROPIC_API_KEY in the environment (never hard-code it).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
BATCH_ID_FILE = P / "silver_batch_id.txt"
SAMPLE_FILE = P / "silver_sample.parquet"
TYPES = ["housing", "homelessness", "food", "medical", "job_income", "bills_debt", "animal"]

GUIDE_FILE = ROOT / "labeling" / "guide.md"
# The team's guide is the single source of truth. Only urgency (not in the guide yet) and the
# output instructions are added here; human-only columns (why_not, changed_from_draft) are ignored.
GUIDE = (
    "You label Reddit posts for HelpRoute using the team's labeling guide below. Apply it exactly, "
    "including the tie-breaker rules, even where you might personally judge differently.\n\n"
    + GUIDE_FILE.read_text(encoding="utf-8")
    + """

## Extra fields for this task
- Need types: set all to false when is_need is false.
- urgency (when is_need is true; "none" when is_need is false):
  immediate = within about 7 days, or a basic need is unmet right now (no food today, sleeping outside tonight, shutoff notice due);
  near_term = within weeks; ongoing = chronic hardship or planning without a near deadline.
- unsure: true if the post could reasonably go either way under the guide.
- reason: one short sentence citing the deciding evidence. Never include usernames.
- Ignore why_not and changed_from_draft; they are for human labelers only."""
)

SCHEMA = {
    "type": "object",
    "properties": {
        "is_need": {"type": "boolean"},
        **{t: {"type": "boolean"} for t in TYPES},
        "urgency": {"type": "string", "enum": ["immediate", "near_term", "ongoing", "none"]},
        "crisis": {"type": "boolean"},
        "unsure": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_need", *TYPES, "urgency", "crisis", "unsure", "reason"],
    "additionalProperties": False,
}


def params(text: str, model: str, effort: str) -> dict:
    return dict(
        model=model,
        max_tokens=4096,
        system=[{"type": "text", "text": GUIDE, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"<post>\n{text}\n</post>"}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
    )


def parse(message) -> dict:
    if message.stop_reason == "refusal":
        return {"error": "refusal"}
    if message.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return {"error": "no_text"}
    return json.loads(text)


def test_exclusion(df: pd.DataFrame) -> pd.Series:
    """True for posts that must never be labeled: test IDs and their near-duplicates."""
    test_path = ROOT / "labeling" / "test_final.csv"
    if not test_path.exists():
        print(f"WARNING: {test_path} not found. Test posts cannot be excluded yet; re-filter "
              "silver labels against the test set before training.")
        return pd.Series(False, index=df.index)
    from build_dataset import dup_key
    ids = set(pd.read_csv(test_path, usecols=["post_id"], dtype={"post_id": str})["post_id"])
    keys = dup_key(df["text"])
    test_keys = set(keys[df["post_id"].isin(ids)])
    return df["post_id"].isin(ids) | keys.isin(test_keys)


def pilot(args, client):
    gold = pd.read_parquet(P / "train_labels.parquet")
    # Prefer person-labeled posts: batch_a may have started from draft verdicts.
    person = gold[gold["source"] != "batch_a"]
    rest = gold[gold["source"] == "batch_a"]
    n_person = min(len(person), args.n)
    sample = pd.concat([person.sample(n_person, random_state=0),
                        rest.sample(max(0, args.n - n_person), random_state=0)])
    print(f"pilot: {len(sample)} gold posts ({n_person} person-labeled) on {args.model}, effort={args.effort}")

    def one(text):
        try:
            return parse(client.messages.create(**params(text, args.model, args.effort)))
        except anthropic.APIStatusError as e:
            return {"error": f"api_{e.status_code}"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(one, sample["text"].fillna("").tolist()))
    silver = pd.DataFrame(out, index=sample.index).add_prefix("silver_")
    res = pd.concat([sample, silver], axis=1)
    res.drop(columns=["dup_key"], errors="ignore").to_parquet(P / f"silver_pilot_{args.model}.parquet", index=False)

    ok = res[res.get("silver_error", pd.Series(np.nan, index=res.index)).isna()]
    lines = [f"# Silver-label pilot ({args.model}, effort {args.effort})", "",
             f"{len(ok)} of {len(res)} posts labeled ({len(res) - len(ok)} errors).", "",
             "| Label | Human + | Claude + | Agreement | Cohen's kappa |", "|---|---:|---:|---:|---:|"]
    for col in ["is_need", *TYPES, "crisis"]:
        h = ok[col].astype(int).to_numpy()
        s = ok[f"silver_{col}"].astype(bool).astype(int).to_numpy()
        kappa = cohen_kappa_score(h, s) if len(set(h) | set(s)) > 1 else float("nan")
        lines.append(f"| {col} | {h.sum()} | {s.sum()} | {(h == s).mean():.3f} | {kappa:.3f} |")
    for name, part in [("person-labeled", ok[ok.source != "batch_a"]), ("batch_a", ok[ok.source == "batch_a"])]:
        if len(part) and part["is_need"].nunique() + part["silver_is_need"].nunique() > 2:
            k = cohen_kappa_score(part["is_need"].astype(int), part["silver_is_need"].astype(int))
            lines.append(f"\nis_need kappa, {name} posts (n={len(part)}): {k:.3f}")
    dis = ok[ok["is_need"].astype(int) != ok["silver_is_need"].astype(int)]
    lines += ["", f"## is_need disagreements ({len(dis)})", ""]
    lines += [f"- human={int(r.is_need)} claude={int(r.silver_is_need)} — {r.silver_reason}" for r in dis.itertuples()]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"silver_pilot_{args.model}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16 + len(TYPES)]))
    print(f"full report: reports/silver_pilot_{args.model}.md")


def submit(args, client):
    clean = pd.read_parquet(P / "reddit_clean.parquet")
    clean["post_id"] = clean["post_id"].astype(str)
    gold_ids = set(pd.read_parquet(P / "train_labels.parquet")["post_id"].astype(str))
    pool = clean[~clean["post_id"].isin(gold_ids) & ~test_exclusion(clean)]
    hits, other = pool[pool["any_need"]], pool[~pool["any_need"]]
    n_hits = min(len(hits), args.n // 2)
    sample = pd.concat([hits.sample(n_hits, random_state=42),
                        other.sample(args.n - n_hits, random_state=42)])
    sample[["post_id", "text", "sub", "sub_group", "bay", "n_words", "any_need"]].to_parquet(SAMPLE_FILE, index=False)
    print(f"sample: {len(sample)} posts ({n_hits} keyword hits, {len(sample) - n_hits} random non-keyword); "
          f"model={args.model}, effort={args.effort}")
    requests = [Request(custom_id=f"p{i}", params=MessageCreateParamsNonStreaming(**params(t, args.model, args.effort)))
                for i, t in enumerate(sample["text"].fillna("").tolist())]
    batch = client.messages.batches.create(requests=requests)
    BATCH_ID_FILE.write_text(batch.id)
    print(f"batch {batch.id} submitted ({batch.processing_status}). Run `python src/silver_label.py collect` later.")


def collect(args, client):
    batch_id = args.batch or BATCH_ID_FILE.read_text().strip()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"{batch.processing_status}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    sample = pd.read_parquet(SAMPLE_FILE)
    rows = {}
    for r in client.messages.batches.results(batch_id):
        i = int(r.custom_id[1:])
        rows[i] = parse(r.result.message) if r.result.type == "succeeded" else {"error": r.result.type}
    labels = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out = pd.concat([sample.reset_index(drop=True), labels.reindex(range(len(sample)))], axis=1)
    out.to_parquet(P / "silver_labels.parquet", index=False)
    good = out["error"].isna() if "error" in out else pd.Series(True, index=out.index)
    print(f"wrote data/processed/silver_labels.parquet: {int(good.sum())} labeled, {int((~good).sum())} errors")
    print(f"is_need rate: keyword hits {out.loc[good & out.any_need, 'is_need'].mean():.2f}, "
          f"non-keyword {out.loc[good & ~out.any_need, 'is_need'].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "submit", "collect"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--batch", default=None, help="batch id for collect (default: last submitted)")
    args = ap.parse_args()
    args.n = args.n or (100 if args.mode == "pilot" else 5000)
    client = anthropic.Anthropic()
    {"pilot": pilot, "submit": submit, "collect": collect}[args.mode](args, client)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
