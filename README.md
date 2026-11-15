# HelpRoute

8th Datathon for Social Good, Advanced Track. People in crisis often post on Reddit instead of calling 211. HelpRoute reads a post, identifies the need (housing, food, bills…) and urgency, and routes the person to open local help in the Bay Area.

**Full plan, labeling guide, models and judging checklist: [CLAUDE.md](CLAUDE.md).** (It's named that so Claude Code picks it up as context automatically; it's a normal Markdown file.)

## Setup

```bash
git clone https://github.com/LeonardChiu567/helproute.git
cd helproute
pip install -r requirements.txt
```

The data is **not in the repo** (the raw file is 5 GB). Get `reddit_posts.csv` from the datathon data drop and put it at `data/raw/reddit_posts.csv`, or point to it with an environment variable:

```bash
export REDDIT_CSV="/path/to/reddit_posts.csv"        # macOS/Linux
$env:REDDIT_CSV = "C:\path\to\reddit_posts.csv"      # Windows PowerShell
```

## Run order

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Profile + clean | Open `profile_reddit.ipynb` → run everything above "Zero-shot screen" (run Jupyter from the repo folder) | ~3 min | `data/processed/reddit_candidates.parquet`, `data/processed/reddit_clean.parquet` |
| 2. Zero-shot screen | Same notebook, "Zero-shot screen" section. Set `LIMIT = 200` for a quick look. First run downloads the model (~1.6 GB). | 30+ min on CPU; minutes on an NVIDIA GPU (cuda) or Apple Silicon (mps), picked automatically | `data/processed/zeroshot_flagged.parquet`, `data/processed/zeroshot_handcheck.csv` |
| 3. Hand-check | Fill `human_real_need` (1/0) for the 30 posts in `zeroshot_handcheck.csv`, then rerun the last cell | ~15 min | agreement between zero-shot and a human |

The zero-shot section can also be run as a script: `python src/zeroshot_screen.py [--limit 200]`.

The profiling and cleaning cells still have their outputs from the last run, so you can read those results without running anything.

## What the notebook does

**Profiling** (chunked read; skips the huge `comments` column): counts by subreddit and month, Bay Area city mentions, keyword hits per need, go/no-go checks.

**Cleaning**, with the rows left after each step:

| Step | Rows left |
|---|---|
| Candidates (keyword hit, Bay Area, or a target subreddit) | 69,266 |
| Drop removed/deleted/link-only bodies (`[removed]`, `[deleted]`, `[eliminado]`…) | 68,148 |
| Drop news, opinion and off-topic subreddits | 64,455 |
| Drop duplicates (same text, keep earliest) | 61,715 |
| Drop posts with fewer than 5 words | 61,469 |
| Keep only the 15 chosen subreddits | **51,054** |

The cleaning also converts curly quotes to straight ones, decodes HTML entities, replaces links with `[url]` and cuts each post to 400 words. Keyword flags are then recomputed on the clean text.

The 15 subreddits: **Bay Area** bayarea, sanfrancisco, oakland, berkeley, sanjose, eastbay; **housing/money** housing, povertyfinance, personalfinance, frugal, homeless; **work** antiwork, jobs, careerguidance, recruitinghell. All are retained for national training; Bay Area fields are retained for routing, the demo, and evaluation.

## Findings so far (Nov 14)

- **Full file:** 230,520 posts from 793 subreddits, January 2025 to November 2026 (most from March to September 2026). `locked_post` is true for only 2,180 posts, so the thread-lock backup idea is out. r/housing has 0 posts.
- **Need flags are keyword matches, not labels.** They're noisy: "how do I read my PG&E bill" counts as a utilities need, and news posts and home-buying questions also match. Keywords are only used to pick posts to label and as baseline B1.
- **National clean set:** 51,054 posts across 15 communities, of which 6,959 match a need keyword. The Bay Area slice contains 11,673 posts and 841 keyword-hit candidates.
- **Routing remains Bay Area-specific:** city from post text → city subreddit → user prompt → 211/findhelp.org fallback.

## Open decisions

1. **Training set:** train and evaluate on all 15 subreddits with no city requirement. Keep the Bay Area filter only for routing, the map, and per-city reporting.
2. **Zero-shot baseline:** run it only on validation and test IDs with `--ids`; do not score all 6,959 keyword-hit posts.

## Next steps

1. Decide 1 above, then draw the labeling sample: 50% keyword hits + 50% random, about 600 gold posts (300 train / 100 val / 200 test; set the test posts aside first).
2. Both of us label the same 100 posts → Cohen's kappa.
3. Silver labels with the Claude API, then baselines B0–B2, M1 TF-IDF + logistic regression, and M2 fine-tuned DistilRoBERTa (see CLAUDE.md).
