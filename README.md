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
| 1. Profile + clean | Open `profile_reddit.ipynb` → Run All (run Jupyter from the repo folder) | ~3 min | `data/processed/reddit_candidates.parquet`, `data/processed/reddit_clean.parquet` |
| 2. Zero-shot screen | `python src/zeroshot_screen.py` (add `--limit 200` for a quick look) | 20–40 min on CPU, 1–2 min on a Colab T4 GPU | `data/processed/zeroshot_flagged.parquet` |

The notebook in the repo still has its outputs from the last run, so you can read the results without running it.

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
| Keep only the 15 chosen subreddits | 52,087 |
| Non-Bay subreddits must mention a Bay Area city | **11,775** |

The cleaning also converts curly quotes to straight ones, decodes HTML entities, replaces links with `[url]` and cuts each post to 400 words. Keyword flags are then recomputed on the clean text.

The 15 subreddits: **Bay Area** bayarea, sanfrancisco, oakland, berkeley, sanjose, eastbay (all posts kept); **housing/money** housing, povertyfinance, personalfinance, frugal, homeless; **work** antiwork, jobs, careerguidance, recruitinghell. For the housing/money and work groups, only posts that mention a Bay Area city are kept.

## Findings so far (Nov 14)

- **Full file:** 230,520 posts from 793 subreddits, January 2025 to November 2026 (most from March to September 2026). `locked_post` is true for only 2,180 posts, so the thread-lock backup idea is out. r/housing has 0 posts.
- **Need flags are keyword matches, not labels.** They're noisy: "how do I read my PG&E bill" counts as a utilities need, and news posts and home-buying questions also match. Keywords are only used to pick posts to label and as baseline B1.
- **Clean Bay Area set:** 11,775 posts, of which 852 match a need keyword. By need: homeless 230, utilities 188, housing 155, medical 140, job loss 102, food 47, debt trap 31. Only 47 also contain urgency words.
- **Tracing a post to a city:** 343 of the 852 name a city in the text and 264 more come from a city subreddit, so 607 (71%) can be traced. 245 (mostly r/bayarea) can't, and the app asks the user for their city. San Francisco, San Jose, Oakland and Berkeley are the only cities with enough posts for per-city numbers.
- **Housing/money and work groups are tiny once a Bay Area city is required:** only 62 posts, versus about 7,700 need-flagged posts across all 15 subreddits.

## Open decisions

1. **Training set:** the Bay Area-only set has 852 need-flagged posts, which is thin for silver labels, a 200-post test set and per-subgroup fairness numbers. The proposal is to **train and evaluate on all 15 subreddits with no city requirement**, and use the Bay Area city filter only for routing, the map and the per-city slide numbers.
2. **Is zero-shot good enough as a screen?** See whether `src/zeroshot_screen.py` separates real needs from questions and news.

## Next steps

1. Decide 1 above, then draw the labeling sample: 50% keyword hits + 50% random, about 600 gold posts (300 train / 100 val / 200 test; set the test posts aside first).
2. Both of us label the same 100 posts → Cohen's kappa.
3. Silver labels with the Claude API, then baselines B0–B2, M1 TF-IDF + logistic regression, and M2 fine-tuned DistilRoBERTa (see CLAUDE.md).
