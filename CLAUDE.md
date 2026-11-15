# HelpRoute — 8th Datathon for Social Good (Advanced Track)

Context handoff from a planning chat (Nov 14, 2026). Team is doing the **Advanced Track** with **Reddit NLP** as the core.

## Advanced Track checklist (judges check this FIRST — missing any item drops us to Open Track)
1. **Model**: train or fine-tune a predictive/NLP model (keyword list or off-the-shelf API used as-is does NOT count).
2. **Baseline**: compare to a simple baseline (majority class, linear regression, published rule).
3. **Evaluation**: task-appropriate metrics (macro-F1, precision/recall, AUC, RMSE) on a held-out test set never used for training or tuning.
4. **Error + fairness analysis**: performance across ≥3 meaningful subgroups + concrete failure examples.
5. **Data decisions**: if we create labels / clean heavily / join datasets, explain how and hand-check a sample.
6. **Reproducibility**: code runs from raw data to reported results.
Then scored on: technical depth, social good impact, presentation.
Submission: project link (GitHub/Colab/site) + slide deck + ≤5-min video, via Google Form.

## Data findings so far
- **Reddit**: only the 1,000-row sample was checked. The sample is a random mix of subreddits (classicwow, IKEA…), none of the 20 the README lists, with dates back to 2012 → must verify the full file. Filter: posts with >10 comments, so "ignored posts" can't be studied. Columns: post_id, title, description, community_name, date_posted, num_upvotes, num_comments, comments (nested JSON w/ upvotes, ~75% filled), community_rules (~17%), locked_post (all False/blank in sample), tag/flair (~4%). No toxicity labels.
- **Google Maps Places** (147k rows): ~12k permanently_closed (8%); 496 closed among food banks/shelters/social services. Categories include Food bank (433), Social services org (~1,000), Homeless shelter (79), Community health center (149), Free clinic (16), Employment agency, Bank, Money transfer service, Pharmacy, Grocery store, Bus stop (20k). lat/lon 100% filled.
- **Places ↔ Reviews join** on place_id: only ~3k reviews / 139 places (mostly urgent care) → too thin for neighborhood comparisons.
- **Health Risk**: labels are deterministic NEWS2 (score 0→Normal, 1–4→Low, 5–6→Medium, 7+→High; 26 exceptions = single-parameter-3 rule). Avoid — any model gets ~100%.
- **Glassdoor**: 175k reviews, sub-ratings ~97% filled, review_pros/review_cons text, job titles span frontline to tech.
- Indeed, Reddit: only 1,000-row samples received; Zillow part 2 missing.

## Chosen project: HelpRoute
**Pitch**: People in crisis often don't call 211 — they post on Reddit. HelpRoute reads a message, identifies needs + urgency, and routes the person to open local help; tested to work equally well for informal and non-native English writers.

**How the person actually gets help (deployment story)**
1. Main: mod-approved resource-reply assistant for opted-in subreddits (r/povertyfinance, r/homeless, r/bayarea…). A separate bot runs the model (Reddit API / mod-installable app), drafts a reply, sends it to mods via modmail, a mod approves, then it's posted as a comment or private message. Note: AutoModerator CAN comment / modmail / private-message the author, but only runs regex/keyword rules — it can't run an ML model. Use an AutoModerator-style keyword rule as a baseline.
2. Same model as a public "describe your situation" page.
3. Aggregate maps for funders / county agencies.
Outside the Bay Area or no city → fall back to 211 / findhelp.org. We never post to Reddit; the bot is a mockup.

**App (Streamlit, deploy on HF Spaces / Streamlit Cloud)**
- Tab 1 Triage: paste text → needs w/ confidence, urgency, highlighted evidence phrases, nearest open resources (map), suggested reply + approve button.
- Tab 2 Needs vs resources map by city.
- Tab 3 Financial Deserts: check-cashing/pawn/money-transfer vs banks/credit unions per city, overlaid with debt-trap post counts.
- Tab 4 Model card: metrics, fairness, failures, limitations.

## Labeling guide
Needs (multi-label): Housing (eviction, can't pay rent, landlord dispute — not apartment shopping), Homelessness (no stable place now), Food (can't afford enough), Bills/utilities (shutoff, behind), Medical costs (can't afford care / medical debt), Job/income loss, Debt trap (payday, stacked BNPL, overdraft spiral), None.
Urgency (single): Immediate (≤~7 days or basic need unmet now), Near-term (weeks), Ongoing.
Edge rules: past tense resolved → None; helper posts ("my mom is getting evicted") → label; venting with a clear need → label.
Excluded on purpose: self-harm / DV detection (needs trained crisis staff).

## Pipeline
- Reddit: use title + description, drop [removed]/[deleted] + dupes, truncate ~400 words. Labeling sample: 50% keyword-screen hits + 50% random (document this). ~600 gold: 300 train / 100 val / 200 test (lock test first). 2 annotators on 100-post overlap → Cohen's kappa (if <0.6, merge confusing labels).
- Silver labels: Claude API labels 5–10k posts with the same guide (JSON output); report silver-vs-gold agreement.
- Places: permanently_closed == False; category→need map (Food: Food bank, Soup kitchen; Housing: social services/nonprofits with housing/tenant/legal in name; Homelessness: shelters; Medical: Community health center, Free clinic; Jobs: Employment agency; Debt: Credit union, Bank). Hand-check 30 mapped places. City from address; centroid = mean lat/lon. BallTree haversine → top 5 per need; flag temporarily_closed.
- City from post: Bay Area city-name gazetteer → fallback subreddit (r/oakland→Oakland) → user picks in app.

## Models
B0 majority/"None"; B1 keyword rule; B2 zero-shot BART-MNLI (off-the-shelf baseline); M1 TF-IDF + one-vs-rest logistic regression (fallback); **M2 fine-tuned DistilRoBERTa** on silver + gold-train (max_len 256, lr 2e-5, 3 epochs, bs 16, BCE multi-label, per-label thresholds tuned on val; Colab T4 ~20 min).

## Evaluation (200 gold test posts only)
Needs macro-F1 + per-label P/R/F1; urgency weighted-F1 + **recall on Immediate** (tune threshold for ≥90% recall, report false-alarm cost); bootstrap 95% CIs; PR curve for Immediate; optional later-months test for drift.

## Fairness + errors
Subgroups: subreddit group (Bay Area local / housing+money / work), post length (<50 words), informal style (hand-flagged), non-native English (hand-flagged). Report recall gaps w/ CIs; mitigation = add informal examples, show before/after. Failure table of 6–8 paraphrased misses grouped by cause (past tense, sarcasm, implicit need, wrong city). Never show usernames.

## Slide numbers to compute
Monthly Bay Area need posts by category; % of Immediate posts keyword rule misses vs HelpRoute; city with most need posts per open resource; worst vs best city check-cashing:bank ratio; closed safety-net listings near hotspots.

## Repo layout
```
helproute/
  README.md (checklist → evidence table)   run_all.sh
  data/raw/ data/processed/
  labeling/ guide.md gold_labels.csv agreement.ipynb
  src/ 01_clean_reddit.py 02_silver_label.py 03_resources.py 04_baselines.py
       05_finetune.py 06_evaluate.py 07_fairness.py
  app/streamlit_app.py
  reports/ model_card.md figures/
```

## Timeline / go-no-go
Day 1 noon: guide + overlap labels done (kappa check). Day 1 6pm: B0–M1 working; if M2 not training by 10pm, ship M1. Day 2 11am: final test numbers, freeze models. Day 2 3pm: app deployed → slides + video.

## Backup ideas (if full Reddit data doesn't support HelpRoute)
1. "Does AI moderation silence poverty?" — Detoxify baseline vs fine-tuned rule-violation model; fairness = false positives on hardship subreddits / AAVE / non-native English.
2. Know-Your-Rights checker — train on Glassdoor, test on Reddit (labor violations: wage theft, missed breaks, unpaid OT, retaliation).
3. Thread-lock predictor — only if locked_post has thousands of True values (zero hand labels).

## STATUS (Nov 14, 2026)
- Done: full Reddit file profiled and cleaned in `profile_reddit.ipynb`. Results, row counts per cleaning step and open decisions are in README.md.
- The full file is 230,520 posts (Jan 2025–Nov 2026). Keyword go/no-go: GO. locked_post is true for only 2,180 posts → backup #3 is out.
- `reddit_clean.parquet` is the national 15-subreddit training corpus: 51,054 posts and 6,959 need-keyword hits. Bay Area fields are retained only for routing and the Bay Area-vs-national evaluation subgroup.
- Keyword "needs" are not labels (e.g. PG&E questions match). `src/zeroshot_screen.py` (BART-MNLI, baseline B2) checks real need vs. question/news.

## NEXT STEP
Draw the ~600-post gold labeling sample from all 15 subreddits (50% keyword hits / 50% random; set the 200 test posts aside first), including a Bay Area slice for subgroup evaluation.
