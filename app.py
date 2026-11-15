"""HelpRoute review dashboard. Run: .venv/bin/streamlit run app.py."""
from collections import Counter
from html import escape
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.predict import HelpRoute
from src.model_download import ensure_models
from src.resource_routing import housing_sections

ROOT = Path(__file__).resolve().parent
LABELS = {"housing": "Housing", "homelessness": "Homelessness", "food": "Food", "medical": "Medical", "job_income": "Jobs & income", "bills_debt": "Bills & debt", "animal": "Animal care"}
RESOURCES = {
    'housing': [
        ('USAGov housing assistance', 'https://www.usa.gov/emergency-housing-assistance', 'Explore housing assistance options and find the relevant programs in your state. This government guide explains where to start if you are at risk of losing housing or have already lost it. Follow the links for your situation and contact the listed organizations to ask about next steps and current assistance.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Find your local 211 service for referrals to rent assistance and community support. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
        ('USAGov rental assistance', 'https://www.usa.gov/rental-housing-programs', 'Explore rental assistance programs and how to find housing support in your state. Read about different rental support options and use the linked agencies to explore programs serving your area. Ask the program directly about income requirements, application steps and waiting lists before applying.'),
    ],
    'homelessness': [
        ('HUD Find Shelter', 'https://www.hud.gov/FindShelter', 'Search for shelter and supportive services near your location across the U.S. Use the location-based search to identify nearby services, then contact a provider about intake instructions and available space. A directory listing does not guarantee an open bed, so check availability before traveling.'),
        ('USAGov emergency housing', 'https://www.usa.gov/emergency-housing-assistance', 'Find guidance on getting help when facing or experiencing homelessness. This government guide explains where to start if you are at risk of losing housing or have already lost it. Follow the links for your situation and contact the listed organizations to ask about next steps and current assistance.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Find your local 211 service for shelter referrals and other basic-needs support. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
    ],
    'food': [
        ('Feeding America food bank finder', 'https://www.feedingamerica.org/find-your-local-foodbank', 'Enter your ZIP code to find a food bank in the nationwide network and ask about nearby distributions. The listed food bank can help you identify partner pantries or food distributions closer to you. Contact it to check distribution times, pickup instructions and whether any registration is needed; food banks may not distribute directly to individuals.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Connect with your local 211 service to ask about food pantries, meals and other basic needs. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
        ('USAGov SNAP benefits', 'https://www.usa.gov/food-stamps', 'Find your state SNAP application and learn how food assistance works. This guide directs you to your state program so you can review eligibility and application options. Follow the state instructions to learn which household and income documents are needed and how to check an application’s progress.'),
    ],
    'medical': [
        ('HRSA Find a Health Center', 'https://findahealthcenter.hrsa.gov/', 'Locate community health centers nationwide and contact a center about care and payment options. Search by location to find centers and their contact details. Call the center to describe the care you need, ask about appointment availability and discuss fees, insurance or income-based payment options.'),
        ('HealthCare.gov low-cost care guide', 'https://www.healthcare.gov/community-health-centers/', 'Learn about community health center services and how income can affect the cost of care. The guide explains services such as primary care and referrals, and describes how charges may depend on income. Use its health-center locator link and ask the clinic which services and payment arrangements apply to you.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Ask about community health services and assistance programs in your area. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
    ],
    'job_income': [
        ('CareerOneStop American Job Centers', 'https://www.careeronestop.org/LocalHelp/AmericanJobCenters/find-american-job-centers.aspx', 'Find employment centers across the U.S. for job-search assistance and career services. Search for a center near you and contact it to ask about job-search support, career guidance and available training referrals. Check appointment requirements and which services the particular center offers before visiting.'),
        ('USAGov unemployment benefits', 'https://www.usa.gov/unemployment-benefits', 'Find your state unemployment program and learn how to apply; eligibility varies by state. Use the guide to reach your state’s official unemployment office and review filing instructions. Benefits depend on state rules and your work circumstances; the state agency can explain required information and how to follow up on a claim.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Ask about employment support and assistance while you look for work. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
    ],
    'bills_debt': [
        ('USAGov utility bill assistance', 'https://www.usa.gov/help-with-utility-bills', 'Explore programs for energy, phone and internet costs, including links to state assistance. Follow the program links to learn where to apply and what costs may be covered. If you have a shutoff notice, also contact your utility provider to ask about payment arrangements while you explore assistance.'),
        ('United Way 211', 'https://211.org/about-us/your-local-211', 'Ask your local 211 service about bill-payment assistance and financial support referrals. Use the national directory to connect with the 211 service covering your community. Explain the kind of help you need and ask which programs are accepting requests, what documents to bring, and how to contact them.'),
        ('CFPB help paying rent and bills', 'https://www.consumerfinance.gov/housing/housing-insecurity/help-for-renters/get-help-paying-rent-and-bills/', 'Review guidance on finding assistance with rent and household bills. This consumer guide explains routes to assistance and links to organizations that may help you navigate financial pressure. Use it to identify appropriate contacts, then ask those providers about eligibility and current program availability.'),
    ],
    'animal': [
        ('Humane World for Animals pet assistance', 'https://www.humaneworld.org/en/resources/having-trouble-affording-your-pet', 'Browse local and national resources for pet food, veterinary expenses and other pet-care needs. The guide brings together options for people struggling with pet-related costs. Review the resources relevant to food or veterinary care, then contact each organization about service areas, application requirements and available assistance.'),
        ('Findhelp', 'https://www.findhelp.org/', 'Search by ZIP code for community support and check whether pet-care assistance is listed in your area. Enter your ZIP code and search for the kind of assistance you need to see programs serving that location. Read each listing carefully and contact the provider to confirm that pet assistance is offered and that you meet its requirements.'),
        ('RedRover Relief Urgent Care Grants', 'https://redrover.org/relief/urgent-care-grants/', 'Learn about limited financial assistance for eligible pets in life-threatening situations; funding is not guaranteed. Review the program’s current eligibility rules and application instructions before submitting a request. Grants are limited and are not a substitute for arranging urgent veterinary care; discuss treatment timing and payment options with the veterinarian.'),
    ],
}
REPLY_INTRO = "Hey! We've noticed that you might need some assitance based on your recent activity. Here are some links that might help:"

EXAMPLES = [
    ("Housing", "I got evicted and have nowhere to sleep tonight, landlord changed the locks"),
    ("Work", "Laid off last month, unemployment ran out and I can't pay PG&E, shutoff notice came"),
    ("Community", "What's the best budgeting app for a family of four?"),
    ("Community", "I want to end it all, I can't go on like this"),
    ("Food", "I have no money for groceries and my children have nothing to eat this week."),
    ("Health", "I cannot afford my prescription and don't have health insurance. I need a low cost clinic."),
    ("Housing", "I've been sleeping in my car for two weeks. I need help finding a shelter."),
    ("Work", "My hours were cut and I need work immediately to support my family."),
    ("Community", "Our neighborhood picnic was wonderful this weekend."),
    ("Bills", "My electric company sent a disconnection notice and I cannot pay the overdue bill."),
    ("Housing", "I'm behind on rent and need rental assistance before the eviction hearing."),
    ("Animal care", "I can't afford food or veterinary care for my dog after losing my job."),
]

st.set_page_config(page_title="HelpRoute · community care", page_icon="🟠", layout="wide")
st.markdown('''<style>
html,body,[class*="css"],.stApp,button,input,textarea,h1,h2,h3,p {font-family:Arial,Helvetica,sans-serif!important}
.block-container{max-width:1420px;padding-top:1.5rem} #MainMenu,footer{visibility:hidden}
.banner{background:white;border-bottom:1px solid #ddd;padding:18px 26px;display:flex;align-items:center;gap:14px;margin-bottom:22px;border-radius:14px}
.brand{color:#ff4500;font-size:30px;font-weight:800;letter-spacing:-1.5px}.subtitle{color:#66777e;font-size:13px}
.hero{background:#fff0e9;border:1px solid #ffd2bf;border-radius:16px;padding:26px 30px;margin:14px 0 24px}
.hero h1{font-size:32px;margin:0 0 8px;letter-spacing:-1px}.hero p{color:#52636b;margin:0}
.post{background:white;border:1px solid #dae0e6;border-radius:14px;padding:20px 24px;margin:14px 0 4px;border-left:4px solid #ff4500}
.meta{color:#687b83;font-size:12px;margin-bottom:10px}.post p{font-size:17px;line-height:1.6}.tag{background:#fff0e9;color:#a63206;padding:5px 10px;border-radius:20px;font-size:12px;margin-right:6px;display:inline-block}
[data-testid="stMetric"]{background:white;border:1px solid #dae0e6;border-radius:12px;padding:18px}
[data-testid="stMetricValue"]{color:#ff4500} [data-testid="stSidebar"]{background:#fff;border-right:1px solid #dae0e6}
</style>''', unsafe_allow_html=True)
LOGO = '''<svg width="42" height="42" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="31" fill="#FF4500"/><g stroke="white" stroke-width="3" fill="none"><path d="M32 26l4-13 11 3"/><circle cx="49" cy="16" r="4"/><circle cx="13" cy="32" r="5"/><circle cx="51" cy="32" r="5"/></g><ellipse cx="32" cy="37" rx="22" ry="16" fill="white"/><g fill="#FF4500"><circle cx="24" cy="35" r="3"/><circle cx="40" cy="35" r="3"/></g><path d="M23 43q9 7 18 0" stroke="#FF4500" stroke-width="2" fill="none"/></svg>'''
st.markdown(f'<div class="banner">{LOGO}<div><div class="brand">HelpRoute</div><div class="subtitle">Community signals. Human support.</div></div></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading the trained model (first launch may take a minute)…"):
        ensure_models(ROOT / "models")
        return HelpRoute()


def clean_text(value):
    text = re.sub(r"(?i)(?:/?u/|@)[A-Za-z0-9_-]+", "[redacted]", str(value))
    return text


def normalize(frame):
    frame = frame.copy()
    if "text" not in frame:
        if "title" in frame or "description" in frame:
            frame["text"] = frame.get("title", pd.Series("", index=frame.index)).fillna("") + " " + frame.get("description", pd.Series("", index=frame.index)).fillna("")
        else:
            raise ValueError("CSV needs a text column (or title and description columns).")
    frame = frame.loc[frame.text.notna() & frame.text.astype(str).str.strip().ne("")].copy()
    frame["text"] = frame.text.map(clean_text)
    frame["group"] = frame.get("sub_group", frame.get("group", pd.Series("Community", index=frame.index))).fillna("Community").astype(str)
    label = next((c for c in ["y", "is_need", "label"] if c in frame), None)
    if label:
        frame["truth"] = pd.to_numeric(frame[label], errors="coerce")
        if not frame.truth.dropna().isin([0, 1]).all():
            raise ValueError("Ground-truth labels must be 0 or 1.")
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def predict_rows(texts):
    hr = load_model()
    return [hr.predict(t) for t in texts]


def recommendations(result, text=""):
    if not result["is_need"] and not result["crisis"]:
        st.caption("No actionable need detected.")
        return
    st.markdown(REPLY_INTRO)
    if result["crisis"]:
        st.markdown("**CRISIS SUPPORT**")
        st.markdown("- [988 Suicide & Crisis Lifeline](tel:988?oai_link_source=model_response_hotline) — Contact crisis support in the United States if you are experiencing suicidal thoughts or emotional distress. A crisis counselor can listen and help you explore support options.\n- [988 online chat](https://988lifeline.org/chat/?oai_link_source=model_response_hotline) — Connect with a crisis counselor online if you prefer typing to speaking on the phone. Open the chat page and follow its prompts to begin connecting with support.\n- [988 Lifeline information](https://988lifeline.org?oai_link_source=model_response_hotline) — Learn about crisis support options and what to expect when reaching out. Visit the website to understand how the service works and review the available ways to connect.")
        st.caption("If there is immediate danger, call 911. Crisis support takes priority over other resource recommendations.")
        if any(category != "emergency_services" for category in result.get("crisis_categories", [])) or not housing_sections(text):
            return
    matched_housing = housing_sections(text)
    for heading, resources in matched_housing:
        st.markdown(f"**{heading}**")
        st.markdown("\n".join(f"- [{name}]({url}) — {description}" for name, url, description in resources))
    for issue in result["top_types"]:
        if matched_housing and issue in {"housing", "homelessness"}:
            continue
        st.markdown(f"**{LABELS[issue].upper()}**")
        st.markdown("\n".join(f"- [{name}]({url}) — {description}" for name, url, description in RESOURCES[issue]))
    st.caption("U.S. national directories and programs. Enter your location on the linked sites; eligibility and availability vary.")


def result_card(text, result, idx=1, group="Community"):
    tags = "".join(f'<span class="tag">{escape(LABELS[t])}</span>' for t in result["top_types"])
    status = "Crisis review" if result["crisis"] else ("Need detected" if result["is_need"] else "No need detected")
    st.markdown(f'<div class="post"><div class="meta">↑ &nbsp; CASE {idx:03d} &nbsp; · &nbsp; {escape(group)} &nbsp; · &nbsp; {status}</div><p>{escape(text)}</p>{tags}<div class="meta" style="margin-top:14px">Need score {result["need_prob"]:.1%} · frozen threshold {result["threshold"]:.4f}</div></div>', unsafe_allow_html=True)
    recommendations(result, text)
    with st.expander("Model detail & evidence", expanded=False):
        if result["types"]:
            st.bar_chart(pd.Series({LABELS[k]: v for k, v in result["types"].items()}), color="#FF4500")
        st.write("Evidence phrases:", ", ".join(result["evidence"]) or "None")
        st.caption("Evidence comes from the M1 baseline; it is not an explanation of the transformer. Type scores were assessed by cross-validation, not the locked test set.")

with st.sidebar:
    st.markdown("### r/HelpRoute")
    st.caption("Find a path to support")
    page = st.radio("Navigate", ["Home", "Model performance"], label_visibility="collapsed")
    st.divider()
    st.caption("Support for housing, food, health, work, bills and pet care across the U.S.")
    st.markdown("[National resource directory](https://211.org/about-us/your-local-211) · [Health centers](https://findahealthcenter.hrsa.gov/)")

if page == "Home":
    st.markdown('<div class="hero"><h1>A little context. A path to support.</h1><p>Paste a post or describe what’s happening. Find resources that could help, anywhere in the U.S.</p></div>', unsafe_allow_html=True)
    text = st.text_area("Paste your post or describe your situation", height=200, placeholder="For example: I lost my job and need help paying rent and buying food.", max_chars=12000, key="home_prompt")
    submitted = st.button("Find helpful resources →", type="primary")
    st.caption("Edit or paste a new post, then click outside the box or press Ctrl+Enter (⌘+Enter on Mac) to update results automatically. Please leave out identifying details; entries are processed by this app and are not saved to disk.")
    safe_text = clean_text(text).strip()
    if not safe_text:
        st.session_state.pop("home_result", None)
        st.session_state.pop("home_text", None)
        if submitted:
            st.warning("Paste a post or describe your situation first.")
    elif submitted or safe_text != st.session_state.get("home_text"):
        st.session_state.pop("home_result", None)
        with st.spinner("Finding resources for you…"):
            new_result = load_model().predict(safe_text)
            st.session_state["home_text"] = safe_text
            st.session_state["home_result"] = new_result
    if "home_result" in st.session_state:
        with st.container(border=True):
            recommendations(st.session_state["home_result"], st.session_state.get("home_text", ""))
        st.subheader("Model results for your post")
        prompt_result = st.session_state["home_result"]
        classification, score, route_col = st.columns(3)
        classification.metric("Need classification", "Need detected" if prompt_result["is_need"] else "No actionable need")
        score.metric("Need score", f"{prompt_result['need_prob']:.1%}")
        route_col.metric("Suggested route", {"crisis": "Crisis support", "resources": "Resources", "none": "No action"}[prompt_result["route"]])
        st.caption(f"This post’s need score is compared with the frozen decision threshold of {prompt_result['threshold']:.4f}. A model score is not a measured probability that the prediction is correct.")
        if prompt_result["types"]:
            st.markdown("**Issue scores for this post**")
            issue_scores = pd.DataFrame({"Issue": [LABELS[k] for k in prompt_result["types"]], "Model score": list(prompt_result["types"].values())})
            st.bar_chart(issue_scores, x="Issue", y="Model score", color="#FF4500")
            st.dataframe(issue_scores.assign(**{"Model score": issue_scores["Model score"].map(lambda value: f"{value:.1%}")}), hide_index=True, width="stretch")
        st.caption("Accuracy, precision and recall require known correct labels across examples; they cannot be measured for an unlabeled prompt. Dataset metrics are on the Model performance page.")

else:
    st.markdown('<div class="hero"><h1>How HelpRoute performs.</h1><p>Explore the frozen model’s test results, review predictions, and understand patterns in the test posts.</p></div>', unsafe_allow_html=True)
    st.subheader("How the final model performed")
    st.caption("Published locked test results · 200 posts · 93 needs · reports/test_results.md")
    cols = st.columns(5)
    for col, name, value in zip(cols, ["Accuracy", "Precision", "Recall", "F1 score", "PR-AUC"], ["73%", "66.1%", "86.0%", "0.748", "0.840"]):
        col.metric(name, value)
    st.caption("Accuracy 73% was verified on the supplied 200 locked posts using the frozen model. Precision, recall, F1 and PR-AUC are from the final report. Recall: Bay Area 65% vs national 91%; only 17 Bay Area needs.")
    with st.expander("Load test posts / another review dataset"):
        st.write("Upload a CSV with text, optional sub_group, and optional ground-truth is_need (0/1). Usernames are never shown. Local locked test files are loaded automatically when available.")
        uploaded = st.file_uploader("Review CSV", type="csv")
        st.caption("Local files: labeling/test_final.csv + data/processed/reddit_clean.parquet; optional labeling/test_key.csv for groups. No retraining or threshold tuning is performed.")
    source = "Demo examples · synthetic, not locked test posts"
    try:
        if uploaded:
            df = normalize(pd.read_csv(uploaded))
            source = "Uploaded review posts"
        elif (ROOT / "labeling/test_final.csv").exists():
            raw = pd.read_csv(ROOT / "labeling/test_final.csv", dtype={"post_id": str})
            if "text" not in raw:
                clean = pd.read_parquet(ROOT / "data/processed/reddit_clean.parquet", columns=["post_id", "text"])
                clean["post_id"] = clean.post_id.astype(str)
                raw = raw.merge(clean, on="post_id", how="left", validate="one_to_one")
                if raw.text.isna().any():
                    raise ValueError("Some locked test posts have no matching text.")
            clean_path = ROOT / "data/processed/reddit_clean.parquet"
            if "sub_group" not in raw and clean_path.exists():
                groups = pd.read_parquet(clean_path, columns=["post_id", "sub_group"])
                groups["post_id"] = groups.post_id.astype(str)
                raw = raw.merge(groups, on="post_id", how="left", validate="one_to_one")
            key = ROOT / "labeling/test_key.csv"
            if key.exists() and "sub_group" not in raw:
                raw = raw.merge(pd.read_csv(key, dtype={"post_id": str})[["post_id", "sub_group"]], on="post_id", how="left", validate="one_to_one")
            df = normalize(raw)
            source = "Local locked test posts · frozen inference"
        else:
            df = normalize(pd.DataFrame(EXAMPLES, columns=["group", "text"]))
    except Exception as exc:
        st.error(f"Could not load review data: {exc}")
        st.stop()
    if df.empty:
        st.info("The selected dataset has no nonempty posts.")
        st.stop()
    st.info(source)
    if len(df) > 500:
        st.warning("Showing the first 500 rows to keep local inference responsive.")
        df = df.head(500)
    with st.spinner(f"Scoring {len(df)} posts with the frozen model…"):
        results = predict_rows(tuple(df.text))
    df["result"] = results
    df["predicted_need"] = [r["is_need"] for r in results]
    df["route"] = [r["route"] for r in results]
    if "truth" in df and df.truth.notna().any():
        labeled = df[df.truth.notna()]
        st.markdown("#### Metrics on the loaded labeled posts")
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{accuracy_score(labeled.truth, labeled.predicted_need):.1%}")
        b.metric("Precision", f"{precision_score(labeled.truth, labeled.predicted_need, zero_division=0):.1%}")
        c.metric("Recall", f"{recall_score(labeled.truth, labeled.predicted_need, zero_division=0):.1%}")
        st.caption(f"Computed with the existing threshold on {len(labeled)} labeled posts. These do not replace the published final test results.")
    filter1, filter2 = st.columns(2)
    issue = filter1.selectbox("Filter by issue", ["All issues", *LABELS.values()])
    route = filter2.selectbox("Filter by route", ["All routes", "resources", "crisis", "none"])
    selected = df.copy()
    if issue != "All issues":
        code = next(k for k, v in LABELS.items() if v == issue)
        selected = selected[selected.result.map(lambda r: code in r["top_types"])]
    if route != "All routes":
        selected = selected[selected.route.eq(route)]
    feed, insights = st.tabs(["Community feed", "Community insights"])
    with feed:
        st.caption(f"{len(selected)} anonymous posts · issue labels are model predictions")
        if selected.empty:
            st.info("No posts match these filters.")
        page_count = max(1, (len(selected) + 9) // 10)
        page_number = st.number_input("Feed page · 10 posts per page", min_value=1, max_value=page_count, value=1, step=1)
        start = (page_number - 1) * 10
        for position, (_, row) in enumerate(selected.iloc[start:start + 10].iterrows(), start + 1):
            result_card(row.text, row.result, position, row.group)
    with insights:
        st.caption("Charts reflect the filtered review posts, not population prevalence. Posts can have multiple predicted issues.")
        records = [{"Issue": LABELS[t], "Group": row.group, "Score": row.result["types"].get(t, 0)} for _, row in selected.iterrows() for t in row.result["top_types"]]
        if not records:
            st.info("No predicted issues in this selection. Change the filters to view charts.")
        else:
            issues = pd.DataFrame(records)
            st.subheader("Where needs intersect")
            st.caption("Issue heatmap · counts of predicted issues by community group")
            heat = issues.groupby(["Group", "Issue"]).size().rename("Posts").reset_index()
            base = alt.Chart(heat).encode(x=alt.X("Issue:N", sort=list(LABELS.values()), axis=alt.Axis(labelAngle=-25)), y="Group:N")
            st.altair_chart((base.mark_rect(cornerRadius=4).encode(color=alt.Color("Posts:Q", scale=alt.Scale(scheme="oranges")), tooltip=["Group", "Issue", "Posts"]) + base.mark_text().encode(text="Posts:Q", color=alt.value("#542109"))).properties(height=300), width="stretch")
            st.subheader("The shape of community needs")
            bubble = issues.groupby("Issue").agg(Posts=("Score", "size"), Mean_score=("Score", "mean")).reset_index()
            st.altair_chart(alt.Chart(bubble).mark_circle(opacity=.8, color="#ff4500", stroke="white", strokeWidth=2).encode(x=alt.X("Posts:Q", title="Posts with predicted issue"), y=alt.Y("Mean_score:Q", title="Mean issue score", scale=alt.Scale(domain=[0, 1])), size=alt.Size("Posts:Q", scale=alt.Scale(range=[200, 2500]), legend=None), tooltip=["Issue", "Posts", alt.Tooltip("Mean_score:Q", format=".1%")]).properties(height=330), width="stretch")
            st.subheader("Words behind the posts")
            st.caption("Word map · frequent terms in selected posts with a detected need. Size indicates frequency.")
            words = Counter(w for text in selected.loc[selected.predicted_need, "text"] for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in ENGLISH_STOP_WORDS and w != "redacted")
            word_df = pd.DataFrame([{"Word": w, "Count": n, "x": i % 6, "y": i // 6} for i, (w, n) in enumerate(words.most_common(30))])
            if not word_df.empty:
                st.altair_chart(alt.Chart(word_df).mark_text(font="Arial", fontWeight="bold").encode(x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None), text="Word:N", size=alt.Size("Count:Q", scale=alt.Scale(range=[16, 42]), legend=None), color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges"), legend=None), tooltip=["Word", "Count"]).properties(height=280).configure_view(strokeWidth=0), width="stretch")
    st.divider()
    st.caption("HelpRoute · community care prototype · Reddit-inspired design, not affiliated with Reddit. Arial interface. Model outputs require human review; need scores are not calibrated certainty.")
