"""Customer Sentiment Analysis & Insights Dashboard (Streamlit).
UI only: all analysis logic lives in sentiment.py."""
import io, uuid
from pathlib import Path
import markdown
import pandas as pd
import plotly.express as px
import streamlit as st
import sentiment as sa

DATA = Path(__file__).parent / "data"   # always resolved relative to app.py, not the working directory
st.set_page_config(page_title="Sentiment Dashboard", page_icon="📊", layout="wide")
COL = {"Positive": "#35b6cb", "Neutral": "#8d99ae", "Negative": "#ee6c4d"}
ORDER = ["Positive", "Neutral", "Negative"]
PAGES = ["Dashboard", "Data Upload", "Sentiment Analysis", "Insights", "Data Explorer", "Reports", "Model Information", "Project Overview"]
st.markdown("""<style>[data-testid=stMetric]{background:rgba(207,239,242,.07);border:1px solid rgba(207,239,242,.16);border-radius:10px;padding:12px 14px}
h1,h2,h3{font-family:Cambria,Georgia,serif}</style>""", unsafe_allow_html=True)
S = st.session_state
S.setdefault("fv", 0)

def style(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10), height=330)
    return fig

def run(raw, text_col, source, demo=False):
    """Analyze and REPLACE the current dataset (old data is discarded)."""
    with st.spinner(f"Analyzing {len(raw):,} records…"):
        res = sa.analyze(raw, text_col, source)
    if res["df"].empty:
        st.error("No valid records found in this data. Check the text column."); return
    res.update(uid=uuid.uuid4().hex[:8], demo=demo); S.res = res; S.pop("report", None); S.fv += 1
    st.toast(f"Analysis complete: {len(res['df']):,} records analyzed.")

def load_sample():
    p = DATA / "sample-reviews-250.csv"
    raw = sa.read_upload(p.read_bytes(), p.name) if p.exists() else sa.make_sample()   # falls back to generated data if the file is missing
    run(raw, "review", "Sample Customer Reviews", demo=True)

if "res" not in S: load_sample()          # primary dataset: the app opens with the 250-row sample loaded
res = S.get("res"); df_all = res["df"]; meta = res["meta"]

# ---------- sidebar: navigation + global filters ----------
st.sidebar.title("📊 SentimentIQ")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider(); st.sidebar.markdown("### Filters")
k = f"{res['uid']}_{S.fv}"
mask = pd.Series(True, index=df_all.index)
mask &= df_all.Predicted.isin(st.sidebar.multiselect("Sentiment", ORDER, default=ORDER, key="s" + k))
if "_date" in df_all and df_all._date.notna().any():
    lo, hi = df_all._date.min().date(), df_all._date.max().date()
    dr = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key="d" + k)
    if len(dr) == 2: mask &= df_all._date.isna() | ((df_all._date.dt.date >= dr[0]) & (df_all._date.dt.date <= dr[1]))
for f in sa.FACETS:                          # filters appear only when the column exists
    c = meta.get(f)
    if c and 1 < df_all[c].nunique() < 60:
        pick = st.sidebar.multiselect(f.replace("_", " ").title(), sorted(x for x in df_all[c].unique() if x), key=f"{f}{k}")
        if pick: mask &= df_all[c].isin(pick)
if "_rating" in df_all and df_all._rating.notna().any():
    pick = st.sidebar.multiselect("Rating", sorted(df_all._rating.dropna().unique()), key="r" + k)
    if pick: mask &= df_all._rating.isin(pick)
mask &= df_all.Confidence >= st.sidebar.slider("Min confidence", 0, 95, 0, 5, format="%d%%", key="c" + k) / 100
if st.sidebar.button("Clear Filters"): S.fv += 1; st.rerun()
df = df_all[mask]
if df.empty: st.warning("No records match the current filters."); st.stop()

title = res["source"] + (" Dashboard" if "sentiment" in res["source"].lower() else " Sentiment Dashboard")

# ---------- pages ----------
def p_dashboard():
    st.title(title + ("  ·  SAMPLE DATA" if res["demo"] else ""))
    n, P, N = len(df), (df.Predicted == "Positive").sum(), (df.Predicted == "Negative").sum(); m = df.Score.mean()
    ds = df._date.dropna() if "_date" in df else pd.Series(dtype="datetime64[ns]")
    rng = f" · {ds.min():%b %Y} – {ds.max():%b %Y}" if len(ds) else ""
    st.caption(f"{n:,} records analyzed{rng} · Overall Tone {'Positive' if m >= .15 else 'Negative' if m <= -.15 else 'Mixed/neutral'}")
    delta = None
    if len(ds) >= 10:
        d = df.dropna(subset=["_date"]).sort_values("_date"); h = len(d) // 2; delta = f"{d.Score.iloc[h:].mean() - d.Score.iloc[:h].mean():+.2f} vs earlier half"
    K = [("Total Records", f"{n:,}", None, "Records that passed validation and match the filters"), ("Positive", int(P), None, "Score of +0.15 or higher"),
         ("Neutral", int(n - P - N), None, "Score between -0.15 and +0.15"), ("Negative", int(N), None, "Score of -0.15 or lower"),
         ("Overall Score", f"{m:.2f}", delta, "Mean sentiment score from -1 to +1"), ("Positive %", f"{P / n:.0%}", None, "Share of positive records"),
         ("Negative %", f"{N / n:.0%}", None, "Share of negative records"), ("Avg Confidence", f"{df.Confidence.mean():.0%}", None, "Heuristic certainty, not a measured accuracy")]
    for row in (K[:4], K[4:]):
        for c, (lab, v, dl, hp) in zip(st.columns(4), row): c.metric(lab, v, dl, help=hp)
    counts = df.Predicted.value_counts().reindex(ORDER).fillna(0).reset_index(); counts.columns = ["Sentiment", "Records"]
    st.plotly_chart(style(px.pie(counts, names="Sentiment", values="Records", hole=.55, color="Sentiment", color_discrete_map=COL, title="Sentiment distribution")), use_container_width=True)

def p_upload():
    st.header("Data Upload"); a, b = st.columns(2)
    with a:
        st.subheader("CSV upload"); f = st.file_uploader("Choose a CSV file", type=["csv", "tsv", "txt"])
        if f:
            try: raw = sa.read_upload(f.getvalue(), f.name)
            except Exception as e: st.error(f"Unable to process this file. {e}")
            else:
                cands = sa.text_candidates(raw); fid = getattr(f, "file_id", (f.name, f.size))
                col = cands[0] if len(cands) == 1 else st.selectbox("Text column to analyze", cands + [c for c in raw.columns if c not in cands])
                if (len(cands) == 1 and S.get("done") != fid) or (len(cands) != 1 and st.button(f"Analyze {len(raw):,} rows", type="primary")):
                    S.done = fid; run(raw, col, sa.clean_name(f.name)); st.rerun()
    with b:
        st.subheader("Type or paste text (one record per line)")
        txt = st.text_area("Comments", height=130, placeholder="Customer service was extremely helpful and resolved my issue quickly.")
        c1, c2 = st.columns(2)
        if c1.button("Analyze Sentiment", type="primary"):
            lines = [x for x in txt.splitlines() if x.strip()]
            if lines: run(pd.DataFrame({"text": lines}), "text", "Typed Text"); st.rerun()
            else: st.warning("Enter some text first.")
        if c2.button("Load Demo Dataset"): load_sample(); st.rerun()
        files = sorted(DATA.glob("*.csv")) if DATA.exists() else []
        for p in files: st.download_button("⬇ " + p.name, p.read_bytes(), p.name, "text/csv", key=p.name)
        if not files: st.download_button("⬇ sample-reviews-250.csv", sa.make_sample().to_csv(index=False), "sample-reviews-250.csv", "text/csv")
    st.info("Each new upload or entry replaces the current dataset, so results always reflect only the newest data.")
    q = res["quality"]; st.subheader("Data Quality Summary")
    for c, (lab, v) in zip(st.columns(6), [("Total rows", q["total"]), ("Valid rows", len(df_all)), ("Missing text", q["missing"]), ("Duplicates", q["duplicates"]), ("Too short", q["short"]), ("Excluded", q["missing"] + q["short"] + q["duplicates"])]): c.metric(lab, f"{v:,}")
    st.caption(f"Invalid dates: {q['bad_dates']} · Invalid ratings: {q['bad_ratings']} (records kept, value ignored)")
    with st.expander("Review excluded records"): st.dataframe(res["excluded"], hide_index=True, use_container_width=True) if len(res["excluded"]) else st.write("No records were excluded.")

def p_analysis():
    st.header("Sentiment Analysis"); a, b = st.columns(2)
    cnt = df.Predicted.value_counts().reindex(ORDER).fillna(0).reset_index(); cnt.columns = ["Sentiment", "Records"]
    a.plotly_chart(style(px.pie(cnt, names="Sentiment", values="Records", hole=.55, color="Sentiment", color_discrete_map=COL, title="Distribution")), use_container_width=True)
    with b:
        g = st.radio("Trend granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
        if "_date" in df and df._date.notna().any():
            fr = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[g]; d = df.dropna(subset=["_date"]).copy(); d["Period"] = d._date.dt.to_period(fr).dt.start_time
            t = d.groupby(["Period", "Predicted"]).size().reset_index(name="Records")
            st.plotly_chart(style(px.line(t, x="Period", y="Records", color="Predicted", color_discrete_map=COL, title="Sentiment trend")), use_container_width=True)
        else: st.info("No dates in the data, so no trend is available.")
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(df, x="Score", nbins=20, range_x=[-1, 1], title="Score distribution", color_discrete_sequence=["#98c1d9"])), use_container_width=True)
    seg = sa.segment_table(df, meta)
    if seg is not None: b.plotly_chart(style(px.bar(seg.sort_values("pos"), x="pos", y="Segment", orientation="h", range_x=[0, 100], title="% positive by segment", color_discrete_sequence=["#3d5a80"], labels={"pos": "% positive"})), use_container_width=True)
    if "_rating" in df and df._rating.notna().any():
        ct = pd.crosstab(df._rating, df.Predicted, normalize="index").mul(100).reset_index().melt("_rating", var_name="Predicted", value_name="%")
        st.plotly_chart(style(px.bar(ct, x="_rating", y="%", color="Predicted", color_discrete_map=COL, title="Rating vs predicted sentiment", labels={"_rating": "Rating"})), use_container_width=True)
    st.subheader("Top keywords"); kp, kn, kc = sa.keyword_stats(df)
    for c, (lab, items) in zip(st.columns(3), [("🟢 Positive", kp), ("🟠 Negative", kn), ("⚪ Neutral / common", kc)]):
        c.markdown(f"**{lab}**\n\n" + ("\n".join(f"- {w} · {n}" for w, n in items) or "_none_"))

def p_insights():
    st.header("AI Insights"); st.caption("Generated from the current (filtered) data, never hardcoded.")
    for h, lines in sa.insights(df, meta).items():
        st.subheader(h); [st.markdown("- " + x) for x in lines]

def p_explorer():
    st.header("Data Explorer"); q = st.text_input("Search text")
    v = df[df.Text.str.contains(q, case=False, regex=False)] if q else df
    show = v[["Record", "Text", "Predicted", "Score", "Confidence"]].copy(); show["Predicted"] = show.Predicted.map({"Positive": "🟢 Positive", "Neutral": "⚪ Neutral", "Negative": "🟠 Negative"})
    if "_date" in v: show["Date"] = v._date.dt.date
    for f in ("category", "product"):
        if meta.get(f): show[f.title()] = v[meta[f]]
    ev = st.dataframe(show, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", height=380,
                      column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"), "Score": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"{len(v):,} rows. Click a row to see its analysis.")
    if ev.selection.rows:
        r = v.iloc[ev.selection.rows[0]]; st.subheader(f"Record #{r.Record}: {r.Predicted}")
        st.write(f"“{r.Text}”"); st.write(f"**Score** {r.Score:.2f} · **Confidence** {r.Confidence:.0%} · **Tone** " + ("Mixed" if r.Mixed else {"Positive": "Satisfied" if r.Score < .6 else "Delighted", "Negative": "Dissatisfied" if r.Score > -.6 else "Frustrated"}.get(r.Predicted, "Neutral")))
        st.write("**Keywords:** " + ", ".join(r.Keywords[:10])); st.caption("Processed text: " + r.Processed)
    out = v.drop(columns=[c for c in ("_date", "_rating", "Keywords", "Mixed") if c in v]); b = io.BytesIO(); out.to_excel(b, index=False)
    c1, c2 = st.columns(2); c1.download_button("⬇ Export filtered CSV", out.to_csv(index=False), "sentiment-results.csv", "text/csv"); c2.download_button("⬇ Export filtered Excel", b.getvalue(), "sentiment-results.xlsx")

def p_reports():
    st.header("Reports")
    if st.button("Generate Insights Report", type="primary"): S.report = sa.report_markdown(df, meta, res, title)
    if S.get("report"):
        st.markdown(S.report); html = "<meta charset='utf-8'><body style='font-family:Arial;max-width:800px;margin:auto'>" + markdown.markdown(S.report)
        c1, c2 = st.columns(2); c1.download_button("⬇ Report (HTML, print to PDF)", html, "insights-report.html", "text/html"); c2.download_button("⬇ Report (Markdown)", S.report, "insights-report.md")
    else: st.info("Click Generate Insights Report to build the 11-section report from the current data.")

def p_model():
    st.header("Model Information")
    st.markdown(f"**Name:** Lexicon + rules (VADER-style)  \n**Type:** Rule-based lexicon with negation, intensifier and “!” handling  \n**Task:** 3-class sentiment classification  \n**Records analyzed:** {len(df_all):,}  \n**Text column:** {res['text_col']}")
    M = sa.metrics(df_all)
    if M:
        st.subheader(f"Performance against the labelled column ({M['n']} records)")
        for c, (l, v) in zip(st.columns(4), [("Accuracy", M["accuracy"]), ("Precision", M["precision"]), ("Recall", M["recall"]), ("F1", M["f1"])]): c.metric(l, f"{v:.1%}")
    else: st.info("Model performance metrics are unavailable because the uploaded dataset does not contain labelled ground-truth sentiment values.")
    st.subheader("How the Sentiment Analysis Works")
    st.markdown("1. You provide text.\n2. The text is cleaned (lowercase, links and punctuation removed).\n3. Words are matched to a dictionary of emotional weights.\n4. Negation and intensifiers adjust the weights.\n5. The total becomes a score from -1 to +1 and a label.\n6. Results are counted and grouped.\n7. Charts and insights are generated from those numbers.")

def p_project():
    st.header("Project Overview"); st.subheader("AI-Powered Sentiment Analysis & Data Insights Dashboard")
    st.write("An interactive sentiment analysis solution that uses natural language processing, machine learning, data visualization, and AI-assisted insights to analyze customer feedback and transform unstructured text into actionable business intelligence.")
    st.markdown("**Core Technologies:** Python · Pandas · NLP · Machine Learning · Sentiment Analysis · Plotly · Streamlit · REST-ready design\n\n**Core Competencies:** Data interpretation · Analytical thinking · Text classification · Machine learning · AI-assisted insights · Data visualization · Business intelligence")

{"Dashboard": p_dashboard, "Data Upload": p_upload, "Sentiment Analysis": p_analysis, "Insights": p_insights, "Data Explorer": p_explorer,
 "Reports": p_reports, "Model Information": p_model, "Project Overview": p_project}[page]()
