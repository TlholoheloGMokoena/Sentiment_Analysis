"""Sentiment engine: preprocessing, model, data quality, analytics and insights.
No Streamlit code lives here, so every function can be tested or reused on its own."""
import csv, io, math, re
from collections import Counter
import pandas as pd

# ---------- 1. Preprocessing ----------
STOP = set("a an the and or but if of to in on at for with from by as is are was were be been am it its this that these those i me my we our you your he she they them their so very just than then too also can could would should will have has had do does did not no nor there here what which who about into over after before up out all any some more most really get got one still only even".split())

def tokenize(text):
    """Lowercase, strip URLs and punctuation (apostrophes kept so n't is detected), split into words."""
    t = re.sub(r"https?://\S+", " ", str(text).lower())
    return [w for w in re.sub(r"[^a-z0-9'\s]", " ", t).split() if w]

def keywords(tokens):
    """Unique meaningful words: no stop words, no numbers, 3+ letters."""
    return list(dict.fromkeys(w for w in tokens if w not in STOP and len(w) > 2 and not w.isdigit()))

# ---------- 2. Sentiment model (replaceable: any function with the same signature) ----------
LEX = {}
for p in ("great:2.5 excellent:3 amazing:3 love:2.8 loved:2.8 good:1.8 happy:2.2 helpful:2.2 fast:1.6 friendly:2 quick:1.5 quickly:1.5 easy:1.6 perfect:3 awesome:3 best:2.6 fantastic:3 wonderful:2.8 lovely:2.2 recommend:2 satisfied:2 pleased:2 smooth:1.5 reliable:2 comfortable:1.8 worth:1.5 impressed:2.4 nice:1.6 beautiful:2.2 brilliant:2.8 early:1.2 resolved:1.8 professional:1.8 durable:1.8 sturdy:1.8 superb:3 solid:1.5 delighted:3 enjoy:2 enjoyed:2 "
          "bad:-2 terrible:-3 awful:-3 horrible:-3 poor:-2.2 disappointed:-2.5 disappointing:-2.5 slow:-1.8 late:-1.8 delay:-1.8 delayed:-1.8 broken:-2.5 broke:-2.4 rude:-2.8 cheap:-1.5 waste:-2.5 refund:-1.2 worst:-3 useless:-2.8 crashing:-2.2 crashes:-2.2 confusing:-1.8 expensive:-1.2 damaged:-2.5 defective:-2.8 faulty:-2.5 unhelpful:-2.4 stopped:-1.2 hate:-2.8 angry:-2.5 frustrating:-2.4 frustrated:-2.4 problem:-1.5 problems:-1.5 issue:-0.8 missing:-1.5 wrong:-1.8 complaint:-1.8 ignored:-2.2 overpriced:-2 noisy:-1.5 failed:-2.2 unacceptable:-2.8 scam:-3 nobody:-1").split():
    w, v = p.split(":"); LEX[w] = float(v)
BOOST = {"very": 1.5, "extremely": 1.6, "really": 1.4, "so": 1.3, "incredibly": 1.6, "slightly": .6, "somewhat": .6}
NEGATORS = {"not", "no", "never", "hardly", "without"}

def predict(tokens, original=""):
    """Lexicon + rules (VADER-style). Returns score in [-1, 1], label and a heuristic confidence."""
    s, pos, neg = 0.0, 0, 0
    for i, w in enumerate(tokens):
        v = LEX.get(w)
        if v is None: continue
        prev = tokens[max(0, i - 3):i]
        for x in prev:
            if x in BOOST: v *= BOOST[x]                       # intensifiers / softeners
        if any(x in NEGATORS or x.endswith("n't") for x in prev): v *= -.75   # negation flips
        pos, neg = (pos + 1, neg) if v > 0 else (pos, neg + 1)
        s += v
    s *= 1 + .1 * min(3, str(original).count("!"))               # exclamation boost
    s = s / math.sqrt(s * s + 6)                                  # normalise to -1..1
    label = "Positive" if s >= .15 else "Negative" if s <= -.15 else "Neutral"
    conf = .5 + .4 * (.15 - abs(s)) / .15 if label == "Neutral" else min(.99, .55 + abs(s) * .6)
    return {"score": round(s, 4), "label": label, "confidence": round(conf, 4), "pos_hits": pos, "neg_hits": neg}

# ---------- 3. Loading ----------
def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    """Decode (UTF-8 or Windows-1252), detect the delimiter, parse. Raises ValueError with a friendly message."""
    if filename.lower().endswith((".xlsx", ".xls")):
        raise ValueError("Excel files are not supported. Save as CSV (UTF-8) and upload that.")
    if not filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise ValueError("Unsupported file type. Please upload a .csv file.")
    if len(data) > 25_000_000: raise ValueError("File is too large (max 25 MB).")
    try: text = data.decode("utf-8-sig")
    except UnicodeDecodeError: text = data.decode("cp1252", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    sep = max([",", ";", "\t", "|"], key=first.count)
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    if df.empty: raise ValueError("The file is empty or has only a header row.")
    df.columns = [c.strip() or f"column_{i+1}" for i, c in enumerate(df.columns)]
    return df

TEXT_HINTS = ["review", "review_text", "comment", "feedback", "message", "text", "description", "sentiment"]
norm = lambda c: re.sub(r"[\s-]+", "_", c.strip().lower())
def text_candidates(df):
    c = [x for x in df.columns if norm(x) in TEXT_HINTS]
    return sorted(c, key=lambda x: TEXT_HINTS.index(norm(x)))

FACETS = ["category", "product", "location", "department", "channel", "customer_type"]
TRUTH = {"positive": "Positive", "pos": "Positive", "negative": "Negative", "neg": "Negative", "neutral": "Neutral", "neu": "Neutral"}

def clean_name(filename):
    """File name -> dashboard title keywords (generic words removed)."""
    base = re.sub(r"\.[^.]+$", "", filename); base = re.sub(r"\(\d+\)", " ", base)
    w = [x for x in re.split(r"[^A-Za-z0-9]+", base) if x and not re.fullmatch(r"(?i)dataset|data|sample|file|export|copy|final|new|csv|v\d+|\d+", x)]
    return " ".join(x.capitalize() for x in w) or "Uploaded Data"

# ---------- 4. Analysis pipeline: validate -> preprocess -> predict ----------
def analyze(raw: pd.DataFrame, text_col: str, source: str):
    """Returns a result dict: data, quality summary, excluded rows, column map, ground-truth flag."""
    cols = {norm(c): c for c in raw.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    truth_col = next((cols[n] for n in ("sentiment", "label") if n in cols and cols[n] != text_col), None)
    if truth_col is not None:
        vals = raw[truth_col].str.strip().str.lower(); nonempty = vals[vals != ""]
        if len(nonempty) == 0 or nonempty.isin(TRUTH).mean() < .8: truth_col = None
    date_c, rate_c = pick("date", "created_at", "timestamp"), pick("rating", "stars")
    meta = {"date": date_c, "rating": rate_c, **{f: pick(f) for f in FACETS}}
    if not meta["location"]: meta["location"] = pick("city", "country")
    q = dict(total=len(raw), missing=0, short=0, duplicates=0, bad_dates=0, bad_ratings=0)
    rows, excluded, seen = [], [], set()
    for i, r in raw.iterrows():
        text = re.sub(r"\s+", " ", str(r[text_col])).strip(); tk = tokenize(text)
        reason = "Missing text" if not text else "Extremely short text" if len(text) < 4 or not tk else None
        if not reason:
            key = (text.lower(), str(r[date_c]) if date_c else "", str(r[pick("customer_id", "customer", "id")]) if pick("customer_id", "customer", "id") else "")
            if key in seen: reason = "Duplicate"
            seen.add(key)
        if reason:
            q["missing" if reason.startswith("Missing") else "short" if reason.startswith("Extremely") else "duplicates"] += 1
            excluded.append({"row": i + 2, "reason": reason, "text": text[:120]}); continue
        p = predict(tk, text); kw = keywords(tk)
        rows.append({**r.to_dict(), "Text": text, "Processed": " ".join(kw), "Keywords": kw, "Predicted": p["label"],
                     "Score": p["score"], "Confidence": p["confidence"], "Mixed": p["pos_hits"] > 0 and p["neg_hits"] > 0,
                     "Truth": TRUTH.get(str(r[truth_col]).strip().lower()) if truth_col else None})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "Record", range(1, len(df) + 1))
        if date_c:
            df["_date"] = pd.to_datetime(df[date_c], errors="coerce"); q["bad_dates"] = int(df["_date"].isna().sum() - (df[date_c].str.strip() == "").sum())
        if rate_c:
            rt = pd.to_numeric(df[rate_c], errors="coerce"); bad = rt.notna() & ~rt.between(1, 5)
            q["bad_ratings"] = int(bad.sum()); df["_rating"] = rt.where(~bad)
    return {"df": df, "quality": q, "excluded": pd.DataFrame(excluded), "meta": meta, "has_truth": truth_col is not None,
            "source": source, "text_col": text_col}

# ---------- 5. Analytics ----------
def metrics(df):
    """Accuracy / macro precision / recall / F1, only when ground-truth labels exist."""
    d = df[df["Truth"].notna()]
    if d.empty: return None
    P, R, F = [], [], []
    for l in ("Positive", "Neutral", "Negative"):
        tp = ((d.Truth == l) & (d.Predicted == l)).sum(); fp = ((d.Truth != l) & (d.Predicted == l)).sum(); fn = ((d.Truth == l) & (d.Predicted != l)).sum()
        if tp + fn == 0: continue
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn); P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0)
    return {"n": len(d), "accuracy": (d.Truth == d.Predicted).mean(), "precision": sum(P) / len(P), "recall": sum(R) / len(R), "f1": sum(F) / len(F)}

def keyword_stats(df, top=10):
    allc, pos, neg = Counter(), Counter(), Counter()
    for kw, lab in zip(df["Keywords"], df["Predicted"]):
        allc.update(kw); (pos if lab == "Positive" else neg if lab == "Negative" else Counter()).update(kw)
    P = [(w, n) for w, n in pos.most_common() if n > neg[w]][:top]; N = [(w, n) for w, n in neg.most_common() if n > pos[w]][:top]
    used = {w for w, _ in P + N}
    return P, N, [(w, n) for w, n in allc.most_common() if w not in used][:top]

def segment_table(df, meta):
    col = meta.get("product") or meta.get("category") or meta.get("department")
    if not col: return None
    g = df.groupby(col)["Predicted"].agg(n="size", pos=lambda s: (s == "Positive").mean() * 100, neg=lambda s: (s == "Negative").mean() * 100)
    return g[g.n >= 3].round(0).reset_index().rename(columns={col: "Segment"})

def insights(df, meta):
    """Plain-English insights generated from the filtered data (nothing hardcoded)."""
    n = len(df); P = (df.Predicted == "Positive").sum(); N = (df.Predicted == "Negative").sum(); U = n - P - N; m = df.Score.mean()
    pct = lambda a: round(a / n * 100)
    tone = "positive" if m >= .15 else "negative" if m <= -.15 else "mixed or neutral"
    kp, kn, _ = keyword_stats(df); out = {}
    out["Overall Insight"] = [f"Overall sentiment is {tone}: {pct(P)}% of {n:,} analyzed records are positive, {pct(N)}% negative and {pct(U)}% neutral (mean score {m:.2f}).",
                              f"Average model confidence is {round(df.Confidence.mean() * 100)}%. Treat neutral records with caution because they include mixed reviews."]
    seg = segment_table(df, meta); best = worst = None
    if seg is not None and len(seg) > 1:
        best, worst = seg.loc[seg.pos.idxmax()], seg.loc[seg.neg.idxmax()]
        out["Overall Insight"].append(f"{best.Segment} has the strongest positive sentiment ({best.pos:.0f}% positive), while {worst.Segment} has the highest share of negative feedback ({worst.neg:.0f}%).")
    out["Positive Drivers"] = [f"Positive reviews most often mention: {', '.join(f'{w} ({c})' for w, c in kp[:6])}."] if kp else ["Not enough positive text to identify drivers."]
    out["Negative Drivers"] = [f"Negative reviews most often mention: {', '.join(f'{w} ({c})' for w, c in kn[:6])}."] if kn else ["Not enough negative text to identify drivers."]
    out["Emerging Trends"] = ["Not enough dated history (about 4+ weeks) to describe a trend."]
    if "_date" in df and df._date.notna().sum() >= 10:
        wk = df.dropna(subset=["_date"]).groupby(df._date.dt.to_period("W"))["Predicted"].agg(list)
        if len(wk) >= 4:
            th = max(1, len(wk) // 3); f = sum(wk.iloc[:th], []); l = sum(wk.iloc[-th:], [])
            pf, pl = round(f.count("Positive") / len(f) * 100), round(l.count("Positive") / len(l) * 100); d = pl - pf
            out["Emerging Trends"] = [f"Positive share moved from {pf}% in the earliest weeks to {pl}% in the latest weeks ({d:+d} pts), so sentiment is {'broadly stable' if abs(d) < 4 else 'improving' if d > 0 else 'declining'}."]
    act = [f"Investigate “{w}”: it appears in {c} negative record(s)." for w, c in kn[:3]]
    if worst is not None and worst.neg >= 20: act.append(f"Prioritise {worst.Segment}: {worst.neg:.0f}% of its feedback is negative.")
    if kp: act.append(f"Protect and promote strengths: {', '.join(w for w, _ in kp[:3])}.")
    out["Recommended Actions"] = act or ["Collect more feedback to produce reliable recommendations."]
    return out

def report_markdown(df, meta, res, title):
    """11-section insights report as Markdown (also rendered to HTML for download)."""
    ins = insights(df, meta); kp, kn, kc = keyword_stats(df); M = metrics(res["df"]); q = res["quality"]; n = len(df)
    fmt = lambda a: ", ".join(f"{w} ({c})" for w, c in a) or "-"; bl = lambda a: "\n".join(f"- {x}" for x in a)
    seg = segment_table(df, meta)
    return f"""# {title}: Insights Report
## 1. Executive Summary
{bl(ins['Overall Insight'])}
## 2. Dataset Overview
{n:,} valid records of {q['total']:,} rows; {q['missing'] + q['short'] + q['duplicates']} excluded (missing {q['missing']}, too short {q['short']}, duplicates {q['duplicates']}). Source: {res['source']}.
## 3. Sentiment Distribution
{', '.join(f'{l}: {(df.Predicted == l).sum()} ({round((df.Predicted == l).mean() * 100)}%)' for l in ('Positive', 'Neutral', 'Negative'))}
## 4. Key Findings
{bl(ins['Overall Insight'][:3])}
## 5. Positive Sentiment Drivers
{bl(ins['Positive Drivers'])}
## 6. Negative Sentiment Drivers
{bl(ins['Negative Drivers'])}
## 7. Trends
{bl(ins['Emerging Trends'])}
## 8. Important Keywords
Positive: {fmt(kp)}  
Negative: {fmt(kn)}  
Common: {fmt(kc)}
## 9. Business Insights
{bl([f'{r.Segment}: {r.pos:.0f}% positive, {r.neg:.0f}% negative' for r in seg.itertuples()]) if seg is not None else 'No product or category column available.'}
## 10. Recommendations
{bl(ins['Recommended Actions'])}
## 11. Technical Methodology
Text is cleaned, tokenized and scored by a lexicon model with negation and intensifier rules (score -1 to +1; labels at +/-0.15). Confidence is a heuristic, not a measured accuracy. {(f"Against the labelled column ({M['n']} records): accuracy {M['accuracy']:.1%}, macro F1 {M['f1']:.1%}.") if M else 'Model performance metrics are unavailable because the uploaded dataset does not contain labelled ground-truth sentiment values.'}
"""

# ---------- 6. Built-in sample data (used if data/sample-reviews-250.csv is missing) ----------
def make_sample(n=250):
    """Generate the 250-row synthetic sample review dataset (seeded, repeatable)."""
    import random, datetime
    R = random.Random(7); pk = R.choice
    P = [("Aurora Headphones", "Electronics", .7), ("Nimbus Smart Lamp", "Home", .5), ("Trail Backpack", "Outdoor", .62), ("PureBrew Kettle", "Home", .3), ("Pulse Fitness Band", "Electronics", .45)]
    T = {"p": ["Customer service was extremely helpful and resolved my issue quickly.", "The {p} works great and the quality is excellent.", "Fast delivery and friendly staff, really happy with my {p}.", "Love the {p}, easy to use and worth the price!", "Packaging was lovely and the {p} arrived early.", "Honestly impressed. The {p} feels solid and reliable."],
         "n": ["The delivery was late and nobody answered my emails.", "My {p} stopped working after a week, very disappointed.", "Terrible support, rude staff and no refund.", "The {p} feels cheap and broke quickly, a waste of money.", "Slow shipping and the app keeps crashing.", "Not happy. The {p} arrived damaged and the process to return it was confusing."],
         "u": ["I ordered the {p} on Tuesday and it came in a standard box.", "The {p} does what it says, nothing more.", "Received the {p}. Still testing it.", "Average experience, delivery took about five days.", "It is a kettle, lamp or band like any other. Not sure yet."],
         "m": ["The {p} is great but delivery was slow.", "Helpful staff, however the price is too expensive.", "Good quality but the app is confusing and support was slow.", "Not bad, though I expected better battery life."]}
    CL = {"p": ["Would order again.", "Highly recommend.", "Arrived well packed."], "n": ["Will not recommend.", "Waited two weeks for a reply.", "Very frustrating."], "u": ["Will update later.", "Packaging was plain."], "m": ["Overall it is fine.", "Mixed feelings."]}
    end, rows = datetime.date(2026, 10, 22), []
    for i in range(n):
        p, c, b = pk(P); day = i * .72; x = R.random(); pb = min(.9, b + day / 180 * .15)
        k = "p" if x < pb else "m" if x < pb + .18 else "u" if x < pb + .3 else "n"
        t = pk(T[k]).replace("{p}", p) + (" " + pk(CL[k]) if R.random() < .4 else "")
        rt = {"p": 5 if R.random() < .8 else 4, "n": 1 if R.random() < .8 else 2, "u": 3, "m": 3}[k]
        rows.append({"id": 1000 + i, "date": (end - datetime.timedelta(days=int(180 - day))).isoformat(), "customer": f"C{100 + i}", "product": p,
                     "category": c, "rating": rt, "review": t, "location": pk(["Johannesburg", "Cape Town", "Durban", "Pretoria", "Gqeberha"])})
    return pd.DataFrame(rows).astype(str)
