# -*- coding: utf-8 -*-
"""
AMERICAN AIRLINES | AI-DRIVEN BRAND REPUTATION INTELLIGENCE
Streamlit application (single file). Dataset: Airline_Tweets.csv in the same folder.
"""
from __future__ import annotations

import html
import importlib.util
import os
import re
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS, CountVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:  # optional libraries - the app degrades gracefully if absent
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except Exception:  # pragma: no cover
    HAS_VADER = False
try:
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:  # pragma: no cover
    HAS_WC = False

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Airline_Tweets.csv"

st.set_page_config(page_title="AA Brand Reputation Intelligence", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
NAVY, TEAL, SLATE = "#0B2545", "#13A8A8", "#5B6B7F"
SENT_ORDER = ["negative", "neutral", "positive"]
SENT_COLORS = {"positive": "#2E9E6B", "neutral": "#E0A830", "negative": "#D64545"}
MODE_ORIG = "Original observations only (default)"
MODE_DEMO = "Demonstration mode (original + generated)"
M_MAJ, M_VADER = "Majority-class baseline", "VADER (lexicon)"
M_LR, M_LSTM = "TF-IDF + Logistic Regression", "Embedding + LSTM"
REQUIRED_COLS = ["tweet_id", "airline_sentiment", "airline", "text"]
SCENARIO_VARS = {
    "scenario_reputation_index_0_100": "Reputation index (0-100)",
    "scenario_rolling_12m_reputation_index": "Rolling 12m reputation index",
    "scenario_negative_sentiment_pct": "Scenario negative sentiment (%)",
    "scenario_monthly_revenue_usd": "Monthly revenue (USD)",
    "scenario_monthly_booking_count": "Monthly bookings",
    "scenario_customer_retention_pct": "Customer retention (%)",
    "scenario_customer_churn_pct": "Customer churn (%)",
    "scenario_nps": "Net Promoter Score",
    "scenario_ontime_performance_pct": "On-time performance (%)",
    "scenario_customer_support_cost_usd": "Customer support cost (USD)",
}
DOMAIN_STOP = {
    "flight", "flights", "amp", "rt", "americanair", "usairways", "united", "southwestair",
    "southwest", "jetblue", "delta", "virginamerica", "virgin", "american", "airline",
    "airlines", "airways", "aa", "dont", "cant", "didnt", "wont", "isnt", "doesnt", "wasnt",
    "im", "ive", "youre", "thats", "just", "got", "get", "will", "can", "would", "http", "https",
}
KEEP_WORDS = {"no", "not", "never"}
STOP_WORDS = (set(ENGLISH_STOP_WORDS) - KEEP_WORDS - {"nor"}) | DOMAIN_STOP

# Management assumptions (editable in the app) - NOT measured from the data
DEFAULT_SEVERITY = {
    "cancelled flight": 5, "lost luggage": 5, "late flight": 4, "customer service issue": 4,
    "damaged luggage": 4, "flight booking problems": 3, "bad flight": 3,
    "flight attendant complaints": 3, "longlines": 2, "can't tell": 2, "not specified": 2,
}
ACTIONS = {
    "customer service issue": ("Social-care and call-centre service-recovery training; escalation scripts; first-contact-resolution targets",
                               "Median social response time; CSAT after contact"),
    "late flight": ("Proactive delay alerts, automated rebooking and turnaround-process review with operations",
                    "On-time departure %; alert lead time"),
    "cancelled flight": ("Disruption playbook: rapid rebooking, hotel/meal vouchers, clear cancellation messaging",
                         "Rebooking completed < 2h (%); cancellation-related negative posts"),
    "lost luggage": ("Real-time bag tracking, faster claim resolution and proactive status updates",
                     "Mishandled bags per 1,000; claim closure days"),
    "damaged luggage": ("Baggage-handling audit at hubs; simplified damage claims",
                        "Damage claims per 1,000 bags; claim cycle time"),
    "flight booking problems": ("Fix booking/app/website defects; strengthen change/refund self-service",
                                "Booking-related contacts; app rating"),
    "bad flight": ("Review cabin experience standards (seat, cleanliness, amenities, inflight service)",
                   "Inflight satisfaction score"),
    "flight attendant complaints": ("Cabin-crew service coaching and complaint feedback loop",
                                    "Crew-related complaints per 10,000 passengers"),
    "longlines": ("Airport queue management, staffing at peak times and digital check-in promotion",
                  "Average check-in/security wait time"),
    "can't tell": ("Improve complaint tagging so unclassified negatives can be routed to owners",
                   "% of negative posts with a valid category"),
    "not specified": ("Improve complaint tagging so unclassified negatives can be routed to owners",
                      "% of negative posts with a valid category"),
}
GENERIC_ACTION = ("Investigate root causes with the owning department and set a service-recovery standard",
                  "Category-level negative-post share")

# ----------------------------------------------------------------------------
# Styling helpers
# ----------------------------------------------------------------------------
CSS = f"""
<style>
.block-container {{padding-top: 1.2rem; max-width: 1500px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #12396b 60%, {TEAL} 130%); color: #fff;
        padding: 26px 30px; border-radius: 16px; margin-bottom: 14px;}}
.hero h1 {{font-size: 1.65rem; margin: 0 0 4px 0; letter-spacing: .4px; color:#fff;}}
.hero p.sub {{margin: 0 0 8px 0; color: #BFE9E9; font-size: 1.0rem;}}
.hero p.desc {{margin: 0; font-size: .85rem; color: #DCE6F2;}}
.kpi {{background: #F6F8FB; border: 1px solid #E3E8EF; border-left: 5px solid {TEAL};
       border-radius: 12px; padding: 12px 14px; min-height: 108px;}}
.kpi .l {{font-size: .72rem; text-transform: uppercase; color: {SLATE}; letter-spacing: .6px;}}
.kpi .v {{font-size: 1.6rem; font-weight: 700; color: {NAVY}; line-height: 1.25;}}
.kpi .n {{font-size: .72rem; color: {SLATE};}}
.card {{background: #F6F8FB; border: 1px solid #E3E8EF; border-radius: 14px; padding: 14px 16px; height: 100%;}}
.card h4 {{margin: 0 0 6px 0; color: {NAVY};}}
.card p {{margin: 0; font-size: .85rem; color: #33415C;}}
.dmaic {{border-top: 5px solid {TEAL}; background:#fff; border-radius: 12px; padding: 12px 14px;
         border: 1px solid #E3E8EF; border-top-width: 5px; height: 100%;}}
.dmaic .t {{font-weight: 800; color: {NAVY}; font-size: 1.05rem;}}
.dmaic .b {{font-size: .82rem; color: #33415C;}}
.pill {{display:inline-block; padding:2px 10px; border-radius: 12px; font-size:.75rem; font-weight:600;}}
.pill-ok {{background:#E3F4EC; color:#1E7A52;}} .pill-warn {{background:#FFF3D6; color:#8A6200;}}
.pill-bad {{background:#FBE4E4; color:#9C2B2B;}}
.caption-box {{font-size: .82rem; color: {SLATE}; margin: -4px 0 8px 0;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)  # static, developer-defined CSS only


def esc(x) -> str:
    return html.escape(str(x))


def kpi_card(label: str, value: str, note: str = "", color: str = TEAL):
    st.markdown(
        f'<div class="kpi" style="border-left-color:{color}"><div class="l">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div><div class="n">{esc(note)}</div></div>',
        unsafe_allow_html=True)


def info_card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{esc(title)}</h4><p>{esc(body)}</p></div>',
                unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<div class="caption-box">{esc(text)}</div>', unsafe_allow_html=True)


def show_fig(fig, key: str):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(d: pd.DataFrame, **kw):
    kw.setdefault("hide_index", True)
    try:
        st.dataframe(d, width="stretch", **kw)
    except Exception:
        st.dataframe(d, use_container_width=True, **kw)


def show_img(arr):
    try:
        st.image(arr, width="stretch")
    except Exception:
        st.image(arr, use_container_width=True)


def style_fig(fig, h=370, title=None):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=48 if title else 18, b=10),
                      font=dict(family="Segoe UI, Inter, sans-serif", size=12, color=NAVY),
                      legend=dict(orientation="h", y=-0.22))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8-sig")


# ----------------------------------------------------------------------------
# Data loading and preparation
# ----------------------------------------------------------------------------
def validate_header(path: Path):
    cols = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.str.strip().tolist()
    return cols, [c for c in REQUIRED_COLS if c not in cols]


def _clean_vader_text(s: pd.Series) -> pd.Series:
    return (s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
             .str.replace(r"@\w+", " ", regex=True)
             .str.replace("&amp;", "&", regex=False)
             .str.replace(r"\s+", " ", regex=True).str.strip())


def _tokens(s: str) -> str:
    return " ".join(w for w in s.split() if w not in STOP_WORDS and (len(w) > 2 or w in KEEP_WORDS))


@st.cache_data(show_spinner="Loading and preparing the dataset...")
def load_data(path: str, mtime: float):
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    raw.columns = raw.columns.str.strip()
    df = raw.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["airline"] = df["airline"].astype(str).str.strip()
    df["airline_sentiment"] = df["airline_sentiment"].astype(str).str.strip().str.lower()
    # dates
    d = pd.to_datetime(df["tweet_date"], errors="coerce") if "tweet_date" in df else pd.Series(pd.NaT, index=df.index)
    if "tweet_created" in df:
        alt = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True).dt.tz_localize(None)
        d = d.fillna(alt)
    df["date"] = pd.to_datetime(d).dt.normalize()
    df["tweet_month"] = df["date"].dt.strftime("%Y-%m")
    # provenance
    if "record_is_observed" in df:
        obs = df["record_is_observed"].astype(str).str.upper().str.strip().isin(["YES", "TRUE", "1", "Y"])
    elif "record_origin" in df:
        obs = df["record_origin"].astype(str).str.upper().str.contains("ORIGINAL|OBSERVED")
    else:
        obs = pd.Series(True, index=df.index)
    df["is_observed"] = obs
    df["record_type"] = np.where(obs, "Observed (original tweet)", "Generated (demonstration)")
    # complaint category (only for negative tweets)
    reason = df["negativereason"].astype("object") if "negativereason" in df else pd.Series(np.nan, index=df.index, dtype="object")
    neg = df["airline_sentiment"] == "negative"
    comp = reason.where(neg)
    comp = comp.where(~(neg & comp.isna()), "Not specified")
    df["complaint"] = comp
    # numerics
    df["retweet_count"] = pd.to_numeric(df.get("retweet_count", 0), errors="coerce").fillna(0)
    for c in list(SCENARIO_VARS):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # text cleaning
    df["text_vader"] = _clean_vader_text(df["text"])
    df["text_norm"] = df["text_vader"].str.lower()
    df["text_clean"] = (df["text_norm"].str.replace("'", "", regex=False)
                        .str.replace("[^a-z ]", " ", regex=True)
                        .str.replace(r"\s+", " ", regex=True).str.strip())
    df["text_tokens"] = df["text_clean"].map(_tokens)
    df["n_words_raw"] = df["text"].str.split().str.len()
    df["n_tokens"] = df["text_tokens"].str.split().str.len()
    info = {
        "raw_rows": len(raw), "raw_cols": raw.shape[1],
        "bad_dates": int(df["date"].isna().sum()),
        "bad_labels": int((~df["airline_sentiment"].isin(SENT_ORDER)).sum()),
        "empty_text": int((df["text_clean"].str.len() == 0).sum()),
    }
    return df, info


@st.cache_data(show_spinner=False)
def cleaned_export(key: str, _df: pd.DataFrame) -> bytes:
    cols = [c for c in _df.columns if c not in ("text_vader",)]
    return csv_bytes(_df[cols])


def apply_filters(df, airlines, dates, sents, comps, comp_all, mode, use_airline=True):
    m = pd.Series(True, index=df.index)
    if mode == MODE_ORIG:
        m &= df["is_observed"]
    if use_airline:
        m &= df["airline"].isin(airlines)
    d0, d1 = dates
    m &= df["date"].between(pd.Timestamp(d0), pd.Timestamp(d1))
    m &= df["airline_sentiment"].isin(sents)
    if set(comps) != set(comp_all):
        m &= (df["airline_sentiment"] != "negative") | df["complaint"].isin(comps)
    return df[m]


# ----------------------------------------------------------------------------
# Analytics functions
# ----------------------------------------------------------------------------
def sentiment_summary(d: pd.DataFrame) -> pd.DataFrame:
    c = d["airline_sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
    tot = int(c.sum())
    return pd.DataFrame({"Sentiment": SENT_ORDER, "Tweets": c.values,
                         "Share (%)": (c.values / tot * 100) if tot else np.zeros(3)})


def daily_sentiment(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    g = (d.groupby(["date", "airline_sentiment"]).size().unstack(fill_value=0)
          .reindex(columns=SENT_ORDER, fill_value=0))
    g["total"] = g.sum(axis=1)
    for s in SENT_ORDER:
        g[f"{s}_pct"] = g[s] / g["total"] * 100
    g["net_sentiment"] = g["positive_pct"] - g["negative_pct"]
    g["rms"] = 50 + g["net_sentiment"] / 2  # Reputation Monitoring Score 0-100
    return g.reset_index()


def time_bin(d: pd.DataFrame):
    """Daily bins for short windows, weekly for long windows."""
    n_days = d["date"].nunique()
    if n_days <= 60:
        return d["date"], "day"
    return d["date"].dt.to_period("W").dt.start_time, "week"


@st.cache_data(show_spinner=False)
def ngram_freq(tokens: pd.Series, n: int) -> pd.DataFrame:
    c = Counter()
    for s in tokens:
        w = s.split()
        if n == 1:
            c.update(w)
        else:
            c.update(" ".join(p) for p in zip(*[w[i:] for i in range(n)]))
    out = pd.DataFrame(c.most_common(1500), columns=["term", "count"])
    return out


@st.cache_data(show_spinner=False)
def cooccurrence(tokens: pd.Series, vocab_size: int):
    docs = [sorted(set(s.split())) for s in tokens]
    df_cnt = Counter(w for d in docs for w in d)
    vocab = {w for w, _ in df_cnt.most_common(vocab_size)}
    pair_cnt = Counter()
    for d in docs:
        v = [w for w in d if w in vocab]
        pair_cnt.update(combinations(v, 2))
    pairs = pd.DataFrame([(a, b, c) for (a, b), c in pair_cnt.items()], columns=["word_1", "word_2", "weight"])
    pairs = pairs.sort_values("weight", ascending=False).reset_index(drop=True)
    wf = pd.Series({w: df_cnt[w] for w in vocab})
    return pairs, wf


def build_network(pairs, wfreq, min_edge, max_nodes):
    p = pairs[pairs["weight"] >= min_edge]
    if p.empty:
        return None
    G = nx.Graph()
    for r in p.itertuples(index=False):
        G.add_edge(r.word_1, r.word_2, weight=int(r.weight))
    deg = dict(G.degree(weight="weight"))
    keep = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
    G = G.subgraph(keep).copy()
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 2:
        return None
    pos = nx.spring_layout(G, k=1.4 / np.sqrt(G.number_of_nodes()), seed=7, weight="weight", iterations=120)
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    cmap = {w: i for i, c in enumerate(comms) for w in c}
    edges = sorted(G.edges(data="weight"), key=lambda e: -e[2])[:160]
    pal = px.colors.qualitative.Safe
    fig = go.Figure()
    ws = np.array([e[2] for e in edges], dtype=float)
    q = np.quantile(ws, [0.5, 0.85]) if len(ws) > 3 else [ws.min(), ws.max()]
    for lo, hi, width, col in [(0, q[0], 0.8, "rgba(150,160,175,.35)"), (q[0], q[1], 1.8, "rgba(110,125,145,.5)"),
                               (q[1], 1e12, 3.4, "rgba(19,168,168,.7)")]:
        xs, ys = [], []
        for a, b, w in edges:
            if lo < w <= hi or (lo == 0 and w <= hi):
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=width, color=col),
                                     hoverinfo="skip", showlegend=False))
    nodes = list(G.nodes())
    fq = np.array([wfreq.get(n, 1) for n in nodes], dtype=float)
    cent = nx.degree_centrality(G)
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes,
        textposition="top center", textfont=dict(size=11, color=NAVY),
        marker=dict(size=12 + 30 * (fq / fq.max()), color=[pal[cmap[n] % len(pal)] for n in nodes],
                    line=dict(width=1, color="white")),
        hovertext=[f"<b>{n}</b><br>tweets containing word: {int(wfreq.get(n, 0))}<br>"
                   f"centrality: {cent[n]:.2f}<br>community: {cmap[n] + 1}" for n in nodes],
        hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    style_fig(fig, 560)
    cent_tbl = pd.DataFrame({"word": nodes, "tweets_with_word": [int(wfreq.get(n, 0)) for n in nodes],
                             "degree_centrality": [round(cent[n], 3) for n in nodes],
                             "community": [cmap[n] + 1 for n in nodes]}).sort_values("degree_centrality", ascending=False)
    return fig, G, cent_tbl, [sorted(c, key=lambda w: -wfreq.get(w, 0))[:6] for c in comms]


@st.cache_data(show_spinner=False)
def make_wordcloud(items: tuple, cmap: str):
    wc = WordCloud(width=900, height=400, background_color="white", colormap=cmap, max_words=80,
                   prefer_horizontal=0.9, random_state=1).generate_from_frequencies(dict(items))
    return wc.to_array()


def complaint_table(d: pd.DataFrame) -> pd.DataFrame:
    neg = d[d["airline_sentiment"] == "negative"]
    if neg.empty:
        return pd.DataFrame(columns=["Complaint category", "Frequency", "Share of negative posts (%)", "Cumulative share (%)"])
    vc = neg["complaint"].value_counts()
    t = pd.DataFrame({"Complaint category": vc.index, "Frequency": vc.values})
    t["Share of negative posts (%)"] = t["Frequency"] / t["Frequency"].sum() * 100
    t["Cumulative share (%)"] = t["Share of negative posts (%)"].cumsum()
    return t


def severity_default(cat: str) -> int:
    return DEFAULT_SEVERITY.get(str(cat).lower(), 3)


def priority_table(ct: pd.DataFrame, sev: dict) -> pd.DataFrame:
    if ct.empty:
        return pd.DataFrame()
    t = ct[["Complaint category", "Frequency", "Share of negative posts (%)"]].copy()
    t["Severity assumption (1-5, management input)"] = t["Complaint category"].map(lambda c: sev.get(c, severity_default(c)))
    raw = t["Share of negative posts (%)"] * t["Severity assumption (1-5, management input)"]
    t["Priority score (0-100)"] = raw / raw.max() * 100 if raw.max() > 0 else 0
    t["Proposed managerial action"] = t["Complaint category"].map(lambda c: ACTIONS.get(str(c).lower(), GENERIC_ACTION)[0])
    t["KPI to monitor"] = t["Complaint category"].map(lambda c: ACTIONS.get(str(c).lower(), GENERIC_ACTION)[1])
    t = t.sort_values("Priority score (0-100)", ascending=False).reset_index(drop=True)
    t.insert(0, "Rank", np.arange(1, len(t) + 1))
    return t


@st.cache_data(show_spinner="Fitting topic model...")
def fit_topic_model(texts: tuple, n_topics: int, method: str, seed: int, n_terms: int = 12):
    if method == "NMF (TF-IDF)":
        vec = TfidfVectorizer(max_features=3000, min_df=3, max_df=0.5, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        model = NMF(n_components=n_topics, init="nndsvd", random_state=seed, max_iter=400)
    else:
        vec = CountVectorizer(max_features=3000, min_df=3, max_df=0.5)
        X = vec.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_topics, learning_method="online",
                                          max_iter=15, random_state=seed)
    W = model.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for comp in model.components_:
        idx = comp.argsort()[::-1][:n_terms]
        topics.append({"terms": terms[idx].tolist(), "weights": (comp[idx] / comp[idx].sum()).tolist()})
    s = W.sum(axis=1, keepdims=True)
    W = np.divide(W, s, out=np.zeros_like(W), where=s > 0)
    return {"topics": topics, "W": W.astype(np.float32)}


# ----------------------------------------------------------------------------
# Machine-learning helpers
# ----------------------------------------------------------------------------
def prepare_model_data(df: pd.DataFrame):
    base = df[df["is_observed"] & df["airline_sentiment"].isin(SENT_ORDER)]
    n0 = len(base)
    empty = int((base["text_clean"].str.len() == 0).sum())
    base = base[base["text_clean"].str.len() > 0]
    dedup = base.drop_duplicates("text_norm", keep="first")
    return dedup, {"observed_labelled": n0, "empty_text_removed": empty,
                   "duplicate_text_removed": len(base) - len(dedup), "usable": len(dedup)}


def make_split(obs, focus, test_size, seed, scope):
    aa = obs[obs["airline"] == focus]
    others = obs[obs["airline"] != focus]
    try:
        aa_tr, aa_te = train_test_split(aa, test_size=test_size, stratify=aa["airline_sentiment"], random_state=seed)
    except ValueError:
        aa_tr, aa_te = train_test_split(aa, test_size=test_size, random_state=seed)
    train = aa_tr if scope.startswith("American") else pd.concat([aa_tr, others])
    return train, aa_te


def evaluate(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=SENT_ORDER, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=SENT_ORDER, average="weighted", zero_division=0)
    rep = pd.DataFrame(classification_report(y_true, y_pred, labels=SENT_ORDER, output_dict=True,
                                             zero_division=0)).T.reset_index().rename(columns={"index": "class"})
    cm = confusion_matrix(y_true, y_pred, labels=SENT_ORDER)
    return {"metrics": {"Accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
                        "Precision (macro)": float(p), "Recall (macro)": float(r),
                        "Macro F1": float(f), "Weighted F1": float(fw)},
            "report": rep, "cm": cm, "pred": np.asarray(y_pred)}


@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()


def vader_label(texts, thr=0.05):
    sia = get_vader()
    out = []
    for t in texts:
        c = sia.polarity_scores(t)["compound"]
        out.append("positive" if c >= thr else "negative" if c <= -thr else "neutral")
    return np.array(out)


def train_lr(train, test, balance, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["text_clean"])  # fitted on training data only
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if balance else None, random_state=seed)
    clf.fit(Xtr, train["airline_sentiment"])
    pred = clf.predict(vec.transform(test["text_clean"]))
    names = np.array(vec.get_feature_names_out())
    top = {c: names[np.argsort(clf.coef_[i])[::-1][:10]].tolist() for i, c in enumerate(clf.classes_)}
    return pred, {"vec": vec, "clf": clf}, top


def train_lstm(train, test, cfg, progress_cb):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    seed = cfg["seed"]
    keras.utils.set_random_seed(seed)
    keras.backend.clear_session()
    label_to_id = {c: i for i, c in enumerate(SENT_ORDER)}
    try:
        tr, va = train_test_split(train, test_size=0.15, stratify=train["airline_sentiment"], random_state=seed)
    except ValueError:
        tr, va = train_test_split(train, test_size=0.15, random_state=seed)
    L, V = int(cfg["maxlen"]), int(cfg["vocab"])
    cnt = Counter(w for s in tr["text_clean"] for w in s.split())  # vocabulary from training split only
    vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(V - 2))}

    def encode(texts):
        X = np.zeros((len(texts), L), dtype="int32")
        for i, s in enumerate(texts):
            ids = [vocab.get(w, 1) for w in s.split()][:L]
            X[i, :len(ids)] = ids
        return X

    Xtr, Xva, Xte = encode(tr["text_clean"]), encode(va["text_clean"]), encode(test["text_clean"])
    ytr = tr["airline_sentiment"].map(label_to_id).values
    yva = va["airline_sentiment"].map(label_to_id).values
    model = keras.Sequential([
        keras.Input(shape=(L,), dtype="int32"),
        layers.Embedding(V, 64, mask_zero=True),
        layers.LSTM(48, dropout=0.2),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = None
    if cfg["balance"]:
        classes = np.unique(ytr)
        w = compute_class_weight("balanced", classes=classes, y=ytr)
        cw = {int(c): float(x) for c, x in zip(classes, w)}
    epochs = int(cfg["epochs"])

    class Prog(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            progress_cb((epoch + 1) / epochs,
                        f"Epoch {epoch + 1}/{epochs} - loss {logs.get('loss', 0):.3f}, val_loss {logs.get('val_loss', 0):.3f}")

    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=int(cfg["batch"]),
                     class_weight=cw, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), Prog()])
    probs = model.predict(Xte, batch_size=256, verbose=0)
    pred = np.array(SENT_ORDER)[probs.argmax(axis=1)]
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    params = int(model.count_params())

    def predict_text(text: str):
        s = _clean_one(text)
        p = model.predict(encode([s]), verbose=0)[0]
        return dict(zip(SENT_ORDER, [float(x) for x in p]))

    return pred, history, params, predict_text, len(tr), len(va)


def _clean_one(text: str) -> str:
    t = re.sub(r"https?://\S+|www\.\S+", " ", text)
    t = re.sub(r"@\w+", " ", t).lower().replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t)).strip()


def linear_forecast(y: np.ndarray, horizon: int):
    n = len(y)
    x = np.arange(n)
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    s = np.sqrt(((y - fit) ** 2).sum() / max(n - 2, 1))
    xf = np.arange(n, n + horizon)
    mean_f = a + b * xf
    sxx = ((x - x.mean()) ** 2).sum()
    se = s * np.sqrt(1 + 1 / n + (xf - x.mean()) ** 2 / sxx)
    return mean_f, mean_f - 1.96 * se, mean_f + 1.96 * se, b, 1 - ((y - fit) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================
if not DATA_FILE.exists():
    st.error(f"Dataset not found: `{DATA_FILE.name}` must be in the same folder as app.py "
             f"(expected at `{DATA_FILE}`). Upload it to the repository root and redeploy.")
    st.stop()
_cols, _missing = validate_header(DATA_FILE)
if _missing:
    st.error(f"The CSV is missing required columns: {_missing}. Found columns: {_cols}")
    st.stop()
try:
    df, info = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as exc:  # friendly message instead of a stack trace
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

airlines_all = sorted(df["airline"].dropna().unique().tolist())
focus = next((a for a in airlines_all if "american" in a.lower()), airlines_all[0])
dated = df["date"].dropna()
date_min, date_max = dated.min().date(), dated.max().date()
COMPLAINT_OPTS = sorted(df["complaint"].dropna().unique().tolist())
MODEL_OPTS = [M_VADER, M_LR, M_LSTM]

DEFAULTS = dict(f_airlines=[focus], f_dates=(date_min, date_max), f_sent=SENT_ORDER.copy(),
                f_comp=COMPLAINT_OPTS.copy(), f_mode=MODE_ORIG, f_topn=15, f_bigram_min=5,
                f_topics=5, f_edge=12)
FILTER_KEYS = ["f_airlines", "f_dates", "f_sent", "f_comp", "f_mode"]
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state[k] = DEFAULTS[k]


with st.sidebar:
    st.markdown("## ✈️ Control Panel")
    st.markdown("**1 · Descriptive-analysis filters**")
    st.caption("Affect charts and tables only. They never change model training or the test set.")
    st.multiselect("Airline", airlines_all, key="f_airlines")
    st.date_input("Date range", min_value=date_min, max_value=date_max, key="f_dates")
    st.multiselect("Sentiment category", SENT_ORDER, key="f_sent")
    st.multiselect("Complaint category (applies to negative tweets)", COMPLAINT_OPTS, key="f_comp")
    st.radio("Data mode", [MODE_ORIG, MODE_DEMO], key="f_mode")
    count_box = st.empty()
    st.button("↺ Reset filters", on_click=reset_filters)
    st.divider()
    st.markdown("**2 · Text-analytics controls**")
    st.slider("Top N words / phrases", 5, 40, key="f_topn")
    st.slider("Bigram frequency threshold (min count)", 1, 100, key="f_bigram_min")
    st.slider("Topic count", 2, 10, key="f_topics")
    st.slider("Word-network edge threshold (min co-occurrences)", 2, 200, key="f_edge")
    st.divider()
    st.markdown("**3 · Model training (fixed, filter-independent)**")
    st.caption("Models always use ORIGINAL observed tweets only; the American Airlines hold-out is set by the seed and test size below.")
    st.multiselect("Models to train / evaluate (baseline always included)", MODEL_OPTS,
                   default=[M_VADER, M_LR, M_LSTM], key="t_models")
    st.radio("Training data scope", ["American Airlines only", "All airlines (evaluated on American hold-out)"], key="t_scope")
    with st.expander("Training configuration", expanded=False):
        st.slider("Hold-out test share", 0.15, 0.35, 0.20, 0.05, key="t_test")
        st.number_input("Random seed", 0, 9999, 42, key="t_seed")
        st.select_slider("Batch size (LSTM)", [16, 32, 64, 128], value=64, key="t_batch")
        st.select_slider("Learning rate (LSTM)", [0.0005, 0.001, 0.002, 0.005], value=0.002, key="t_lr")
        st.slider("Maximum epochs (LSTM, early stopping on)", 2, 30, 8, key="t_epochs")
        st.slider("Maximum sequence length", 20, 60, 40, 5, key="t_maxlen")
        st.slider("Vocabulary size (LSTM)", 2000, 15000, 6000, 1000, key="t_vocab")
        st.checkbox("Handle class imbalance (balanced class weights)", True, key="t_balance")
    train_clicked = st.button("🚀 Train / evaluate models", type="primary")
    st.caption("Nothing is trained automatically. The LSTM trains only when this button is pressed.")

# --- Apply descriptive filters -----------------------------------------------
_dr = st.session_state["f_dates"]
_dr = tuple(_dr) if isinstance(_dr, (list, tuple)) else (_dr,)
dates_sel = (_dr[0], _dr[-1]) if len(_dr) >= 1 else (date_min, date_max)
f_args = (st.session_state["f_airlines"], dates_sel, st.session_state["f_sent"],
          st.session_state["f_comp"], COMPLAINT_OPTS, st.session_state["f_mode"])
fdf = apply_filters(df, *f_args)
base_noair = apply_filters(df, *f_args, use_airline=False)
mode_demo = st.session_state["f_mode"] == MODE_DEMO
count_box.markdown(f"**Selected observations: {len(fdf):,}** of {len(df):,}")

# --- Model-training runner (executes only on button press) -------------------


def run_training(df_all: pd.DataFrame):
    cfg = dict(models=st.session_state["t_models"], scope=st.session_state["t_scope"],
               test=st.session_state["t_test"], seed=int(st.session_state["t_seed"]),
               batch=st.session_state["t_batch"], lr=st.session_state["t_lr"],
               epochs=st.session_state["t_epochs"], maxlen=st.session_state["t_maxlen"],
               vocab=st.session_state["t_vocab"], balance=st.session_state["t_balance"])
    res = {"models": {}, "errors": {}, "config": cfg}
    predictors = {}
    with st.status("Training / evaluating models on original observed tweets...", expanded=True) as status:
        obs, dinfo = prepare_model_data(df_all)
        train, test = make_split(obs, focus, cfg["test"], cfg["seed"], cfg["scope"])
        y_test = test["airline_sentiment"].values
        overlap = len(set(train["text_norm"]) & set(test["text_norm"]))
        id_overlap = len(set(train["tweet_id"]) & set(test["tweet_id"])) if "tweet_id" in train else 0
        res["split"] = {**dinfo, "train_n": len(train), "test_n": len(test), "train_scope": cfg["scope"],
                        "text_overlap_train_test": overlap, "tweet_id_overlap_train_test": id_overlap,
                        "generated_records_in_train_or_test": int((~train["is_observed"]).sum() + (~test["is_observed"]).sum()),
                        "test_class_counts": test["airline_sentiment"].value_counts().to_dict(),
                        "train_class_counts": train["airline_sentiment"].value_counts().to_dict()}
        res["test_index"] = test.index.tolist()
        st.write(f"Training records: **{len(train):,}** · Test records (American hold-out): **{len(test):,}** · "
                 f"duplicate texts removed beforehand: **{dinfo['duplicate_text_removed']:,}**")
        maj = train["airline_sentiment"].value_counts().idxmax()
        r = evaluate(y_test, np.full(len(test), maj))
        r.update(train_n=len(train), note=f"Always predicts '{maj}'")
        res["models"][M_MAJ] = r
        if M_VADER in cfg["models"]:
            st.write("Scoring VADER (no training required)...")
            if HAS_VADER:
                r = evaluate(y_test, vader_label(test["text_vader"]))
                r.update(train_n=0, note="Lexicon, thresholds ±0.05 on compound score")
                res["models"][M_VADER] = r
                predictors[M_VADER] = lambda t: dict(get_vader().polarity_scores(t))
            else:
                res["errors"][M_VADER] = "vaderSentiment is not installed."
        if M_LR in cfg["models"]:
            st.write("Fitting TF-IDF + Logistic Regression...")
            try:
                pred, art, top = train_lr(train, test, cfg["balance"], cfg["seed"])
                r = evaluate(y_test, pred)
                r.update(train_n=len(train), note="TF-IDF (1-2 grams) fitted on training split only", top_features=top)
                res["models"][M_LR] = r
                predictors[M_LR] = lambda t, a=art: dict(zip(a["clf"].classes_, a["clf"].predict_proba(
                    a["vec"].transform([_clean_one(t)]))[0].tolist()))
            except Exception as exc:
                res["errors"][M_LR] = str(exc)
        if M_LSTM in cfg["models"]:
            if importlib.util.find_spec("tensorflow") is None:
                res["errors"][M_LSTM] = "TensorFlow is not installed in this environment."
            else:
                st.write("Training Embedding + LSTM network (this may take a few minutes on CPU)...")
                bar = st.progress(0.0, text="Initialising TensorFlow...")
                try:
                    pred, hist, params, ptxt, ntr, nva = train_lstm(
                        train, test, cfg, lambda f, m: bar.progress(min(f, 1.0), text=m))
                    r = evaluate(y_test, pred)
                    r.update(train_n=ntr, val_n=nva, history=hist, params=params,
                             note=f"Embedding(64) → LSTM(48) → Dense(32, ReLU) → Softmax; {params:,} parameters; early stopping")
                    res["models"][M_LSTM] = r
                    predictors[M_LSTM] = ptxt
                    bar.progress(1.0, text="LSTM training complete")
                except Exception as exc:
                    res["errors"][M_LSTM] = f"TensorFlow could not train the network: {exc}"
        res["seconds"] = round(time.time() - t0_train, 1)
        status.update(label=f"Training finished in {res['seconds']}s", state="complete", expanded=False)
    st.session_state["model_results"] = res
    st.session_state["predictors"] = predictors


# --- Header ------------------------------------------------------------------
_obs = df[df["is_observed"]]
_p0, _p1 = _obs["date"].min(), _obs["date"].max()
_period = (f"{_p0:%d %b %Y} – {_p1:%d %b %Y} ({_obs['date'].nunique()} distinct days)"
           if pd.notna(_p0) else "n/a")
st.markdown(
    f"""<div class="hero"><h1>AMERICAN AIRLINES | AI-DRIVEN BRAND REPUTATION INTELLIGENCE</h1>
<p class="sub">Social Media Sentiment • NLP • Deep Learning • Reputation Risk • Business Insights</p>
<p class="desc">IIM Ranchi · AI-Driven Business Analytics · Group Assignment 06 (Prof. Sasadhar Bera). This application analyses public
tweets to measure sentiment, surface recurring complaints, test classification models and convert evidence into managerial actions.
<b>Observed tweet period: {esc(_period)}.</b> Generated records and scenario business metrics are demonstration data only.</p></div>""",
    unsafe_allow_html=True)

if train_clicked:
    t0_train = time.time()
    run_training(df)

if fdf.empty:
    st.warning("No records match the current filters. Widen the date range, add airlines or sentiment categories, "
               "switch data mode, or press **Reset filters** in the sidebar.")
    st.stop()

if mode_demo:
    st.warning("Demonstration mode is ON: generated (non-authentic) records are included in descriptive analyses. "
               "Model evaluation still uses original observed tweets only.", icon="⚠️")

# --- Core derived objects (cheap, recomputed per filter change) --------------
FIGS: dict = {}
S = sentiment_summary(fdf)
n_sel = len(fdf)
pct = dict(zip(S["Sentiment"], S["Share (%)"]))
n_obs_sel = int(fdf["is_observed"].sum())
daily = daily_sentiment(fdf)
CT = complaint_table(fdf)
sev_map = {c: st.session_state.get(f"sev_{c}", severity_default(c)) for c in CT["Complaint category"]} if not CT.empty else {}
PT = priority_table(CT, sev_map)
neg_rate = pct["negative"] / (pct["negative"] + pct["positive"]) * 100 if (pct["negative"] + pct["positive"]) > 0 else float("nan")
dom_complaint = CT.iloc[0]["Complaint category"] if not CT.empty else "n/a"
obs_days = int(fdf.loc[fdf["is_observed"], "date"].nunique())


def terms_for(d: pd.DataFrame, sentiment: str | None, n: int):
    sub = d if sentiment is None else d[d["airline_sentiment"] == sentiment]
    return ngram_freq(sub["text_tokens"], n) if len(sub) else pd.DataFrame(columns=["term", "count"])


tabs = st.tabs(["1 · Executive Overview", "2 · Data & Quality", "3 · Sentiment", "4 · Text & Networks",
                "5 · Complaints & Risk", "6 · Topic Modeling", "7 · AI Model Lab",
                "8 · Reputation & Scenarios", "9 · Insights & Six Sigma", "10 · Reports & Methods"])

# ============================================================================
# TAB 1 - EXECUTIVE OVERVIEW
# ============================================================================
with tabs[0]:
    st.subheader("Executive overview")
    caption("All figures below are computed live from the filtered data. Percentages describe the selected tweets, "
            "not all passengers (Twitter users are a self-selected, complaint-prone sample).")
    c = st.columns(4)
    with c[0]: kpi_card("Selected tweet records", f"{n_sel:,}", "Rows after sidebar filters")
    with c[1]: kpi_card("Observed (authentic) tweets", f"{n_obs_sel:,}", f"{n_obs_sel / n_sel * 100:.1f}% of selection")
    with c[2]: kpi_card("Positive sentiment", f"{pct['positive']:.1f}%", "Share of selected tweets", SENT_COLORS["positive"])
    with c[3]: kpi_card("Negative sentiment", f"{pct['negative']:.1f}%", "Primary reputation-risk signal", SENT_COLORS["negative"])
    c = st.columns(4)
    with c[0]: kpi_card("Neutral sentiment", f"{pct['neutral']:.1f}%", "Informational / unclear tone", SENT_COLORS["neutral"])
    with c[1]: kpi_card("Dominant complaint", str(dom_complaint), "Largest share of negative posts", NAVY)
    with c[2]: kpi_card("Complaint categories", f"{CT.shape[0]}", "Distinct reasons in selection", NAVY)
    with c[3]: kpi_card("Negative sentiment rate", "n/a" if np.isnan(neg_rate) else f"{neg_rate:.1f}%",
                        "Negative ÷ (positive + negative)", SENT_COLORS["negative"])
    st.write("")
    a, b = st.columns(2)
    with a:
        fig = px.pie(S, names="Sentiment", values="Tweets", hole=0.55, color="Sentiment", color_discrete_map=SENT_COLORS)
        fig.update_traces(textinfo="percent+label", sort=False)
        style_fig(fig, 340, "Sentiment distribution")
        FIGS["sentiment_distribution"] = fig
        show_fig(fig, "t1_pie")
    with b:
        if len(daily) >= 2:
            long = daily.melt("date", [f"{s}_pct" for s in SENT_ORDER], var_name="s", value_name="pct")
            long["s"] = long["s"].str.replace("_pct", "")
            fig = px.line(long, x="date", y="pct", color="s", markers=True, color_discrete_map=SENT_COLORS)
            fig.update_yaxes(title="% of tweets that day")
            style_fig(fig, 340, "Daily sentiment trend")
            FIGS["sentiment_trend"] = fig
            show_fig(fig, "t1_trend")
        else:
            st.info("A trend needs at least two distinct dates in the selection.")
    a, b = st.columns(2)
    with a:
        if not CT.empty:
            top = CT.head(8).iloc[::-1]
            fig = px.bar(top, x="Frequency", y="Complaint category", orientation="h", text="Frequency",
                         color_discrete_sequence=[SENT_COLORS["negative"]])
            style_fig(fig, 340, "Top complaint categories")
            show_fig(fig, "t1_comp")
        else:
            st.info("No negative tweets in the current selection.")
    with b:
        tp, tn = terms_for(fdf, "positive", 1).head(8), terms_for(fdf, "negative", 1).head(8)
        fig = go.Figure()
        if len(tn):
            fig.add_bar(y=tn["term"][::-1], x=tn["count"][::-1], orientation="h", name="Negative themes",
                        marker_color=SENT_COLORS["negative"])
        if len(tp):
            fig.add_bar(y=tp["term"][::-1], x=tp["count"][::-1], orientation="h", name="Positive themes",
                        marker_color=SENT_COLORS["positive"])
        style_fig(fig, 340, "Top positive and negative words")
        fig.update_layout(barmode="group")
        show_fig(fig, "t1_themes")
    st.markdown("#### Key business takeaways (auto-generated)")
    take = [f"**{pct['negative']:.1f}%** of the {n_sel:,} selected tweets are negative versus **{pct['positive']:.1f}%** positive "
            f"(net sentiment {pct['positive'] - pct['negative']:+.1f} points)."]
    if not CT.empty:
        take.append(f"**{CT.iloc[0]['Complaint category']}** is the largest complaint driver "
                    f"({CT.iloc[0]['Share of negative posts (%)']:.1f}% of negative posts); the top three categories account for "
                    f"**{CT['Share of negative posts (%)'].head(3).sum():.1f}%**.")
    if len(daily) >= 3:
        w = daily[daily["total"] >= 10]
        if len(w):
            wr = w.loc[w["negative_pct"].idxmax()]
            take.append(f"Negative sentiment peaked on **{wr['date']:%d %b %Y}** ({wr['negative_pct']:.1f}% of {int(wr['total'])} tweets).")
    take.append(f"Observed tweets in the selection span only **{obs_days} distinct day(s)** – treat trends as short-term signals, not long-run history.")
    for t in take:
        st.markdown(f"- {t}")
    with st.expander("Project objectives and research questions", expanded=False):
        st.markdown("""**Business problem:** How can American Airlines systematically analyse public social-media discussions to understand
customer sentiment, identify recurring complaints, detect reputation risks and prioritise managerial action?

1. What are the positive, neutral and negative sentiment patterns? → Tabs 1, 3
2. Which recurring problems drive negative brand discussions? → Tabs 4, 5, 6
3. How do sentiment and reputation-risk indicators change across the observation period? → Tabs 3, 8
4. How effectively can machine learning and neural networks classify sentiment? → Tab 7
5. What managerial actions could address the identified problems? → Tabs 5, 9 (DMAIC), 10""")

# ============================================================================
# TAB 2 - DATA EXPLORER AND QUALITY
# ============================================================================
DESC = {
    "tweet_id": "Unique tweet identifier", "airline_sentiment": "Sentiment label (positive/neutral/negative)",
    "airline_sentiment_confidence": "Annotator confidence in the sentiment label", "negativereason": "Complaint reason (negative tweets)",
    "negativereason_confidence": "Annotator confidence in the complaint reason", "airline": "Airline mentioned",
    "airline_sentiment_gold": "Gold-standard sentiment (sparse)", "name": "Twitter user name", "negativereason_gold": "Gold complaint reason (sparse)",
    "retweet_count": "Number of retweets", "text": "Raw tweet text (unstructured)", "tweet_coord": "Tweet coordinates (sparse)",
    "tweet_created": "Tweet timestamp with timezone", "tweet_location": "User-declared location", "user_timezone": "User timezone",
    "record_origin": "Provenance: original vs generated record", "record_is_observed": "YES if an authentic observed tweet",
    "tweet_date": "Tweet date", "tweet_month": "Tweet month",
    "original_airline_daily_tweet_count": "Daily observed tweet count per airline (derived)",
    "original_airline_daily_negative_sentiment_pct": "Daily negative % per airline from observed tweets (derived)",
    "business_data_origin": "Origin of scenario business columns", "business_metric_unit": "Grain of scenario values (repeated per tweet)",
    "date": "Parsed tweet date (app-derived)", "is_observed": "Boolean observed flag (app-derived)", "record_type": "Observed vs generated label (app-derived)",
    "complaint": "Complaint category for negative tweets (app-derived)", "text_vader": "Text without URLs/@mentions, case kept (app-derived)",
    "text_norm": "Lower-cased text used for duplicate detection (app-derived)", "text_clean": "Letters-only lower-case text (app-derived)",
    "text_tokens": "Stop-word-free tokens for frequency analysis (app-derived)", "n_words_raw": "Word count of raw tweet",
    "n_tokens": "Token count after cleaning",
}
with tabs[1]:
    st.subheader("Data explorer and data-quality assessment")
    c = st.columns(4)
    with c[0]: kpi_card("Rows × columns (file)", f"{info['raw_rows']:,} × {info['raw_cols']}", "Loaded in full, nothing dropped")
    with c[1]: kpi_card("Observed / generated", f"{int(df['is_observed'].sum()):,} / {int((~df['is_observed']).sum()):,}", "From provenance flags")
    with c[2]: kpi_card("Airlines", f"{df['airline'].nunique()}", ", ".join(airlines_all)[:60], NAVY)
    with c[3]: kpi_card("Observed date range", f"{_p0:%d %b} – {_p1:%d %b %Y}" if pd.notna(_p0) else "n/a", f"{_obs['date'].nunique()} distinct days", NAVY)
    st.markdown("##### Structured metadata vs unstructured text")
    st.markdown("**Structured** fields (airline, sentiment label, complaint reason, dates, retweets, scenario indicators) can be counted, "
                "grouped and modelled directly. **Unstructured** text (the tweet itself) has no fixed schema, so it needs NLP: cleaning, "
                "tokenisation, n-grams, topic models and neural embeddings before it becomes analysable.")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Provenance summary")
        prov_cols = [x for x in ["record_origin", "record_is_observed", "business_data_origin", "business_metric_unit"] if x in df]
        prov = df.groupby(prov_cols, dropna=False).size().reset_index(name="Rows") if prov_cols else pd.DataFrame()
        show_df(prov)
        caption("Observed = authentic tweets. Generated rows and all scenario_* business metrics are hypothetical demonstration data.")
    with b:
        st.markdown("##### Airline distribution")
        ad = df.groupby(["airline", "record_type"]).size().reset_index(name="Rows")
        fig = px.bar(ad, x="airline", y="Rows", color="record_type", color_discrete_sequence=[TEAL, "#9AA7B8"])
        show_fig(style_fig(fig, 300), "t2_air")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Missing-value analysis (full file)")
        mv = pd.DataFrame({"Column": df.columns, "Missing": df.isna().sum().values})
        mv["Missing (%)"] = mv["Missing"] / len(df) * 100
        mv = mv[~mv["Column"].isin(["text_vader", "text_norm", "text_clean", "text_tokens"])].sort_values("Missing (%)", ascending=False)
        fig = px.bar(mv.head(14).iloc[::-1], x="Missing (%)", y="Column", orientation="h", color_discrete_sequence=[NAVY])
        show_fig(style_fig(fig, 380), "t2_miss")
        caption("Negative-reason fields are expected to be empty for non-negative tweets; gold/coordinate fields are sparse by design.")
    with b:
        st.markdown("##### Duplicate analysis")
        dup_id = int(df.duplicated("tweet_id").sum())
        dup_txt_all = int(df.duplicated("text_norm").sum())
        o = df[df["is_observed"]]
        dup_txt_obs = int(o.duplicated("text_norm").sum())
        dup_sel = int(fdf.duplicated("text_norm").sum())
        show_df(pd.DataFrame({"Check": ["Duplicate tweet_id (all rows)", "Duplicate cleaned text (all rows)",
                                        "Duplicate cleaned text (observed only)", "Duplicate cleaned text (current selection)"],
                              "Rows": [dup_id, dup_txt_all, dup_txt_obs, dup_sel]}))
        caption("Duplicates are kept for descriptive counts but removed before model training to avoid train/test leakage.")
        st.markdown("##### Data-quality indicators")
        conf = df["airline_sentiment_confidence"].mean() if "airline_sentiment_confidence" in df else np.nan
        dq = pd.DataFrame({"Indicator": ["Required-column completeness", "Valid sentiment labels", "Parsable dates",
                                         "Non-empty cleaned text", "Mean label confidence"],
                           "Value": [f"{100 - df[REQUIRED_COLS].isna().mean().mean() * 100:.1f}%",
                                     f"{100 - info['bad_labels'] / len(df) * 100:.2f}%",
                                     f"{100 - info['bad_dates'] / len(df) * 100:.2f}%",
                                     f"{100 - info['empty_text'] / len(df) * 100:.2f}%",
                                     "n/a" if np.isnan(conf) else f"{conf:.3f}"]})
        show_df(dq)
    a, b = st.columns(2)
    with a:
        st.markdown("##### Sentiment distribution by record type (current selection)")
        sd = fdf.groupby(["record_type", "airline_sentiment"]).size().reset_index(name="Tweets")
        fig = px.bar(sd, x="record_type", y="Tweets", color="airline_sentiment", color_discrete_map=SENT_COLORS, barmode="stack")
        show_fig(style_fig(fig, 300), "t2_sd")
    with b:
        st.markdown("##### Cleaning methodology")
        st.markdown(f"""1. Load all {info['raw_rows']:,} rows (no row is dropped). Rows with unparsable dates ({info['bad_dates']}) or invalid labels ({info['bad_labels']}) are only excluded from views that need those fields.
2. Parse `tweet_date` (fallback `tweet_created`) and derive the observed flag from `record_is_observed`.
3. Remove URLs and @mentions, decode `&amp;`, lower-case, strip non-letters.
4. Tokenise; remove English + domain stop-words (keeping *not/no/never*).
5. Negative tweets without a reason are labelled *Not specified*.
6. Before modelling: drop empty texts and duplicate texts (documented in Tab 7).""")
    st.markdown("##### Before-and-after preprocessing examples")
    seed_ex = st.number_input("Sample seed", 0, 999, 1, key="ex_seed")
    ex = fdf.sample(min(6, len(fdf)), random_state=int(seed_ex))[["text", "text_clean", "text_tokens"]]
    show_df(ex.rename(columns={"text": "Raw tweet", "text_clean": "Cleaned text", "text_tokens": "Tokens (stop-words removed)"}))
    st.markdown("##### Searchable table of selected records")
    q = st.text_input("Search tweet text contains…", "")
    show_cols = [c for c in ["tweet_id", "date", "airline", "airline_sentiment", "complaint", "retweet_count", "record_type", "text"] if c in fdf]
    view = fdf[fdf["text"].str.contains(q, case=False, regex=False)] if q else fdf
    st.caption(f"{len(view):,} matching records (showing first 2,000)")
    show_df(view[show_cols].head(2000))
    st.download_button("⬇ Download filtered records (CSV)", csv_bytes(view[show_cols]), "filtered_records.csv", "text/csv")
    st.markdown("##### Data dictionary (auto-generated)")
    dd = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values,
                       "Non-null": df.notna().sum().values, "Unique": df.nunique().values,
                       "Description": [DESC.get(c, "Scenario business indicator (hypothetical)" if c.startswith("scenario_") else "—") for c in df.columns]})
    show_df(dd)
    st.download_button("⬇ Download data dictionary (CSV)", csv_bytes(dd), "data_dictionary.csv", "text/csv")

# ============================================================================
# TAB 3 - SENTIMENT INTELLIGENCE
# ============================================================================
with tabs[2]:
    st.subheader("Sentiment intelligence")
    caption("Defaults to original observed tweets. Twitter users are a biased sample; results do not represent all passengers.")
    net = pct["positive"] - pct["negative"]
    a, b = st.columns([1, 2])
    with a:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=net, number={"suffix": " pts"},
                                     title={"text": "Sentiment balance (positive% − negative%)"},
                                     gauge={"axis": {"range": [-100, 100]}, "bar": {"color": NAVY},
                                            "steps": [{"range": [-100, -10], "color": "#F6D0D0"}, {"range": [-10, 10], "color": "#FBEFCB"},
                                                      {"range": [10, 100], "color": "#D3EEDF"}]}))
        show_fig(style_fig(fig, 300), "t3_gauge")
    with b:
        fig = px.bar(S, x="Sentiment", y="Tweets", color="Sentiment", color_discrete_map=SENT_COLORS, text="Tweets")
        show_fig(style_fig(fig, 300, "Sentiment counts (selection)"), "t3_bar")
    st.markdown(f"**Interpretation:** negative tweets {'outnumber' if net < 0 else 'are outweighed by'} positive tweets by "
                f"{abs(net):.1f} points. Neutral tweets form {pct['neutral']:.1f}% of the conversation.")
    a, b = st.columns(2)
    with a:
        comp = base_noair.groupby(["airline", "airline_sentiment"]).size().reset_index(name="n")
        comp["pct"] = comp["n"] / comp.groupby("airline")["n"].transform("sum") * 100
        fig = px.bar(comp, x="airline", y="pct", color="airline_sentiment", color_discrete_map=SENT_COLORS,
                     category_orders={"airline_sentiment": SENT_ORDER}, labels={"pct": "% of tweets"})
        show_fig(style_fig(fig, 340, "Airline-wise sentiment (all airlines, same date/sentiment/mode filters)"), "t3_air")
    with a: pass
    with b:
        by = st.selectbox("Compare sentiment by", ["airline", "record_type", "tweet_month", "retweeted?"], key="t3_by")
        tmp = base_noair if by == "airline" else fdf
        tmp = tmp.assign(**{"retweeted?": np.where(tmp["retweet_count"] > 0, "Retweeted", "Not retweeted")})
        g = tmp.groupby([by, "airline_sentiment"]).size().reset_index(name="n")
        g["pct"] = g["n"] / g.groupby(by)["n"].transform("sum") * 100
        fig = px.bar(g, x=by, y="pct", color="airline_sentiment", color_discrete_map=SENT_COLORS, labels={"pct": "% of tweets"})
        show_fig(style_fig(fig, 340, f"Sentiment mix by {by}"), "t3_by_fig")
    if len(daily) >= 2:
        a, b = st.columns(2)
        with a:
            fig = px.area(daily, x="date", y=SENT_ORDER, color_discrete_map=SENT_COLORS)
            fig.update_layout(legend_title="")
            show_fig(style_fig(fig, 330, "Daily tweet volume by sentiment"), "t3_vol")
        with b:
            fig = px.line(daily, x="date", y="negative_pct", markers=True, color_discrete_sequence=[SENT_COLORS["negative"]])
            fig.update_yaxes(title="Negative %")
            FIGS["daily_negative_pct"] = fig
            show_fig(style_fig(fig, 330, "Daily negative sentiment %"), "t3_neg")
        minvol = st.slider("Minimum daily tweets for 'high-negativity' ranking", 1, 100, 10, key="t3_minvol")
        hn = daily[daily["total"] >= minvol].sort_values("negative_pct", ascending=False).head(5)
        st.markdown("##### High-negativity dates")
        show_df(hn[["date", "total", "negative", "negative_pct", "net_sentiment"]].rename(
            columns={"total": "Tweets", "negative": "Negative tweets", "negative_pct": "Negative %", "net_sentiment": "Net sentiment"}).round(1))
    else:
        st.info("Daily trends need at least two distinct dates in the selection.")
    st.markdown("##### Retweet engagement by sentiment")
    eng = fdf.groupby("airline_sentiment").agg(Tweets=("text", "size"), Mean_retweets=("retweet_count", "mean"),
                                               Share_retweeted=("retweet_count", lambda x: (x > 0).mean() * 100)).reindex(SENT_ORDER).reset_index()
    a, b = st.columns([1, 2])
    with a:
        show_df(eng.round(2))
    with b:
        fig = px.bar(eng, x="airline_sentiment", y="Share_retweeted", color="airline_sentiment", color_discrete_map=SENT_COLORS,
                     labels={"Share_retweeted": "% of tweets retweeted at least once"})
        show_fig(style_fig(fig, 260), "t3_rt")
    sm = S.copy()
    st.download_button("⬇ Download sentiment summary (CSV)", csv_bytes(sm), "sentiment_summary.csv", "text/csv")

# ============================================================================
# TAB 4 - TEXT ANALYTICS AND NETWORKS
# ============================================================================
with tabs[3]:
    st.subheader("Text analytics and word networks")
    tok_total = int(fdf["n_tokens"].sum())
    c = st.columns(4)
    with c[0]: kpi_card("Avg words / tweet (raw)", f"{fdf['n_words_raw'].mean():.1f}", "Before cleaning", NAVY)
    with c[1]: kpi_card("Avg tokens / tweet", f"{fdf['n_tokens'].mean():.1f}", "After cleaning + stop-words", NAVY)
    with c[2]: kpi_card("Total tokens", f"{tok_total:,}", "Selected corpus")
    uni_all = terms_for(fdf, None, 1)
    with c[3]: kpi_card("Distinct terms", f"{len(uni_all):,}", "Unigrams (top 1,500 kept)")
    caption("Pipeline: remove URLs/@mentions → lower-case → letters only → tokenise → remove stop-words → count unigrams / bigrams / co-occurrence.")
    sub = st.radio("Corpus for frequency charts", ["All selected", "positive", "neutral", "negative"], horizontal=True, key="t4_sub")
    sub_df = fdf if sub == "All selected" else fdf[fdf["airline_sentiment"] == sub]
    topn, bmin = st.session_state["f_topn"], st.session_state["f_bigram_min"]
    col_c = SENT_COLORS.get(sub, TEAL)
    a, b = st.columns(2)
    with a:
        u = ngram_freq(sub_df["text_tokens"], 1).head(topn) if len(sub_df) else pd.DataFrame(columns=["term", "count"])
        fig = px.bar(u.iloc[::-1], x="count", y="term", orientation="h", color_discrete_sequence=[col_c])
        show_fig(style_fig(fig, 460, f"Top {topn} unigrams"), "t4_uni")
    with b:
        bg = ngram_freq(sub_df["text_tokens"], 2) if len(sub_df) else pd.DataFrame(columns=["term", "count"])
        bg = bg[bg["count"] >= bmin].head(topn)
        if bg.empty:
            st.info(f"No bigram reaches the threshold of {bmin}. Lower the bigram frequency threshold in the sidebar.")
        else:
            fig = px.bar(bg.iloc[::-1], x="count", y="term", orientation="h", color_discrete_sequence=[TEAL])
            show_fig(style_fig(fig, 460, f"Top {topn} bigrams (count ≥ {bmin})"), "t4_bi")
    st.markdown("##### Word clouds")
    a, b = st.columns(2)
    for col, s_name, cmap in [(a, "positive", "Greens"), (b, "negative", "Reds")]:
        with col:
            f = terms_for(fdf, s_name, 1).head(80)
            st.markdown(f"**{s_name.capitalize()} tweets**")
            if f.empty:
                st.info(f"No {s_name} tweets in the selection.")
            elif HAS_WC:
                show_img(make_wordcloud(tuple(zip(f["term"], f["count"])), cmap))
            else:
                show_fig(style_fig(px.bar(f.head(20), x="term", y="count", color_discrete_sequence=[SENT_COLORS[s_name]]), 320), f"t4_wc_{s_name}")
    st.markdown("##### Word co-occurrence network")
    a, b, c3 = st.columns(3)
    net_sub = a.selectbox("Network corpus", ["All selected", "negative", "positive"], key="t4_netsub")
    max_nodes = b.slider("Maximum words (nodes) shown", 10, 60, 30, key="t4_nodes")
    vocab_n = c3.slider("Candidate vocabulary (top words)", 30, 120, 70, 10, key="t4_vocab")
    net_df = fdf if net_sub == "All selected" else fdf[fdf["airline_sentiment"] == net_sub]
    if len(net_df) < 20:
        st.info("At least 20 tweets are needed to build a meaningful network.")
    else:
        pairs, wf = cooccurrence(net_df["text_tokens"], vocab_n)
        res_net = build_network(pairs, wf, st.session_state["f_edge"], max_nodes)
        if res_net is None:
            st.info(f"No word pair co-occurs at least {st.session_state['f_edge']} times. Lower the edge threshold in the sidebar.")
        else:
            fig, G, cent, comms = res_net
            show_fig(fig, "t4_net")
            caption("Nodes = words (size = number of tweets containing the word); edges = words appearing in the same tweet; colours = communities detected by modularity.")
            top_pairs = pairs.head(3)
            st.markdown("**What this reveals:** the strongest word pairs are " +
                        ", ".join(f"*{r.word_1}–{r.word_2}* ({r.weight})" for r in top_pairs.itertuples()) +
                        f". The network contains {G.number_of_nodes()} words in {len(comms)} community(ies); " +
                        "; ".join(f"community {i + 1}: {', '.join(c_[:4])}" for i, c_ in enumerate(comms[:3])) +
                        ". Dense communities indicate recurring conversation themes; high-centrality words act as bridges between themes.")
            a, b = st.columns(2)
            with a:
                st.markdown("**Co-occurrence frequency table**")
                show_df(pairs.head(40).rename(columns={"weight": "co-occurrences"}))
            with b:
                st.markdown("**Node centrality**")
                show_df(cent.head(25))

# ============================================================================
# TAB 5 - COMPLAINTS AND REPUTATION DRIVERS
# ============================================================================
with tabs[4]:
    st.subheader("Complaints and reputation drivers")
    if CT.empty:
        st.info("No negative tweets in the current selection – include the negative sentiment category to analyse complaints.")
    else:
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure()
            fig.add_bar(x=CT["Complaint category"], y=CT["Frequency"], name="Frequency", marker_color=SENT_COLORS["negative"])
            fig.add_scatter(x=CT["Complaint category"], y=CT["Cumulative share (%)"], name="Cumulative %", yaxis="y2",
                            mode="lines+markers", line=dict(color=NAVY))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="Cumulative %"))
            FIGS["pareto"] = fig
            show_fig(style_fig(fig, 380, "Pareto analysis of complaint categories"), "t5_pareto")
        with b:
            show_df(CT.round(1))
            st.download_button("⬇ Download complaint analysis (CSV)", csv_bytes(CT), "complaint_analysis.csv", "text/csv")
        n80 = int((CT["Cumulative share (%)"] < 80).sum() + 1)
        st.markdown(f"**Pareto reading:** the top **{min(n80, len(CT))}** of {len(CT)} categories explain at least 80% of negative posts – "
                    f"{', '.join(CT['Complaint category'].head(min(n80, len(CT))))}.")
        cs = [c for c in CT["Complaint category"] if "service" in str(c).lower()]
        if cs:
            r0 = CT[CT["Complaint category"] == cs[0]].iloc[0]
            st.markdown(f"**Customer-service complaints:** {int(r0['Frequency']):,} tweets ({r0['Share of negative posts (%)']:.1f}% of negative posts).")
        a, b = st.columns(2)
        with a:
            neg_all = base_noair[base_noair["airline_sentiment"] == "negative"]
            top_cats = CT["Complaint category"].head(8).tolist()
            ac = neg_all[neg_all["complaint"].isin(top_cats)].groupby(["airline", "complaint"]).size().reset_index(name="n")
            ac["share"] = ac["n"] / neg_all.groupby("airline").size().reindex(ac["airline"]).values * 100
            fig = px.bar(ac, x="complaint", y="share", color="airline", barmode="group", labels={"share": "% of airline's negative tweets"})
            show_fig(style_fig(fig, 360, "Airline-wise complaint comparison"), "t5_air")
        with b:
            nn = fdf[(fdf["airline_sentiment"] == "negative") & fdf["complaint"].isin(CT["Complaint category"].head(6))]
            if nn["date"].nunique() >= 2:
                bins, unit = time_bin(nn)
                tr = nn.assign(period=bins).groupby(["period", "complaint"]).size().reset_index(name="n")
                fig = px.line(tr, x="period", y="n", color="complaint", markers=True)
                show_fig(style_fig(fig, 360, f"Complaint trends (per {unit}, top 6 categories)"), "t5_trend")
            else:
                st.info("Complaint trends need at least two distinct dates.")
        st.markdown("##### Severity assumptions (management input – not measured from the data)")
        st.caption("Rate how damaging each complaint type is for the brand (1 = minor, 5 = critical). Defaults are illustrative; adjust to your judgement.")
        cols = st.columns(5)
        for i, cat in enumerate(CT["Complaint category"]):
            with cols[i % 5]:
                st.number_input(str(cat), 1, 5, severity_default(cat), key=f"sev_{cat}")
        st.markdown("##### Severity × frequency matrix")
        fig = px.scatter(PT, x="Share of negative posts (%)", y="Severity assumption (1-5, management input)", size="Priority score (0-100)",
                         text="Complaint category", color="Priority score (0-100)", color_continuous_scale=["#9AD6D6", "#F2B84B", "#D64545"], size_max=45)
        fig.update_traces(textposition="top center")
        fig.update_yaxes(range=[0.5, 5.7])
        FIGS["severity_matrix"] = fig
        show_fig(style_fig(fig, 400), "t5_matrix")
        st.markdown("##### Risk priority table")
        st.caption("Priority score = share of negative posts × severity assumption, rescaled so the highest category = 100. "
                   "Severity is a management input; frequency is measured.")
        show_df(PT.round(1))
        st.download_button("⬇ Download risk priority table (CSV)", csv_bytes(PT), "risk_priority_table.csv", "text/csv")
        st.markdown("##### Explore a complaint category")
        cat = st.selectbox("Complaint category", CT["Complaint category"].tolist(), key="t5_cat")
        cdf = fdf[(fdf["airline_sentiment"] == "negative") & (fdf["complaint"] == cat)]
        a, b = st.columns([1, 2])
        with a:
            ph = ngram_freq(cdf["text_tokens"], 2)
            ph = ph[ph["count"] >= 2].head(15)
            st.markdown("**Frequent phrases**")
            show_df(ph.rename(columns={"term": "bigram"}) if len(ph) else pd.DataFrame({"info": ["Not enough repeated phrases"]}))
        with b:
            st.markdown(f"**Associated tweets** ({len(cdf):,}; top 25 by retweets)")
            show_df(cdf.sort_values("retweet_count", ascending=False)[["date", "airline", "retweet_count", "text"]].head(25))

# ============================================================================
# TAB 6 - TOPIC MODELING
# ============================================================================
with tabs[5]:
    st.subheader("Topic modeling")
    st.markdown("**Sentiment classification vs topic modeling:** classification predicts a *label* (positive/neutral/negative) from labelled "
                "examples (supervised). Topic modeling discovers *themes* (groups of co-occurring words) without labels (unsupervised). "
                "Combining both shows *what* people talk about and *how* they feel about it.")
    a, b, c3, d = st.columns(4)
    method = a.selectbox("Algorithm", ["NMF (TF-IDF)", "LDA (counts)"], key="tp_method")
    k_topics = b.number_input("Topic count (from sidebar)", 2, 10, st.session_state["f_topics"], disabled=True)
    max_docs = c3.slider("Max tweets used", 500, 8000, 4000, 500, key="tp_max")
    seed_tp = d.number_input("Seed", 0, 999, 7, key="tp_seed")
    corpus = fdf[fdf["n_tokens"] >= 3]
    if len(corpus) > max_docs:
        corpus = corpus.sample(max_docs, random_state=int(seed_tp))
    sig = f"{method}|{st.session_state['f_topics']}|{len(corpus)}|{int(pd.util.hash_pandas_object(corpus['text_tokens']).sum())}"
    if st.button("Extract topics", type="primary", key="tp_btn"):
        if len(corpus) < 100:
            st.warning("At least 100 tweets with 3+ tokens are needed for stable topics.")
        else:
            try:
                out = fit_topic_model(tuple(corpus["text_tokens"]), int(st.session_state["f_topics"]), method, int(seed_tp))
                st.session_state["topic_result"] = {"out": out, "index": corpus.index.to_numpy(), "sig": sig,
                                                    "method": method, "k": int(st.session_state["f_topics"])}
            except Exception as exc:
                st.error(f"Topic model failed: {exc}")
    tr_ = st.session_state.get("topic_result")
    TOPIC_EXPORT = None
    if not tr_:
        st.info("Press **Extract topics** to fit the model on the current selection. The fitted model is cached and not refitted on every interaction.")
    else:
        if tr_["sig"] != sig:
            st.warning("Filters or settings changed since topics were extracted – results below are for the earlier selection. Press **Extract topics** to refresh.")
        out, idx, K = tr_["out"], tr_["index"], tr_["k"]
        W = out["W"]
        labels = []
        with st.expander("Optional: edit topic labels", expanded=False):
            for i, t in enumerate(out["topics"]):
                labels.append(st.text_input(f"Topic {i + 1}", f"T{i + 1}: " + ", ".join(t["terms"][:3]), key=f"tl_{i}_{tr_['sig'][:8]}"))
        if not labels:
            labels = [f"T{i + 1}: " + ", ".join(t["terms"][:3]) for i, t in enumerate(out["topics"])]
        dom = W.argmax(axis=1)
        prev = pd.DataFrame({"Topic": labels, "Mean weight (%)": W.mean(axis=0) * 100,
                             "Dominant in tweets": np.bincount(dom, minlength=K)})
        prev["Dominant share (%)"] = prev["Dominant in tweets"] / prev["Dominant in tweets"].sum() * 100
        cols = st.columns(min(K, 3))
        for i, t in enumerate(out["topics"]):
            with cols[i % len(cols)]:
                tt = pd.DataFrame({"term": t["terms"][:10], "weight": t["weights"][:10]}).iloc[::-1]
                fig = px.bar(tt, x="weight", y="term", orientation="h", color_discrete_sequence=[TEAL])
                show_fig(style_fig(fig, 300, labels[i][:40]), f"t6_topic_{i}")
        a, b = st.columns(2)
        with a:
            fig = px.bar(prev, x="Topic", y="Dominant share (%)", color_discrete_sequence=[NAVY])
            show_fig(style_fig(fig, 320, "Topic prevalence (dominant-topic share)"), "t6_prev")
        with b:
            sents = df.loc[idx, "airline_sentiment"].values
            ct_ = pd.crosstab(pd.Series(np.array(labels)[dom], name="Topic"), pd.Series(sents, name="Sentiment"))
            ct_ = ct_.reindex(columns=SENT_ORDER, fill_value=0)
            long = (ct_.div(ct_.sum(axis=1), axis=0) * 100).reset_index().melt("Topic", var_name="Sentiment", value_name="%")
            fig = px.bar(long, x="Topic", y="%", color="Sentiment", color_discrete_map=SENT_COLORS)
            show_fig(style_fig(fig, 320, "Topic vs sentiment (% within topic)"), "t6_sent")
        st.markdown("##### Representative tweets")
        pick = st.selectbox("Topic", labels, key="tp_pick")
        ti = labels.index(pick)
        order = np.argsort(-W[:, ti])[:8]
        rep = df.loc[idx[order], ["airline_sentiment", "text"]].assign(topic_weight=W[order, ti].round(2))
        show_df(rep)
        show_df(prev.round(1))
        TOPIC_EXPORT = pd.concat([pd.DataFrame({"tweet_index": idx, "dominant_topic": np.array(labels)[dom],
                                                "airline_sentiment": df.loc[idx, "airline_sentiment"].values,
                                                "text": df.loc[idx, "text"].values})], axis=1)
        st.download_button("⬇ Download topic assignments (CSV)", csv_bytes(TOPIC_EXPORT), "topic_modeling_output.csv", "text/csv")

# ============================================================================
# TAB 7 - AI MODEL LAB
# ============================================================================
with tabs[6]:
    st.subheader("AI model lab")
    st.markdown("**Task:** classify tweets as negative / neutral / positive. **Evaluation:** a fixed hold-out of *original observed American Airlines* "
                "tweets. Generated records are never used for training or testing. Tokenisers, TF-IDF and vocabularies are fitted on training data only; "
                "duplicate texts are removed before splitting.")
    tf_present = importlib.util.find_spec("tensorflow") is not None
    c = st.columns(3)
    with c[0]: kpi_card("TensorFlow", "Installed" if tf_present else "Not installed", "Initialised on first training click", NAVY)
    with c[1]: kpi_card("VADER", "Available" if HAS_VADER else "Missing", "Lexicon model, no training", NAVY)
    MR = st.session_state.get("model_results")
    with c[2]: kpi_card("Training status", "Trained" if MR else "Not trained yet", "Use the sidebar button", SENT_COLORS["positive"] if MR else SENT_COLORS["neutral"])
    if not MR:
        st.info("No model has been trained in this session, so no performance metrics are shown. Configure options in the sidebar and press "
                "**🚀 Train / evaluate models**.")
    else:
        sp, cfg = MR["split"], MR["config"]
        st.markdown("##### Data split and leakage checks")
        show_df(pd.DataFrame({"Item": ["Observed labelled tweets", "Empty texts removed", "Duplicate texts removed", "Usable unique tweets",
                                       "Training records", "Test records (American hold-out)", "Training scope",
                                       "Text overlap train∩test", "tweet_id overlap train∩test", "Generated records in train/test"],
                              "Value": [sp["observed_labelled"], sp["empty_text_removed"], sp["duplicate_text_removed"], sp["usable"], sp["train_n"],
                                        sp["test_n"], sp["train_scope"], sp["text_overlap_train_test"], sp["tweet_id_overlap_train_test"],
                                        sp["generated_records_in_train_or_test"]]}).astype(str))
        st.caption(f"Class counts - train: {sp['train_class_counts']} · test: {sp['test_class_counts']} · seed {cfg['seed']} · trained in {MR['seconds']}s. "
                   "Near-duplicates (e.g. retweets with edits) may remain, so scores can be slightly optimistic.")
        for m, msg in MR["errors"].items():
            st.error(f"{m}: {msg} – other tabs and models are unaffected; no substitute results are shown.")
        rows = []
        for m, r in MR["models"].items():
            rows.append({"Model": m, **{k: round(v, 4) for k, v in r["metrics"].items()}, "Training records": r.get("train_n", 0),
                         "Test records": sp["test_n"], "Scope": sp["train_scope"] if r.get("train_n", 0) else "n/a"})
        comp_tbl = pd.DataFrame(rows)
        st.markdown("##### Model comparison")
        show_df(comp_tbl)
        long = comp_tbl.melt("Model", ["Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1", "Weighted F1"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", color_discrete_sequence=[SLATE, TEAL, NAVY, "#E0A830"])
        fig.update_yaxes(range=[0, 1])
        show_fig(style_fig(fig, 380, "Model comparison on the same hold-out set"), "t7_cmp")
        best = comp_tbl.sort_values("Macro F1", ascending=False).iloc[0]
        st.markdown(f"**Reading:** highest Macro F1 on this hold-out is **{best['Model']}** ({best['Macro F1']:.3f}). "
                    "This is one split on one small test set – differences of a few points may be noise, and no model is assumed best a priori.")
        st.download_button("⬇ Download model-performance table (CSV)", csv_bytes(comp_tbl), "model_performance.csv", "text/csv")
        pick = st.selectbox("Inspect model", list(MR["models"].keys()), key="t7_pick")
        r = MR["models"][pick]
        st.caption(r.get("note", ""))
        a, b = st.columns(2)
        with a:
            cm = r["cm"]
            fig = px.imshow(cm, x=SENT_ORDER, y=SENT_ORDER, text_auto=True, color_continuous_scale=["#FFFFFF", NAVY],
                            labels=dict(x="Predicted", y="Actual", color="Tweets"))
            show_fig(style_fig(fig, 340, "Confusion matrix"), "t7_cm")
        with b:
            st.markdown("**Classification report**")
            show_df(r["report"].round(3))
        if "history" in r:
            h = r["history"]
            ep = list(range(1, len(h["loss"]) + 1))
            a, b = st.columns(2)
            with a:
                fig = go.Figure()
                fig.add_scatter(x=ep, y=h["loss"], name="Training loss", mode="lines+markers")
                fig.add_scatter(x=ep, y=h["val_loss"], name="Validation loss", mode="lines+markers")
                show_fig(style_fig(fig, 320, "Training vs validation loss"), "t7_loss")
            with b:
                fig = go.Figure()
                fig.add_scatter(x=ep, y=h["accuracy"], name="Training accuracy", mode="lines+markers")
                fig.add_scatter(x=ep, y=h["val_accuracy"], name="Validation accuracy", mode="lines+markers")
                show_fig(style_fig(fig, 320, "Training vs validation accuracy"), "t7_acc")
            gap = h["accuracy"][-1] - h["val_accuracy"][-1]
            st.markdown(f"Final train–validation accuracy gap: **{gap:+.3f}** ({'possible overfitting' if gap > 0.08 else 'no strong overfitting signal'}). "
                        f"Regularisation used: dropout 0.2/0.3, L2 on the dense layer, early stopping on validation loss; {len(ep)} epoch(s) run.")
        if "top_features" in r:
            with st.expander("Most influential words per class (Logistic Regression coefficients)"):
                show_df(pd.DataFrame(dict([(k, v) for k, v in r["top_features"].items()])))
        st.markdown("##### Misclassification examples")
        test_df = df.loc[MR["test_index"]]
        wrong = test_df.assign(predicted=r["pred"])
        wrong = wrong[wrong["predicted"] != wrong["airline_sentiment"]]
        show_df(wrong[["airline_sentiment", "predicted", "text"]].rename(columns={"airline_sentiment": "actual"}).head(20))
        st.markdown("##### Try a tweet")
        preds = st.session_state.get("predictors", {})
        txt = st.text_input("Type a tweet to score with the trained models", "Thanks for the great service, crew were wonderful!", key="t7_try")
        if txt and preds:
            out_rows = []
            for m, fn in preds.items():
                try:
                    out_rows.append({"Model": m, **{k: round(float(v), 3) for k, v in fn(txt).items()}})
                except Exception as exc:
                    out_rows.append({"Model": m, "error": str(exc)})
            show_df(pd.DataFrame(out_rows))
    with st.expander("Neural-network concepts used (activation, regularisation, LSTM)"):
        st.markdown("""- **Embedding layer** maps each word index to a 64-dim learned vector. **LSTM** processes words in order and keeps a memory cell controlled by sigmoid gates (input/forget/output) and tanh – useful for negation and word order. 
- **ReLU** in the dense layer adds non-linearity; **softmax** converts outputs to class probabilities.
- **Overfitting** is tackled with dropout, L2 weight penalty and early stopping on validation loss. **Class weights** counter class imbalance (negative tweets dominate).
- **GRU** is a lighter gated alternative to LSTM; it is described here but not trained, to keep the app within Community Cloud limits.""")

# ============================================================================
# TAB 8 - REPUTATION AND BUSINESS SCENARIOS
# ============================================================================
with tabs[7]:
    st.subheader("Reputation monitoring and business scenarios")
    st.markdown("#### A · Observed sentiment-based reputation monitoring (measured from tweets)")
    st.caption("Reputation Monitoring Score (RMS) = 50 + (positive% − negative%) ÷ 2, per day. 50 = balanced, 100 = all positive, 0 = all negative. "
               "It is a transparent descriptive index, not an official brand metric.")
    n_days_all = int(df.loc[df["is_observed"], "date"].nunique())
    n_months_obs = int(df.loc[df["is_observed"], "tweet_month"].nunique())
    c = st.columns(3)
    with c[0]: kpi_card("Distinct observed days", f"{n_days_all}", "Whole dataset, observed tweets", NAVY)
    with c[1]: kpi_card("Distinct observed months", f"{n_months_obs}", "Short coverage limits forecasting", NAVY)
    with c[2]: kpi_card("Mean RMS (selection)", f"{daily['rms'].mean():.1f}" if len(daily) else "n/a", "0–100 scale")
    if len(daily) >= 2:
        sm = daily.copy()
        sm["rms_7d"] = sm["rms"].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=sm["date"], y=sm["rms"], name="Daily RMS", mode="lines+markers", line=dict(color=TEAL))
        fig.add_scatter(x=sm["date"], y=sm["rms_7d"], name="3-day moving average", line=dict(color=NAVY, dash="dash"))
        fig.add_hline(y=50, line_dash="dot", line_color=SLATE)
        FIGS["rms_trend"] = fig
        show_fig(style_fig(fig, 340, "Reputation Monitoring Score"), "t8_rms")
        a, b, c3 = st.columns(3)
        k_sig = a.slider("Alert sensitivity (σ above mean negative %)", 0.5, 3.0, 1.0, 0.25, key="t8_sig")
        minv = b.slider("Minimum tweets per day", 1, 100, 10, key="t8_minv")
        base_d = daily[daily["total"] >= minv]
        if len(base_d) >= 3:
            thr = base_d["negative_pct"].mean() + k_sig * base_d["negative_pct"].std(ddof=0)
            alerts = base_d[base_d["negative_pct"] > thr]
            c3.metric("Alert threshold (negative %)", f"{thr:.1f}%")
            if alerts.empty:
                st.success("No days breach the alert threshold under the current settings.")
            else:
                st.warning(f"{len(alerts)} potential reputation alert day(s) above {thr:.1f}% negative.")
                show_df(alerts[["date", "total", "negative_pct", "rms"]].rename(columns={"total": "Tweets", "negative_pct": "Negative %", "rms": "RMS"}).round(1))
        else:
            st.info("Not enough days with sufficient volume to set alert thresholds.")
    if not CT.empty:
        st.markdown("**Complaint severity indicators** (frequency measured; severity = management input from Tab 5)")
        show_df(PT[["Rank", "Complaint category", "Frequency", "Severity assumption (1-5, management input)", "Priority score (0-100)"]].head(6).round(1))
    st.markdown("##### Observed-tweet forecast eligibility")
    if n_days_all < 30:
        st.info(f"Forecasting is **disabled** for observed tweets: only {n_days_all} distinct days ({n_months_obs} month) are available, below the "
                "defensible minimum of 30 daily points. The tweets do not provide long-term history, so no forecast is manufactured.")
    else:
        obs_daily = daily_sentiment(df[df["is_observed"] & df["airline"].isin(st.session_state["f_airlines"])])
        hz = st.slider("Forecast horizon (days)", 3, 14, 7, key="t8_hz_obs")
        if len(obs_daily) >= 30:
            mf, lo, hi, slope, r2 = linear_forecast(obs_daily["negative_pct"].values, hz)
            fd = pd.date_range(obs_daily["date"].max() + pd.Timedelta(days=1), periods=hz)
            fig = go.Figure()
            fig.add_scatter(x=obs_daily["date"], y=obs_daily["negative_pct"], name="Observed")
            fig.add_scatter(x=fd, y=mf, name="Linear-trend forecast")
            fig.add_scatter(x=list(fd) + list(fd[::-1]), y=list(hi) + list(lo[::-1]), fill="toself", name="95% interval", line=dict(width=0), opacity=0.25)
            show_fig(style_fig(fig, 340, "Linear-trend forecast of daily negative %"), "t8_fc_obs")
            st.caption(f"Assumptions: straight-line trend, independent errors, no seasonality. Slope {slope:+.3f} pts/day, R² {r2:.2f}.")

    st.markdown("---")
    st.markdown("#### B · HYPOTHETICAL SCENARIO DEMONSTRATION (generated data – not real American Airlines results)")
    st.warning("The scenario_* fields are generated demonstration values stored once per airline-month and repeated on every tweet row. "
               "They are aggregated here at the airline × month level (never summed across tweets) and do not prove real revenue, churn or causality.")
    sc_vars = {k: v for k, v in SCENARIO_VARS.items() if k in df}
    if not sc_vars or "tweet_month" not in df:
        st.info("No scenario columns found in the dataset.")
    else:
        sa = st.multiselect("Airlines (scenario view)", airlines_all, default=st.session_state["f_airlines"] or [focus], key="t8_sc_air")
        var = st.selectbox("Scenario variable", list(sc_vars), format_func=lambda k: sc_vars[k], key="t8_var")
        sub_s = df[df["airline"].isin(sa) & df["tweet_month"].notna()]
        if sub_s.empty:
            st.info("No scenario rows for the selected airlines.")
        else:
            monthly = sub_s.groupby(["airline", "tweet_month"])[list(sc_vars)].mean().reset_index()
            var_chk = sub_s.groupby(["airline", "tweet_month"])[var].nunique().max()
            n_months_sc = monthly.groupby("airline")["tweet_month"].nunique()
            st.caption(f"Aggregation check: max distinct values of the selected metric within an airline-month = {int(var_chk) if pd.notna(var_chk) else 'n/a'} "
                       "(1 means the value is repeated per tweet, so the mean equals the stored value). "
                       f"Months available: {', '.join(f'{a_}: {n}' for a_, n in n_months_sc.items())}.")
            fig = px.line(monthly, x="tweet_month", y=var, color="airline", markers=True, labels={var: sc_vars[var], "tweet_month": "Month"})
            FIGS["scenario_metric"] = fig
            show_fig(style_fig(fig, 360, f"SCENARIO (hypothetical): {sc_vars[var]}"), "t8_sc")
            show_df(monthly[["airline", "tweet_month", var]].round(2))
            st.download_button("⬇ Download scenario monthly table (CSV)", csv_bytes(monthly), "scenario_monthly.csv", "text/csv")
            xvar = "scenario_reputation_index_0_100"
            if xvar in monthly and var != xvar and len(monthly) >= 8:
                z = monthly[[xvar, var]].dropna()
                if len(z) >= 8 and z[xvar].std() > 0:
                    slope, icpt = np.polyfit(z[xvar], z[var], 1)
                    corr = z[xvar].corr(z[var])
                    st.markdown("**What-if (association only, not causal):** linear association across scenario airline-months")
                    d_idx = st.slider("Assumed change in scenario reputation index (points)", -20, 20, -5, key="t8_whatif")
                    st.markdown(f"Fitted slope = {slope:,.2f} per index point (r = {corr:.2f}, n = {len(z)}). A {d_idx:+d}-point change implies "
                                f"**{slope * d_idx:+,.1f}** change in *{sc_vars[var]}* under this hypothetical linear relationship.")
            st.markdown("##### Scenario forecast eligibility")
            one = st.selectbox("Airline for forecast", sa, key="t8_fc_air")
            ser = monthly[monthly["airline"] == one].sort_values("tweet_month")[var].dropna().values
            if len(ser) < 12:
                st.info(f"Forecast **disabled**: {len(ser)} monthly point(s) for {one}; a linear-trend forecast requires at least 12. "
                        "No forecast is manufactured from insufficient history.")
            else:
                hz = st.slider("Forecast horizon (months)", 1, 6, 3, key="t8_hz_sc")
                mf, lo, hi, slope, r2 = linear_forecast(ser, hz)
                months = sorted(monthly[monthly["airline"] == one]["tweet_month"].unique())
                fm = pd.period_range(pd.Period(months[-1], "M") + 1, periods=hz, freq="M").astype(str)
                fig = go.Figure()
                fig.add_scatter(x=months, y=ser, name="Scenario history")
                fig.add_scatter(x=fm, y=mf, name="Hypothetical forecast")
                fig.add_scatter(x=list(fm) + list(fm[::-1]), y=list(hi) + list(lo[::-1]), fill="toself", line=dict(width=0), opacity=0.25, name="95% interval")
                show_fig(style_fig(fig, 340, "HYPOTHETICAL scenario forecast (linear trend)"), "t8_fc_sc")
                st.caption(f"Assumptions: linear trend on generated data, slope {slope:+,.2f}/month, R² {r2:.2f}; no seasonality; illustrative only.")

# ============================================================================
# TAB 9 - MANAGERIAL INSIGHTS AND SIX SIGMA
# ============================================================================
neg_sub = fdf[fdf["airline_sentiment"] == "negative"]
daily_ok = daily[daily["total"] >= 10] if len(daily) else daily
base_neg_mean = float(daily_ok["negative_pct"].mean()) if len(daily_ok) else float("nan")
base_neg_sd = float(daily_ok["negative_pct"].std(ddof=0)) if len(daily_ok) else float("nan")
pos_terms = terms_for(fdf, "positive", 1).head(5)["term"].tolist()
neg_terms = terms_for(fdf, "negative", 1).head(5)["term"].tolist()
MR_ = st.session_state.get("model_results")
model_line = "No model trained yet"
if MR_ and MR_["models"]:
    bm = max(MR_["models"].items(), key=lambda kv: kv[1]["metrics"]["Macro F1"])
    model_line = f"{bm[0]} achieved the highest Macro F1 ({bm[1]['metrics']['Macro F1']:.3f}) on the American hold-out"

reco_rows = []
if not PT.empty:
    for _, r in PT.head(5).iterrows():
        pr = "High" if r["Priority score (0-100)"] >= 66 else "Medium" if r["Priority score (0-100)"] >= 33 else "Low"
        reco_rows.append({"Analytical finding": f"{r['Complaint category']}: {int(r['Frequency']):,} tweets, {r['Share of negative posts (%)']:.1f}% of negative posts",
                          "Business implication": f"Recurring pain point (priority score {r['Priority score (0-100)']:.0f}) that drives negative brand talk and potential churn",
                          "Proposed action": r["Proposed managerial action"], "Priority": pr, "KPI": r["KPI to monitor"]})
RECO = pd.DataFrame(reco_rows)

with tabs[8]:
    st.subheader("Managerial insights and Six Sigma DMAIC")
    ins = [("Top negative-sentiment drivers", ", ".join(f"{a_} ({b_:.1f}%)" for a_, b_ in zip(CT["Complaint category"].head(3), CT["Share of negative posts (%)"].head(3))) if not CT.empty else "n/a"),
           ("Major positive themes (frequent words)", ", ".join(pos_terms) or "n/a"),
           ("Frequent negative words", ", ".join(neg_terms) or "n/a"),
           ("Most frequent complaint", f"{dom_complaint}"),
           ("Potential reputation-risk areas", ", ".join(PT["Complaint category"].head(3)) if not PT.empty else "n/a"),
           ("Sentiment balance (pos% − neg%)", f"{pct['positive'] - pct['negative']:+.1f} points"),
           ("Relevant model finding", model_line)]
    INS = pd.DataFrame(ins, columns=["Insight", "Calculated result"])
    show_df(INS)
    st.markdown("##### Recommendation matrix")
    if RECO.empty:
        st.info("Recommendations appear when negative tweets are present in the selection.")
    else:
        show_df(RECO)
        st.caption("Priorities combine measured complaint frequency with the severity assumptions set in Tab 5. Actions are proposals; none has been implemented.")
        st.download_button("⬇ Download managerial recommendations (CSV)", csv_bytes(RECO), "managerial_recommendations.csv", "text/csv")
    st.markdown("##### DMAIC improvement framework")
    cA = st.columns(5)
    dm = [("DEFINE", f"Problem: negative public sentiment and recurring service complaints damage brand reputation. Scope: {', '.join(st.session_state['f_airlines']) or 'selected airlines'}."),
          ("MEASURE", f"Negative sentiment {pct['negative']:.1f}%, negative rate {'n/a' if np.isnan(neg_rate) else f'{neg_rate:.1f}%'}; {CT.shape[0]} complaint categories; top: {dom_complaint}."),
          ("ANALYZE", f"Pareto top categories: {', '.join(CT['Complaint category'].head(3)) if not CT.empty else 'n/a'}. Frequent negative words: {', '.join(neg_terms[:4]) or 'n/a'}."),
          ("IMPROVE", "Proposed (not yet implemented): service-recovery training, proactive disruption alerts, baggage tracking, faster social care."),
          ("CONTROL", f"Monitor daily negative % and RMS; escalate above thresholds below. Baseline mean {'n/a' if np.isnan(base_neg_mean) else f'{base_neg_mean:.1f}%'}.")]
    for col, (t_, b_) in zip(cA, dm):
        with col:
            st.markdown(f'<div class="dmaic"><div class="t">{esc(t_)}</div><div class="b">{esc(b_)}</div></div>', unsafe_allow_html=True)
    st.markdown("##### Control plan: monitoring indicators and proposed escalation thresholds")
    if np.isnan(base_neg_mean):
        st.info("Not enough days with ≥10 tweets to compute control thresholds.")
        CTRL = pd.DataFrame()
    else:
        k1 = st.slider("Amber threshold (σ above baseline)", 0.5, 2.0, 1.0, 0.25, key="t9_k1")
        k2 = st.slider("Red threshold (σ above baseline)", 1.0, 3.5, 2.0, 0.25, key="t9_k2")
        CTRL = pd.DataFrame({"Indicator": ["Daily negative sentiment %", "Reputation Monitoring Score (RMS)", f"Top-complaint share ({dom_complaint})"],
                             "Baseline (selection)": [round(base_neg_mean, 1), round(float(daily_ok['rms'].mean()), 1), round(float(CT.iloc[0]['Share of negative posts (%)']), 1) if not CT.empty else np.nan],
                             "Amber trigger": [round(base_neg_mean + k1 * base_neg_sd, 1), round(float(daily_ok['rms'].mean() - k1 * daily_ok['rms'].std(ddof=0)), 1), "—"],
                             "Red trigger": [round(base_neg_mean + k2 * base_neg_sd, 1), round(float(daily_ok['rms'].mean() - k2 * daily_ok['rms'].std(ddof=0)), 1), "—"],
                             "Proposed owner": ["Social-media care lead", "Brand / CX director", "Operations + CX"]})
        show_df(CTRL.astype(str))
        st.caption("Thresholds are statistical proposals derived from the short observed window; recalibrate with longer history before operational use.")

# ============================================================================
# TAB 10 - REPORTS, EXPORTS AND METHODOLOGY
# ============================================================================


def build_summary() -> list:
    L = [f"Scope: {', '.join(st.session_state['f_airlines']) or 'n/a'}; mode: {st.session_state['f_mode']}; {n_sel:,} tweets "
         f"({n_obs_sel:,} observed) between {dates_sel[0]} and {dates_sel[1]}.",
         f"Sentiment: {pct['positive']:.1f}% positive, {pct['neutral']:.1f}% neutral, {pct['negative']:.1f}% negative.",
         f"Dominant complaint: {dom_complaint}; {CT.shape[0]} complaint categories identified."]
    if not PT.empty:
        L.append("Highest-priority risks (frequency × management severity): " + ", ".join(PT["Complaint category"].head(3)) + ".")
    L.append(f"Model finding: {model_line}.")
    L.append(f"Observed tweets cover {obs_days} distinct day(s) in this selection – short-term evidence only; scenario business metrics are hypothetical.")
    return L


with tabs[9]:
    st.subheader("Reports, exports and methodology")
    SUM = build_summary()
    st.markdown("#### Executive analytical summary")
    for s_ in SUM:
        st.markdown(f"- {s_}")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Objectives")
        st.markdown("""**Business:** understand sentiment toward American Airlines, find recurring complaints, detect reputation risks, prioritise actions.  
**Technical:** clean text, classify sentiment (baseline, VADER, TF-IDF+LR, LSTM), build word networks, model topics, evaluate without leakage.""")
        st.markdown("#### Data source")
        st.markdown(f"`Airline_Tweets.csv` – {info['raw_rows']:,} rows × {info['raw_cols']} columns; original observed tweets plus generated demonstration records and scenario business indicators, separated by provenance flags.")
        st.markdown("#### Preprocessing & EDA")
        st.markdown("URL/mention removal, lower-casing, tokenisation, stop-word removal, provenance filtering; EDA via distributions, daily trends, Pareto, n-grams, co-occurrence networks, LDA/NMF topics.")
    with b:
        st.markdown("#### Models & evaluation")
        st.markdown("Majority baseline, VADER, TF-IDF + Logistic Regression, Embedding + LSTM. Metrics: accuracy, macro precision/recall/F1, weighted F1, confusion matrix; fixed American hold-out of original tweets.")
        st.markdown("#### Limitations")
        st.markdown("""- Twitter sample is self-selected and complaint-heavy; not representative of all passengers.
- Observed tweets span a short window; no long-term trend or forecast is supported.
- Class imbalance and a small test set make metrics uncertain; near-duplicates may remain.
- Scenario business fields and generated tweets are hypothetical; no causal claims.
- Severity scores are management assumptions.""")
    st.markdown("#### How this application meets the five evaluation areas")
    show_df(pd.DataFrame({"Criterion (5 marks each)": ["Business & technical objectives", "Data requirement analysis", "Exploratory data analysis", "Data modeling", "Results, insights & recommendations"],
                          "Where addressed": ["Tabs 1, 10", "Tab 2", "Tabs 3, 4, 5, 6", "Tabs 6, 7", "Tabs 5, 8, 9, 10"]}))
    st.markdown("#### Methodology reference")
    st.markdown("Load → validate → parse dates & provenance → clean text → filter → aggregate (sentiment, complaints) → n-grams / networks / topics → split original American tweets (stratified, seeded) → fit on train only → evaluate on hold-out → translate to Pareto-based priorities & DMAIC controls.")
    st.markdown("#### References")
    st.markdown("""- Hutto & Gilbert (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation*.
- Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. *JMLR*.
- Lee & Seung (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*.
- Pyzdek & Keller. *The Six Sigma Handbook* (DMAIC).
- Course material: AI-Driven Business Analytics, IIM Ranchi (Prof. Sasadhar Bera).""")
    st.markdown("#### Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Cleaned dataset (CSV)", cleaned_export(str(DATA_FILE.stat().st_mtime), df), "cleaned_dataset.csv", "text/csv")
    exp_cols = [c for c in fdf.columns if c != "text_vader"]
    d2.download_button("⬇ Filtered dataset (CSV)", csv_bytes(fdf[exp_cols]), "filtered_dataset.csv", "text/csv")
    d3.download_button("⬇ Sentiment summary (CSV)", csv_bytes(S), "sentiment_summary.csv", "text/csv")
    d4, d5, d6 = st.columns(3)
    d4.download_button("⬇ Complaint analysis (CSV)", csv_bytes(PT if not PT.empty else CT), "complaint_analysis.csv", "text/csv")
    d5.download_button("⬇ Recommendations (CSV)", csv_bytes(RECO), "managerial_recommendations.csv", "text/csv", disabled=RECO.empty)
    d6.download_button("⬇ Topic output (CSV)", csv_bytes(TOPIC_EXPORT) if TOPIC_EXPORT is not None else b"", "topic_modeling_output.csv",
                       "text/csv", disabled=TOPIC_EXPORT is None, help=None if TOPIC_EXPORT is not None else "Run topic extraction in Tab 6 first")
    d7, d8, d9 = st.columns(3)
    if MR_ and MR_["models"]:
        mt = pd.DataFrame([{"Model": m, **r["metrics"]} for m, r in MR_["models"].items()])
        d7.download_button("⬇ Model evaluation (CSV)", csv_bytes(mt), "model_evaluation.csv", "text/csv")
    else:
        d7.download_button("⬇ Model evaluation (CSV)", b"", "model_evaluation.csv", "text/csv", disabled=True, help="Train models in Tab 7 first")
    body = "".join(f"<li>{esc(s_)}</li>" for s_ in SUM)
    page = ("<html><head><meta charset='utf-8'><title>Executive summary</title></head><body style='font-family:Segoe UI,Arial;max-width:800px;margin:30px auto'>"
            "<h2>American Airlines – AI-Driven Brand Reputation Intelligence</h2><h3>Executive analytical summary</h3>"
            f"<ul>{body}</ul><p><i>Generated records and scenario metrics are hypothetical; see Limitations.</i></p></body></html>")
    d8.download_button("⬇ Executive summary (HTML)", page.encode("utf-8"), "executive_summary.html", "text/html")
    d9.download_button("⬇ Executive summary (TXT)", "\n".join(SUM).encode("utf-8"), "executive_summary.txt", "text/plain")
    figs_avail = {k: v for k, v in FIGS.items()}
    if figs_avail:
        fk = st.selectbox("Figure export (interactive HTML)", list(figs_avail), key="t10_fig")
        st.download_button("⬇ Download selected figure (HTML)", figs_avail[fk].to_html(include_plotlyjs="cdn").encode("utf-8"), f"{fk}.html", "text/html")
