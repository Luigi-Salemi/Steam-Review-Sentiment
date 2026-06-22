"""
Steam Review Sentiment — interactive dashboard
==============================================
Traditional NLP vs. Transformers on real Steam reviews (2024-2026) collected via
the official Steam Reviews API. Sidebar nav sections, Lucide icons, dark theme.
The live demo runs the fine-tuned model in-process (pre-warmed in the background).

Run:  streamlit run app.py
"""

import os
import threading
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import results as R

st.set_page_config(page_title="Steam Review Sentiment — ADS-509",
                   page_icon=":material/sports_esports:", layout="wide",
                   initial_sidebar_state="expanded")
pio.templates.default = "plotly_dark"

# ── palette ───────────────────────────────────────────────────────────
BG, PANEL, ELEV = "#1C2127", "#252A31", "#2F343C"
BORDER, BORDER2 = "#383E47", "#404854"
TEXT, SUBTLE, MUTE = "#F6F7F9", "#ABB3BF", "#738091"
PRIMARY, PRIMARY2 = "#688ae8", "#9db4f0"
POS, NEG, ACCENT = "#40bfa9", "#e07f9d", "#dfb52c"
GAIN = round((R.DISTILBERT_METRICS["F1 (weighted)"] - R.LR_METRICS["F1 (weighted)"]) * 100, 1)
HAS_MODEL = os.path.isdir(os.path.join(os.path.dirname(__file__), "model"))

# ── Lucide icons (inline SVG, MIT licensed) ───────────────────────────
_ICONS = {
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "flask": '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    "gamepad": '<line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/><line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.544-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.152A4 4 0 0 0 17.32 5z"/>',
    "thumbsUp": '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
    "thumbsDown": '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "type": '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>',
    "layers": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "split": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
}


def icon(name, size=20, color=PRIMARY, sw=2.0, mr=8):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' "
            f"fill='none' stroke='{color}' stroke-width='{sw}' stroke-linecap='round' stroke-linejoin='round' "
            f"style='vertical-align:middle;margin-right:{mr}px'>{_ICONS[name]}</svg>")


# ── styling ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background:{BG}; }}
  .block-container {{ padding-top:1.6rem; max-width:1200px; }}
  h1,h2,h3,h4 {{ color:{TEXT}; }}
  .hero {{ background:linear-gradient(135deg,{PANEL} 0%,{ELEV} 100%); border:1px solid {BORDER};
           border-radius:14px; padding:20px 24px; }}
  .mcard {{ position:relative; background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {PRIMARY};
            border-radius:12px; padding:14px 16px; height:116px; display:flex; flex-direction:column; justify-content:center; }}
  .mcard .ic {{ position:absolute; top:13px; right:15px; opacity:.85; }}
  .mcard .v {{ font-size:25px; font-weight:750; line-height:1.1; color:{TEXT}; }}
  .mcard .l {{ font-size:11.5px; color:{SUBTLE}; text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }}
  .mcard .d {{ position:absolute; bottom:11px; left:16px; font-size:12px; font-weight:600; }}
  .sec {{ display:flex; align-items:center; gap:9px; margin:6px 0 4px; }}
  .sec .t {{ font-size:18px; font-weight:700; color:{TEXT}; }}
  .panel {{ border-radius:10px; padding:13px 16px; display:flex; gap:11px; align-items:flex-start; margin:8px 0;
            line-height:1.5; font-size:14px; color:{TEXT}; }}
  .pill {{ display:inline-flex; align-items:center; background:{PRIMARY}1f; color:{PRIMARY2};
           border:1px solid {PRIMARY}55; border-radius:999px; padding:4px 12px; margin:3px 6px 3px 0;
           font-size:13px; font-weight:600; }}
  .pill.hot {{ background:{ACCENT}1f; color:{ACCENT}; border-color:{ACCENT}66; }}
  .badge {{ display:inline-flex; align-items:center; border-radius:999px; padding:3px 11px; font-size:12.5px; font-weight:700; }}
  .step {{ border:1px solid {BORDER}; border-left:4px solid {PRIMARY}; background:{PANEL};
           border-radius:0 10px 10px 0; padding:11px 15px; margin-bottom:9px; min-height:86px; color:{TEXT}; }}
  .note {{ color:{SUBTLE}; font-size:13px; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:6px; position:sticky; top:0; z-index:100; background:{BG}; padding:8px 0 0; }}
  .stTabs [data-baseweb="tab"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:9px 9px 0 0; padding:9px 20px; color:{SUBTLE}; }}
  .stTabs [aria-selected="true"] {{ background:{ELEV}; color:{TEXT}; border-bottom:2px solid {PRIMARY}; }}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────
def card(icon_name, value, label, accent=PRIMARY, delta=None):
    d = ""
    if delta:
        dc = POS if str(delta).startswith("+") else (NEG if str(delta).startswith("-") else SUBTLE)
        d = f"<div class='d' style='color:{dc}'>{delta}</div>"
    st.markdown(f"<div class='mcard' style='border-left-color:{accent}'>"
                f"<div class='ic'>{icon(icon_name,20,accent,2,0)}</div>"
                f"<div class='v'>{value}</div><div class='l'>{label}</div>{d}</div>", unsafe_allow_html=True)


def section(icon_name, title, accent=PRIMARY):
    st.markdown(f"<div class='sec'>{icon(icon_name,20,accent,2.2,0)}<span class='t'>{title}</span></div>",
                unsafe_allow_html=True)


def panel(icon_name, html, accent=PRIMARY):
    st.markdown(f"<div class='panel' style='background:{accent}14;border:1px solid {accent}40'>"
                f"<div style='flex:0 0 auto;margin-top:1px'>{icon(icon_name,18,accent,2,0)}</div><div>{html}</div></div>",
                unsafe_allow_html=True)


def badge(text, color=PRIMARY, icon_name=None):
    ic = icon(icon_name, 14, color, 2.4, 5) if icon_name else ""
    return f"<span class='badge' style='background:{color}1f;color:{color};border:1px solid {color}55'>{ic}{text}</span>"


def style(fig, h=340):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=SUBTLE), height=h, margin=dict(t=24, b=10, l=10, r=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBTLE)))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER2, color=SUBTLE)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER2, color=SUBTLE)
    return fig


def report_df(rep):
    df = pd.DataFrame(rep).T.rename(columns={"f1": "f1-score"})
    df["support"] = df["support"].astype(int)
    return df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:d}"})


def hbar(d, color=PRIMARY, h=340, xtitle="Reviews"):
    s = pd.Series(d).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color))
    fig.update_layout(xaxis_title=xtitle, yaxis_title="")
    return style(fig, h)


def verdict(is_pos, conf, caption):
    color = POS if is_pos else NEG
    st.markdown(f"<div style='background:{color}1f;border:1px solid {color}55;border-radius:10px;padding:14px 16px'>"
                f"<div style='display:flex;align-items:center;font-size:19px;font-weight:750;color:{color}'>"
                f"{icon('thumbsUp' if is_pos else 'thumbsDown',22,color,2.2,8)}{'Positive' if is_pos else 'Negative'}</div>"
                f"<div class='note' style='margin-top:7px'>Confidence <b style='color:{TEXT}'>{conf*100:.0f}%</b></div>"
                f"<div style='color:{MUTE};font-size:12px;margin-top:3px'>{caption}</div></div>", unsafe_allow_html=True)


# ── models (cached) + background warm-up ──────────────────────────────
@st.cache_resource(show_spinner="Training the classic model…")
def classic():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "reviews.csv"))
    df = df[df["label"].isin([0, 1])].dropna(subset=["text"])
    p = Pipeline([("tf", TfidfVectorizer(stop_words="english", max_features=5000)),
                  ("lr", LogisticRegression(max_iter=1000))])
    p.fit(df["text"].astype(str), df["label"].astype(int))
    return p, len(df)


@st.cache_resource(show_spinner=False)
def transformer():
    from transformers import pipeline
    local = os.path.join(os.path.dirname(__file__), "model")
    if os.path.isdir(local):
        return pipeline("sentiment-analysis", model=local, tokenizer=local), "our fine-tuned DistilBERT"
    return (pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
            "pretrained DistilBERT (fallback)")


@st.cache_resource(show_spinner=False)
def _warm_once():
    """Start loading the transformer in a daemon thread on first page load so it's
    likely ready by the time someone opens 'Try it Live' — never blocks/crashes the app."""
    def job():
        try:
            transformer()
        except Exception:
            pass
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        th = threading.Thread(target=job, daemon=True)
        try:
            add_script_run_ctx(th, get_script_run_ctx())
        except Exception:
            pass
        th.start()
    except Exception:
        pass
    return True


_warm_once()

# ── sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    _logo = os.path.join(os.path.dirname(__file__), "assets", "usd_logo.svg")
    if os.path.exists(_logo):
        st.markdown(f"<div style='text-align:center; margin:2px 0 10px'>{open(_logo).read()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;align-items:center;gap:8px;font-size:17px;font-weight:750;color:{TEXT}'>"
                f"{icon('gamepad',22,PRIMARY)}Steam Sentiment</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:15px;color:{TEXT};font-weight:700;line-height:1.35;margin-top:3px'>{R.PROJECT['course']}</div>"
        f"<div style='font-size:12.5px;color:{MUTE};margin-top:3px'>{R.PROJECT['school']}</div>",
        unsafe_allow_html=True)
    st.divider()
    nav = st.radio("Sections", ["Data & EDA", "Results", "Try it Live"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"{icon('users',16,SUBTLE,2,6)}<b style='color:{TEXT};font-size:15px'>Team</b>", unsafe_allow_html=True)
    for mm in R.PROJECT["team"]:
        st.markdown(f"<div style='font-size:17.5px; color:{TEXT}; font-weight:650; margin:7px 0 7px 4px'>{icon('check',14,POS,2,7)}{mm}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(badge("Fine-tuned model loaded", POS, "check") if HAS_MODEL
                else badge("Pretrained fallback", ACCENT, "info"), unsafe_allow_html=True)
    st.caption("Live-demo transformer (pre-warming in the background)")
    st.divider()
    with st.popover("Data provenance", icon=":material/fact_check:"):
        st.markdown(f"**[{getattr(R, 'DATASET_NAME', 'Dataset')}]({getattr(R, 'DATASET_URL', '#')})**")
        st.caption(R.PROVENANCE)


# ── hero + headline metrics ───────────────────────────────────────────
st.markdown(f"<div style='display:flex;align-items:center;gap:11px;margin-top:2px'>"
            f"{icon('gamepad',28,PRIMARY,2.2,0)}<h1 style='margin:0;font-size:28px'>{R.PROJECT['title']}</h1></div>"
            f"<div class='note' style='margin-top:8px;font-size:14px;max-width:1020px'>{R.PROJECT['objective']}</div>",
            unsafe_allow_html=True)
st.write("")
m = st.columns(4)
with m[0]:
    card("database", f"{R.RAW_REVIEW_COUNT:,}", "Real reviews", PRIMARY)
with m[1]:
    card("gamepad", f"{R.N_GAMES}", "Games", PRIMARY)
with m[2]:
    card("trophy", f"{R.DISTILBERT_METRICS['F1 (weighted)']*100:.1f}%", "DistilBERT F1",
         ACCENT, delta=f"+{GAIN} pts F1 vs baseline")
with m[3]:
    card("calendar", "2024–2026", "Recent reviews", PRIMARY)

st.write("")

# ====================================================================== DATA & EDA
if nav == "Data & EDA":
    st.caption(f"{R.RAW_REVIEW_COUNT:,} recent reviews across {R.N_GAMES} games, via the official Steam Reviews API. "
               f"Labels are each review's recommend / not-recommend ({R.LABEL_SOURCE}) — Steam has no star ratings.")
    d = st.columns(3)
    with d[0]:
        card("split", f"{R.TRAIN_SIZE:,} / {R.TEST_SIZE:,}", "Train / test")
    with d[1]:
        card("layers", f"{R.UNIQUE_REVIEWS:,}", "Unique reviews")
    with d[2]:
        card("list", f"{R.DUPLICATE_REVIEWS:,}", "Duplicates")

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        section("chart", "Top 20 games by reviews")
        st.plotly_chart(hbar(R.GAME_DIST, PRIMARY, 470), width="stretch")
    with c2:
        section("scale", "Class balance (50 / 50)")
        names = list(R.LABEL_BALANCE.keys())
        fig = go.Figure(go.Bar(x=names, y=list(R.LABEL_BALANCE.values()),
                               marker_color=[POS if "Positive" in n else NEG for n in names],
                               text=list(R.LABEL_BALANCE.values()), textposition="outside"))
        fig.update_traces(textfont_color=TEXT)
        fig.update_layout(yaxis_title="Reviews", xaxis_title="")
        st.plotly_chart(style(fig, 470), width="stretch")

    st.divider()
    section("list", "Preprocessing pipeline")
    steps = [("Collect", "Steam Reviews API, 247 games, 2024–2026"),
             ("Label", "recommend = positive (1), not-recommend = negative (0)"),
             ("Clean & dedupe", "drop < 5 chars, de-duplicate by review id"),
             ("Balance & split", f"{R.BALANCED_TOTAL:,} balanced, stratified 80/20")]
    sc = st.columns(4)
    for i, (t, desc) in enumerate(steps):
        with sc[i]:
            st.markdown(f"<div class='step'><b>{i+1}. {t}</b><br><span class='note'>{desc}</span></div>",
                        unsafe_allow_html=True)
    with st.expander("How exactly was the data scraped?"):
        st.markdown("See **DATA_COLLECTION.md**. In short: SteamSpy provides the game pool "
                    "(top-played + most-owned + modern seed titles incl. EA SPORTS FC); each game's recent "
                    "positive & negative reviews are pulled from `store.steampowered.com/appreviews/<appid>`, "
                    "filtered to 2024-01-01+, de-duplicated, then balanced.")

    st.divider()
    e1, e2 = st.columns(2)
    with e1:
        section("type", "Avg review length by sentiment")
        sent, vals = list(R.AVG_REVIEW_LENGTH), list(R.AVG_REVIEW_LENGTH.values())
        fig = go.Figure(go.Bar(x=sent, y=vals, text=vals, textposition="outside",
                               marker_color=[NEG if "Negative" in s else POS for s in sent]))
        fig.update_traces(textfont_color=TEXT)
        fig.update_layout(yaxis_title="Words", xaxis_title="")
        st.plotly_chart(style(fig, 320), width="stretch")
    with e2:
        section("chart", "Top 15 words (stopwords removed)")
        st.plotly_chart(hbar(dict(R.TOP_WORDS[:15]), PRIMARY, 320, "Frequency"), width="stretch")

    neg, pos = R.AVG_REVIEW_LENGTH.get("Negative (0)"), R.AVG_REVIEW_LENGTH.get("Positive (1)")
    if neg and pos:
        panel("info", f"Negative reviews average <b>{neg}</b> words vs. <b>{pos}</b> for positive — "
                      "unhappy players write longer, more detailed complaints.")

    section("sparkles", "Word cloud")
    wc = os.path.join(os.path.dirname(__file__), "assets", "wordcloud.png")
    if os.path.exists(wc):
        st.image(wc, width="stretch")
    st.markdown("<b style='color:%s'>Top TF-IDF terms</b>&nbsp; " % TEXT
                + "".join(f"<span class='pill {'hot' if t in R.TFIDF_DISCRIMINATIVE else ''}'>{t}</span>"
                          for t in R.TFIDF_TOP_FEATURES), unsafe_allow_html=True)

# ====================================================================== RESULTS
elif nav == "Results":
    section("cpu", "Model comparison")
    st.caption("Both models evaluated on the same held-out test set. F1 (weighted) is the headline metric.")
    comp = pd.DataFrame(R.MODEL_COMPARISON)
    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=comp["Model"], y=comp["Accuracy"], marker_color=PRIMARY,
                text=[f"{v*100:.1f}%" for v in comp["Accuracy"]], textposition="outside")
    fig.add_bar(name="F1", x=comp["Model"], y=comp["F1 (weighted)"], marker_color=ACCENT,
                text=[f"{v*100:.1f}%" for v in comp["F1 (weighted)"]], textposition="outside")
    fig.update_traces(textfont_color=TEXT)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.0], tickformat=".0%"), yaxis_title="Score")
    st.plotly_chart(style(fig, 360), width="stretch")
    panel("trophy", f"Fine-tuned <b>DistilBERT (F1 {R.DISTILBERT_METRICS['F1 (weighted)']*100:.1f}%)</b> beats the "
                    f"<b>TF-IDF + Logistic Regression baseline (F1 {R.LR_METRICS['F1 (weighted)']*100:.1f}%)</b> by "
                    f"<b>~{GAIN} F1 points</b> — contextual embeddings catch the sarcasm and slang in game reviews.", ACCENT)

    st.divider()
    L, Rr = st.columns(2)
    with L:
        section("scale", "TF-IDF + Logistic Regression")
        st.markdown(badge("Baseline", PRIMARY, "scale"), unsafe_allow_html=True)
        cc = st.columns(2)
        with cc[0]:
            card("gauge", f"{R.LR_METRICS['F1 (weighted)']*100:.1f}%", "F1 (weighted)", ACCENT)
        with cc[1]:
            card("gauge", f"{R.LR_METRICS['Accuracy']*100:.1f}%", "Accuracy")
        st.dataframe(report_df(R.LR_REPORT), width="stretch")
    with Rr:
        section("cpu", "Fine-tuned DistilBERT")
        st.markdown(badge("Best model", ACCENT, "trophy"), unsafe_allow_html=True)
        cc = st.columns(2)
        with cc[0]:
            card("gauge", f"{R.DISTILBERT_METRICS['F1 (weighted)']*100:.1f}%", "F1 (weighted)", ACCENT)
        with cc[1]:
            card("gauge", f"{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%", "Accuracy")
        st.dataframe(report_df(R.DISTILBERT_REPORT), width="stretch")

    st.write("")
    cm1, cm2 = st.columns(2)
    with cm1:
        section("layers", "Confusion matrix")
        lab = ["Negative", "Positive"]
        cm = go.Figure(go.Heatmap(z=R.DISTILBERT_CONFUSION, x=lab, y=lab, text=R.DISTILBERT_CONFUSION,
                                  texttemplate="%{text}", textfont=dict(size=18, color=TEXT),
                                  colorscale=[[0, PANEL], [1, PRIMARY]], showscale=False, xgap=4, ygap=4))
        cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(cm, 300), width="stretch")
    with cm2:
        section("check", "On the held-out test set")
        cc = st.columns(2)
        with cc[0]:
            card("check", f"{R.DISTILBERT_CORRECT:,}", f"Correct / {R.TEST_SIZE:,}")
        with cc[1]:
            card("alert", f"{R.DISTILBERT_MISCLASSIFIED}", "Misclassified", NEG)
        with st.expander("DistilBERT fine-tuning config"):
            st.table(pd.DataFrame({"Setting": list(R.DISTILBERT_CONFIG), "Value": list(R.DISTILBERT_CONFIG.values())}))
        with st.expander("Takeaways & limitations"):
            st.markdown(f"{icon('check',15,POS,2,5)}**Takeaways**", unsafe_allow_html=True)
            for t in R.KEY_TAKEAWAYS:
                st.markdown(f"- {t}")
            st.markdown(f"{icon('alert',15,ACCENT,2,5)}**Limitations & next steps**", unsafe_allow_html=True)
            for t in R.LIMITATIONS:
                st.markdown(f"- {t}")

# ====================================================================== TRY IT LIVE
elif nav == "Try it Live":
    section("flask", "Try it live")
    st.markdown(("The transformer runs **our fine-tuned DistilBERT** in-process. "
                 if HAS_MODEL else "The fine-tuned model isn't bundled here, so the transformer falls back to a pretrained one. ")
                + "The classic model is trained live on the bundled reviews. "
                + "(The transformer pre-loads in the background when the page opens.)")
    st.markdown(badge("Fine-tuned DistilBERT ready", POS, "check") if HAS_MODEL
                else badge("Pretrained fallback", ACCENT, "info"), unsafe_allow_html=True)
    st.write("")

    @st.fragment
    def demo():
        ex = R.EXAMPLE_REVIEWS
        if "rev" not in st.session_state:
            st.session_state["rev"] = ex[R.DEFAULT_EXAMPLE]
        pick = st.pills("Load an example", list(ex), selection_mode="single", default=None, key="expick")
        if pick and pick != st.session_state.get("_lp"):
            st.session_state["rev"] = ex[pick]
            st.session_state["_lp"] = pick
        st.text_area("Your game review", key="rev", height=110)
        run_tf = st.toggle("Also run the transformer (compare both)", value=True)
        if st.button("Classify", type="primary", icon=":material/play_arrow:"):
            txt = st.session_state["rev"]
            if not txt.strip():
                st.warning("Type a review first.")
                return
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"{icon('scale',17,PRIMARY,2,7)}<b style='color:{TEXT}'>Classic · TF-IDF + LogReg</b>",
                            unsafe_allow_html=True)
                mdl, n = classic()
                pr = mdl.predict_proba([txt])[0]
                pred = int(pr.argmax())
                verdict(pred == 1, pr[pred], f"Trained live on {n:,} reviews.")
            with c2:
                st.markdown(f"{icon('cpu',17,PRIMARY,2,7)}<b style='color:{TEXT}'>Transformer · DistilBERT</b>",
                            unsafe_allow_html=True)
                if not run_tf:
                    panel("info", "Toggle the transformer above to compare both models.")
                else:
                    try:
                        clf, which = transformer()
                        out = clf(txt[:512])[0]
                        verdict(out["label"].upper().startswith("POS"), out["score"], f"Using {which}.")
                    except Exception as e:
                        panel("alert", f"Transformer couldn't load here (often a memory limit on free hosting): {e}", NEG)

    demo()
    st.divider()
    st.caption(R.PROVENANCE)
