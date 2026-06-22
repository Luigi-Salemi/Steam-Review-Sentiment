"""
Steam Review Sentiment — simple Streamlit dashboard
===================================================
Traditional NLP vs. Transformers on real Steam reviews (2024-2026) collected via
the official Steam Reviews API. All numbers from run_steam.py.

Run:  streamlit run app.py
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import results as R

st.set_page_config(page_title="Steam Review Sentiment — ADS-509", page_icon="🎮", layout="wide")
pio.templates.default = "plotly_dark"

PRIMARY, GOLD, POS, NEG = "#4C90F0", "#F0B726", "#32A467", "#E76A6E"
SUBTLE, BORDER, TEXT = "#ABB3BF", "#383E47", "#F6F7F9"
GAIN = round((R.DISTILBERT_METRICS["Accuracy"] - R.LR_METRICS["Accuracy"]) * 100, 1)


def style(fig, h=360):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=SUBTLE), height=h, margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def report_df(rep):
    df = pd.DataFrame(rep).T.rename(columns={"f1": "f1-score"})
    df["support"] = df["support"].astype(int)
    return df.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                            "f1-score": "{:.2f}", "support": "{:d}"})


# ── Header ──
st.title("🎮 " + R.PROJECT["title"])
st.caption(f"{R.PROJECT['course']} · {R.PROJECT['school']}  —  " + " · ".join(R.PROJECT["team"]))
st.write(R.PROJECT["objective"])

a, b, c, d = st.columns(4)
a.metric("Reviews (2024-26)", f"{R.RAW_REVIEW_COUNT:,}", border=True)
b.metric("Balanced sample", f"{R.BALANCED_TOTAL:,}", border=True)
c.metric("DistilBERT accuracy", f"{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%",
         f"+{GAIN} pts vs baseline", border=True)
d.metric("Games", f"{R.N_GAMES}", border=True)

# ── Dataset ──
st.divider()
st.subheader("Dataset")
st.caption(f"{R.RAW_REVIEW_COUNT:,} recent (2024-2026) reviews across {R.N_GAMES} games, "
           f"collected via the official Steam Reviews API. Labels come from each review's "
           f"**recommend / not-recommend** (Steam has no star ratings). See DATA_COLLECTION.md.")
left, right = st.columns(2)
with left:
    st.caption("Top 20 games by reviews collected")
    g = pd.DataFrame({"Game": list(R.GAME_DIST), "n": list(R.GAME_DIST.values())}).sort_values("n")
    figg = go.Figure(go.Bar(x=g["n"], y=g["Game"], orientation="h", marker_color=PRIMARY))
    figg.update_layout(xaxis_title="Reviews", yaxis_title="")
    st.plotly_chart(style(figg, 460), width="stretch")
with right:
    st.caption("Class balance (recommend vs not)")
    labels = list(R.LABEL_BALANCE.keys())
    figp = go.Figure(go.Pie(labels=labels, values=list(R.LABEL_BALANCE.values()), hole=0.55,
                            marker=dict(colors=[POS if "Positive" in l else NEG for l in labels]),
                            textfont=dict(color=TEXT)))
    figp.update_layout(legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(style(figp, 300), width="stretch")
    st.metric("Unique reviews", f"{R.UNIQUE_REVIEWS:,}", border=True)

# ── EDA ──
st.divider()
st.subheader("Exploratory analysis")
left, right = st.columns(2)
with left:
    st.caption("Average review length by sentiment (words)")
    sent = list(R.AVG_REVIEW_LENGTH)
    vals = list(R.AVG_REVIEW_LENGTH.values())
    figl = go.Figure(go.Bar(x=sent, y=vals, text=vals, textposition="outside",
                            marker_color=[NEG if "Negative" in s else POS for s in sent]))
    figl.update_traces(textfont_color=TEXT)
    st.plotly_chart(style(figl, 320), width="stretch")
with right:
    st.caption("Top 15 words (stopwords removed)")
    tw = pd.DataFrame(R.TOP_WORDS, columns=["Word", "n"]).head(15).sort_values("n")
    figw = go.Figure(go.Bar(x=tw["n"], y=tw["Word"], orientation="h", marker_color=PRIMARY))
    st.plotly_chart(style(figw, 320), width="stretch")

wc = os.path.join(os.path.dirname(__file__), "assets", "wordcloud.png")
if os.path.exists(wc):
    st.image(wc, width="stretch")

# ── Model comparison ──
st.divider()
st.subheader("Model comparison")
comp = pd.DataFrame(R.MODEL_COMPARISON)
figb = go.Figure()
figb.add_bar(name="Accuracy", x=comp["Model"], y=comp["Accuracy"], marker_color=PRIMARY,
             text=[f"{v*100:.1f}%" for v in comp["Accuracy"]], textposition="outside")
figb.add_bar(name="F1", x=comp["Model"], y=comp["F1 (weighted)"], marker_color=GOLD,
             text=[f"{v*100:.1f}%" for v in comp["F1 (weighted)"]], textposition="outside")
figb.update_traces(textfont_color=TEXT)
figb.update_layout(barmode="group", yaxis=dict(range=[0.8, 1.0], tickformat=".0%"))
st.plotly_chart(style(figb, 380), width="stretch")
st.success(f"Fine-tuned **DistilBERT ({R.DISTILBERT_METRICS['Accuracy']*100:.1f}%)** beats the "
           f"**TF-IDF + Logistic Regression baseline ({R.LR_METRICS['Accuracy']*100:.1f}%)** "
           f"by ~{GAIN} points.")

left, right = st.columns(2)
with left:
    st.caption("TF-IDF + Logistic Regression")
    st.dataframe(report_df(R.LR_REPORT), width="stretch")
with right:
    st.caption("Fine-tuned DistilBERT")
    st.dataframe(report_df(R.DISTILBERT_REPORT), width="stretch")
    labels = ["Negative", "Positive"]
    cmf = go.Figure(go.Heatmap(z=R.DISTILBERT_CONFUSION, x=labels, y=labels,
                               text=R.DISTILBERT_CONFUSION, texttemplate="%{text}",
                               textfont=dict(size=18, color=TEXT),
                               colorscale=[[0, "#252A31"], [1, PRIMARY]], showscale=False, xgap=4, ygap=4))
    cmf.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(cmf, 300), width="stretch")

# ── Try it live ──
st.divider()
st.subheader("Try it live")
st.caption("Type a game review and classify it. The classic model is trained live on the bundled "
           "reviews; the transformer is an off-the-shelf pretrained model (optional).")


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


@st.cache_resource(show_spinner="Loading the fine-tuned DistilBERT…")
def transformer():
    """Use OUR fine-tuned model (./model) if present, else fall back to a pretrained one."""
    from transformers import pipeline
    local = os.path.join(os.path.dirname(__file__), "model")
    if os.path.isdir(local):
        return pipeline("sentiment-analysis", model=local, tokenizer=local), "our fine-tuned DistilBERT"
    return (pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
            "pretrained DistilBERT (fallback)")


text = st.text_area("Review", value=list(R.EXAMPLE_REVIEWS.values())[0], height=100)
run_tf = st.checkbox("Also run the fine-tuned transformer (optional, slower on first use)")

if st.button("Classify", type="primary") and text.strip():
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classic · TF-IDF + LogReg**")
        model, n = classic()
        proba = model.predict_proba([text])[0]
        pred = int(proba.argmax())
        (st.success if pred == 1 else st.error)(
            f"{'Positive 👍' if pred == 1 else 'Negative 👎'} — {proba[pred]*100:.0f}% confident")
        st.caption(f"Trained on {n:,} real Steam reviews.")
    with c2:
        st.markdown("**Transformer · Fine-tuned DistilBERT**")
        if not run_tf:
            st.info("Tick the box above to run the transformer.")
        else:
            try:
                clf, which = transformer()
                out = clf(text[:512])[0]
                pos = out["label"].upper().startswith("POS")
                (st.success if pos else st.error)(
                    f"{'Positive 👍' if pos else 'Negative 👎'} — {out['score']*100:.0f}% confident")
                st.caption(f"Using {which}.")
            except Exception as e:
                st.warning(f"Couldn't load the transformer here (memory limit on free hosting): {e}")

st.divider()
st.caption(R.PROVENANCE)
