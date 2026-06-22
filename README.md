# 🎮 Steam Review Sentiment — Traditional NLP vs. Transformers

ADS-509 (Applied Text Mining, University of San Diego) final project. We collect **real Steam
game reviews via the official Steam Reviews API**, then compare a classic **TF-IDF + Logistic
Regression** baseline against a fine-tuned **DistilBERT** transformer for binary sentiment.

> **Team:** Gagandeep Singh · Shivam Patel · Luigi Salemi

This repo contains **both** the data-collection + modeling **notebook** and an interactive
**Streamlit dashboard**.

## Data
- **Source:** official **Steam Reviews API** (public, no key). See [DATA_COLLECTION.md](DATA_COLLECTION.md).
- **12,661 recent reviews (2024–2026)** across **247 games** (modern hits + divisive titles like
  EA SPORTS FC, Overwatch 2, Battlefield 2042, plus a broad most-owned pool for variety).
- **Labels are real**, not inferred: each review's own **recommend / not-recommend** (`voted_up`).

## Results (one end-to-end run on the balanced 80/20 split)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| TF-IDF + Logistic Regression | **78.9%** | 78.9% |
| Fine-tuned DistilBERT | **85.1%** | 85.1% |

**DistilBERT beats the baseline by ~6.2 points** — its edge is larger here than on cleaner
datasets because game reviews are full of sarcasm, slang, and memes that bag-of-words misses.

## Repo contents
```
Steam_Review_Sentiment.ipynb   # scraping + EDA + TF-IDF/LR + DistilBERT (run end-to-end)
app.py                         # interactive dashboard
DATA_COLLECTION.md             # exactly how the data was scraped
fetch_steam.py                 # Steam API collector (247 games, 2024-2026, voted_up labels)
run_steam.py                   # pipeline -> results_real.json
build_results.py               # results_real.json -> results.py
results.py / results_real.json # the numbers shown in the dashboard
data/reviews.csv               # collected reviews (used by the live demo)
model/                         # the fine-tuned DistilBERT (Git LFS) — used live in the app
assets/wordcloud.png
```

The **Try it Live** tab runs the project's **own fine-tuned DistilBERT** (saved to `model/` by the
notebook), not an off-the-shelf model — so you classify with the exact model trained here.

## Run it
```bash
pip install -r requirements.txt

# (optional) re-collect fresh reviews from the Steam API
python fetch_steam.py

# reproduce the results, then launch the dashboard
python run_steam.py && python build_results.py
streamlit run app.py
```

## Deploy
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → New app → `app.py`.
