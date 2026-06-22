"""Generate Steam_Review_Sentiment.ipynb — the data-collection + modeling notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
def md(t): c.append(nbf.v4.new_markdown_cell(t))
def code(t): c.append(nbf.v4.new_code_cell(t))

md("""# Steam Review Sentiment — Traditional NLP vs. Transformers
**ADS-509: Applied Text Mining · University of San Diego**

Collect real product reviews from the **official Steam Reviews API** (100+ modern
games) and compare a classic **TF-IDF + Logistic Regression** baseline against a
fine-tuned **DistilBERT** transformer for binary sentiment.

Sentiment label comes from each review's *recommend / not-recommend* (`voted_up`).""")

md("## 0. Imports")
code("""%matplotlib inline
import os, time, urllib.request, urllib.parse, json, re
from collections import Counter
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk; from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding)
nltk.download('stopwords', quiet=True)""")

md("""## 1. Data collection — official Steam Reviews API
We take the current top ~100 games (SteamSpy) plus modern divisive titles
(EA SPORTS FC, Call of Duty, Overwatch 2, …) so we get **both** positive and
negative reviews, then pull recent reviews of each via the public Steam endpoint
`https://store.steampowered.com/appreviews/<appid>?json=1`.""")
code('''HDR = {"User-Agent": "Mozilla/5.0 (academic review collection)"}
PER_TYPE = 28   # few reviews per game -> spread across MANY games for variety

SEED = {2669320:"EA SPORTS FC 25", 2195250:"EA SPORTS FC 24", 1811260:"FIFA 23",
        2357570:"Overwatch 2", 1665460:"eFootball 2024", 1517290:"Battlefield 2042",
        2344520:"Diablo IV", 1716740:"Starfield", 2054970:"Dragon's Dogma 2",
        2429640:"Throne and Liberty", 1086940:"Baldur's Gate 3", 1623730:"Palworld",
        2358720:"Black Myth: Wukong", 553850:"Helldivers 2", 2183900:"Space Marine 2",
        2767030:"Marvel Rivals", 1245620:"Elden Ring", 1091500:"Cyberpunk 2077"}

def get_top_games():
    "Merge several SteamSpy top lists for a broad, varied pool (~200+ games)."
    pool = {}
    for name in ["top100in2weeks", "top100forever", "top100owned"]:
        try:
            req = urllib.request.Request(f"https://steamspy.com/api.php?request={name}", headers=HDR)
            for a, v in json.load(urllib.request.urlopen(req, timeout=30)).items():
                pool[int(a)] = v.get("name", f"App {a}")
            time.sleep(1)
        except Exception as e:
            print(name, "unavailable:", e)
    return pool

def fetch(appid, review_type, want):
    out, cursor = [], "*"
    while len(out) < want:
        q = urllib.parse.urlencode({"json":1,"num_per_page":100,"filter":"recent","language":"english",
                                    "review_type":review_type,"purchase_type":"all","cursor":cursor})
        try:
            req = urllib.request.Request(f"https://store.steampowered.com/appreviews/{appid}?{q}", headers=HDR)
            data = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            break
        revs = data.get("reviews", [])
        if not revs: break
        out.extend(revs); cursor = data.get("cursor", "")
        if not cursor: break
        time.sleep(0.2)
    return out[:want]

MAX_GAMES, MIN_TS = 250, 1704067200   # MIN_TS = 2024-01-01 (keep recent reviews only)

# Reuse the already-collected file if present (fast + consistent); else scrape live.
if os.path.exists("steam_reviews.csv"):
    df = pd.read_csv("steam_reviews.csv")[["product_name","text","label"]].dropna()
    print("loaded", len(df), "reviews from steam_reviews.csv")
else:
    games = dict(list({**SEED, **get_top_games()}.items())[:MAX_GAMES])
    rows, seen = [], set()
    for appid, name in games.items():
        for rtype, label in [("positive",1), ("negative",0)]:
            for v in fetch(appid, rtype, PER_TYPE):
                rid = v.get("recommendationid")
                if rid in seen: continue
                seen.add(rid)
                if v.get("timestamp_created",0) < MIN_TS: continue
                text = (v.get("review") or "").replace("\\n"," ").strip()
                if len(text) < 5: continue
                rows.append({"product_name":name, "text":text, "label":label})
    df = pd.DataFrame(rows); df.to_csv("steam_reviews.csv", index=False)
print("reviews:", len(df), "| games:", df.product_name.nunique())
df.head()''')

md("## 2. Balanced binary sample\nLabels come from recommend (1) / not-recommend (0). Balance the two classes.")
code('''n = min((df.label==1).sum(), (df.label==0).sum())
df = pd.concat([df[df.label==1].sample(n, random_state=42),
                df[df.label==0].sample(n, random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)
print("balanced:", len(df)); print(df.label.value_counts())''')

md("## 3. Exploratory data analysis")
code('''df["len"] = df["text"].str.split().map(len)
print(df.groupby("label")["len"].mean())
df["len"].clip(upper=300).hist(bins=40); plt.title("Review length"); plt.show()''')
code('''stop = set(stopwords.words("english"))
words = [w.lower() for w in re.findall(r"[A-Za-z']{3,}", " ".join(df["text"])) if w.lower() not in stop]
freq = Counter(words)
top = pd.DataFrame(freq.most_common(20), columns=["word","n"])
plt.barh(top["word"][::-1], top["n"][::-1]); plt.title("Top 20 words"); plt.show()''')
code('''WordCloud(width=1000, height=400, background_color="white").generate_from_frequencies(dict(freq.most_common(150)))
plt.figure(figsize=(12,5)); plt.imshow(WordCloud(width=1000,height=400,background_color="white").generate_from_frequencies(dict(freq.most_common(150)))); plt.axis("off"); plt.show()''')

md("## 4. Baseline — TF-IDF + Logistic Regression")
code('''Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
vec = TfidfVectorizer(stop_words="english", max_features=5000)
lr = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)
pred = lr.predict(vec.transform(Xte))
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print(classification_report(yte, pred, target_names=["Negative","Positive"]))''')

md("## 5. Fine-tune DistilBERT\n`distilbert-base-uncased`, 1 epoch, AdamW (lr 2e-5), batch 8, max-len 128.")
code('''device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
hf = Dataset.from_pandas(df[["text","label"]], preserve_index=False).train_test_split(test_size=0.2, seed=42)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = hf.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"]).rename_column("label","labels")
enc = enc.remove_columns([c for c in enc["train"].column_names if c not in ["input_ids","attention_mask","labels"]])
coll = DataCollatorWithPadding(tokenizer=tok)
tl = DataLoader(enc["train"], shuffle=True, batch_size=8, collate_fn=coll)
el = DataLoader(enc["test"], batch_size=8, collate_fn=coll)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for step, b in enumerate(tl):
    b = {k:v.to(device) for k,v in b.items()}
    out = model(**b); out.loss.backward(); opt.step(); opt.zero_grad()
    if step % 200 == 0: print("step", step, "loss", round(out.loss.item(),4))
model.eval(); preds, labels = [], []
with torch.no_grad():
    for b in el:
        y = b["labels"]; b = {k:v.to(device) for k,v in b.items()}
        preds += torch.argmax(model(**b).logits, -1).cpu().tolist(); labels += y.tolist()
print("DistilBERT accuracy:", round(accuracy_score(labels, preds), 4))
print(classification_report(labels, preds, target_names=["Negative","Positive"]))
ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=["Negative","Positive"]).plot(); plt.show()

# Save the fine-tuned model so the Streamlit app can use it for live inference
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model.save_pretrained("model"); tok.save_pretrained("model")
print("saved fine-tuned model to ./model")''')

md("""## 6. Results
DistilBERT's contextual embeddings typically beat the TF-IDF + Logistic Regression
baseline by a few points on this dataset. The interactive dashboard (`app.py`)
presents these results; see the README to run it.""")

nb["cells"] = c
nbf.write(nb, "Steam_Review_Sentiment.ipynb")
print("wrote Steam_Review_Sentiment.ipynb with", len(c), "cells")
