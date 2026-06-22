"""
Run the notebook's modeling pipeline on the Steam review dataset.
=================================================================
Data: steam_reviews.csv (collected via the official Steam Reviews API).
Labels come from each review's recommend / not-recommend (voted_up).

Pipeline: balanced binary sample -> EDA -> TF-IDF + Logistic Regression ->
fine-tune DistilBERT (1 epoch, AdamW 2e-5, batch 8, max-len 128).
Writes results_real.json.
"""

import json, os, re
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "steam_reviews.csv")
OUT = os.path.join(HERE, "results_real.json")
SEED = 42

raw = pd.read_csv(CSV)
raw_count = len(raw)
game_dist = {str(k): int(v) for k, v in raw["product_name"].value_counts().items()}

df = raw[raw["label"].isin([0, 1])].dropna(subset=["text"]).copy()
df["label"] = df["label"].astype(int)
n = min(int((df.label == 1).sum()), int((df.label == 0).sum()))
df = pd.concat([df[df.label == 1].sample(n, random_state=SEED),
                df[df.label == 0].sample(n, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)[["text", "label"]]
df["text"] = df["text"].fillna("")
balanced_total = len(df)
label_balance = {"Positive (1)": int((df.label == 1).sum()), "Negative (0)": int((df.label == 0).sum())}
unique_reviews = int(df["text"].nunique())
duplicate_reviews = int(balanced_total - unique_reviews)
print("balanced:", balanced_total, label_balance, flush=True)

# EDA
df["rlen"] = df["text"].apply(lambda x: len(x.split()))
avg = df.groupby("label")["rlen"].mean()
avg_review_length = {"Negative (0)": round(float(avg[0]), 1), "Positive (1)": round(float(avg[1]), 1)}

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))
toks = re.findall(r"[A-Za-z']{3,}", " ".join(df["text"]))
words = [w.lower() for w in toks if w.lower() not in stop]
freq = Counter(words)
top_words = [[w, int(c)] for w, c in freq.most_common(20)]
top_words_full = [[w, int(c)] for w, c in freq.most_common(120)]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_features = sorted(TfidfVectorizer(max_features=20, stop_words="english").fit(df["text"]).get_feature_names_out().tolist())

# TF-IDF + Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"])
vec = TfidfVectorizer(stop_words="english", max_features=5000)
lrm = LogisticRegression(max_iter=1000, random_state=SEED).fit(vec.fit_transform(Xtr), ytr)
lrp = lrm.predict(vec.transform(Xte))
rep = classification_report(yte, lrp, output_dict=True)
lr = {"accuracy": round(accuracy_score(yte, lrp), 4), "precision": round(precision_score(yte, lrp, average="weighted"), 4),
      "recall": round(recall_score(yte, lrp, average="weighted"), 4), "f1": round(f1_score(yte, lrp, average="weighted"), 4),
      "report": {"Negative (0)": {k: round(rep["0"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(rep["0"]["support"])},
                 "Positive (1)": {k: round(rep["1"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(rep["1"]["support"])}},
      "train_size": int(Xtr.shape[0]), "test_size": int(Xte.shape[0])}
print("LR:", lr["accuracy"], lr["f1"], flush=True)

# DistilBERT
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("device:", device, flush=True)
hf = Dataset.from_pandas(df[["text", "label"]], preserve_index=False).train_test_split(test_size=0.2, seed=SEED)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = hf.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"]).rename_column("label", "labels")
enc = enc.remove_columns([c for c in enc["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]])
coll = DataCollatorWithPadding(tokenizer=tok)
tl = DataLoader(enc["train"], shuffle=True, batch_size=8, collate_fn=coll)
el = DataLoader(enc["test"], batch_size=8, collate_fn=coll)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for step, b in enumerate(tl):
    b = {k: v.to(device) for k, v in b.items()}
    o = model(**b); o.loss.backward(); opt.step(); opt.zero_grad()
    if step % 150 == 0:
        print(f"step {step}/{len(tl)} loss {o.loss.item():.4f}", flush=True)
model.eval()
preds, labels = [], []
with torch.no_grad():
    for b in el:
        y = b["labels"]; b = {k: v.to(device) for k, v in b.items()}
        preds.extend(torch.argmax(model(**b).logits, -1).cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
drep = classification_report(labels, preds, output_dict=True)
cm = confusion_matrix(labels, preds)
distilbert = {"accuracy": round(accuracy_score(labels, preds), 4), "f1": round(f1_score(labels, preds, average="weighted"), 4),
              "report": {"Negative": {k: round(drep["0"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(drep["0"]["support"])},
                         "Positive": {k: round(drep["1"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(drep["1"]["support"])}},
              "confusion": cm.tolist(), "correct": int(np.trace(cm)), "misclassified": int(cm.sum() - np.trace(cm)), "test_size": int(len(labels))}
print("DistilBERT:", distilbert["accuracy"], distilbert["f1"], flush=True)

json.dump({"dataset": "Steam game reviews (official Steam Reviews API)", "raw_count": raw_count,
           "n_games": len(game_dist), "game_dist": game_dist, "balanced_total": balanced_total,
           "label_balance": label_balance, "unique_reviews": unique_reviews, "duplicate_reviews": duplicate_reviews,
           "avg_review_length": avg_review_length, "top_words": top_words, "top_words_full": top_words_full,
           "tfidf_features": tfidf_features, "lr": lr, "distilbert": distilbert}, open(OUT, "w"), indent=2)
print("WROTE", OUT, flush=True)
