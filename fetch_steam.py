"""
Collect REAL product reviews from the official Steam Reviews API — 100+ games.
=============================================================================
No key, no signup, free.
  - Game list: current top ~100 by players (SteamSpy) + modern divisive seeds
    (EA SPORTS FC, Call of Duty, Overwatch 2, ...) so there are real negatives.
  - Reviews: https://store.steampowered.com/appreviews/<appid>?json=1
    Each review's `voted_up` is a built-in positive/negative label.

Writes steam_reviews.csv in the project schema. Run: python fetch_steam.py
"""

import csv
import json
import os
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steam_reviews.csv")
PER_TYPE = int(os.environ.get("PER_TYPE", "28"))   # reviews per game per sentiment (kept small for variety)
MAX_GAMES = int(os.environ.get("MAX_GAMES", "250"))
MIN_TS = 1704067200   # 2024-01-01 UTC — keep only recent (2024-2026) reviews
HDR = {"User-Agent": "Mozilla/5.0 (academic review collection)"}

# modern + divisive titles (guarantee variety + a real negative class). FIFA is
# now EA SPORTS FC.
SEED = {
    2669320: "EA SPORTS FC 25", 2195250: "EA SPORTS FC 24", 1811260: "FIFA 23",
    2357570: "Overwatch 2", 1665460: "eFootball 2024", 1517290: "Battlefield 2042",
    2344520: "Diablo IV", 1716740: "Starfield", 2054970: "Dragon's Dogma 2",
    2429640: "Throne and Liberty", 1086940: "Baldur's Gate 3", 1623730: "Palworld",
    2358720: "Black Myth: Wukong", 553850: "Helldivers 2", 2183900: "Space Marine 2",
    2767030: "Marvel Rivals", 1245620: "Elden Ring", 1091500: "Cyberpunk 2077",
    990080: "Hogwarts Legacy", 1966720: "Lethal Company", 1145350: "Hades II",
    2694490: "Path of Exile 2", 1778820: "Tekken 8", 2139460: "Once Human",
}


def get_top_games():
    """Broad, varied pool: current most-played (modern) + most-owned 1000."""
    pool = {}
    # modern / currently popular first
    try:
        req = urllib.request.Request("https://steamspy.com/api.php?request=top100in2weeks", headers=HDR)
        for a, d in json.load(urllib.request.urlopen(req, timeout=30)).items():
            pool[int(a)] = d.get("name", f"App {a}")
        time.sleep(1)
    except Exception as e:
        print("top100in2weeks unavailable:", e, flush=True)
    # broad variety: most-owned games (up to ~1000)
    try:
        req = urllib.request.Request("https://steamspy.com/api.php?request=all&page=0", headers=HDR)
        for a, d in json.load(urllib.request.urlopen(req, timeout=60)).items():
            pool.setdefault(int(a), d.get("name", f"App {a}"))
    except Exception as e:
        print("all/page0 unavailable:", e, flush=True)
    return pool


def fetch(appid, review_type, want):
    """Newest-first; keep only reviews created on/after MIN_TS (2024+)."""
    out, cursor = [], "*"
    while len(out) < want:
        q = urllib.parse.urlencode({"json": 1, "num_per_page": 100, "filter": "recent",
                                    "language": "english", "review_type": review_type,
                                    "purchase_type": "all", "cursor": cursor})
        try:
            req = urllib.request.Request(f"https://store.steampowered.com/appreviews/{appid}?{q}", headers=HDR)
            data = json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            break
        revs = data.get("reviews", [])
        if not revs:
            break
        before = len(out)
        out.extend(v for v in revs if v.get("timestamp_created", 0) >= MIN_TS)
        # recent filter is newest-first: if a full page added nothing in range, stop
        if len(out) == before and revs[-1].get("timestamp_created", 0) < MIN_TS:
            break
        cursor = data.get("cursor", "")
        if not cursor:
            break
        time.sleep(0.2)
    return out[:want]


games = {**SEED, **get_top_games()}          # seeds guaranteed in the first slots
games = dict(list(games.items())[:MAX_GAMES])
print(f"collecting from {len(games)} games...", flush=True)

rows, seen = [], set()
for i, (appid, name) in enumerate(games.items(), 1):
    for rtype, label in [("positive", 1), ("negative", 0)]:
        for v in fetch(appid, rtype, PER_TYPE):
            rid = v.get("recommendationid")
            if rid in seen:
                continue
            seen.add(rid)
            text = (v.get("review") or "").replace("\n", " ").replace("\r", " ").strip()
            if len(text) < 5:
                continue
            ts = v.get("timestamp_created", 0)
            # Steam has no star rating — label is the reviewer's recommend/not-recommend
            rows.append({"review_id": rid, "product_name": name, "review_title": "",
                         "review_text": text, "text": text, "label": label,
                         "review_date": time.strftime("%Y-%m-%d", time.gmtime(ts)),
                         "source_url": f"https://store.steampowered.com/app/{appid}",
                         "category": "Video Games"})
    if i % 10 == 0:
        print(f"  {i}/{len(games)} games -> {len(rows)} reviews", flush=True)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
pos = sum(r["label"] == 1 for r in rows)
print(f"\nWROTE {OUT}: {len(rows)} reviews from {len({r['product_name'] for r in rows})} games "
      f"({pos} positive / {len(rows)-pos} negative)")
