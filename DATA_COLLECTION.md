# Data Collection — How the reviews were scraped

All review data is collected from the **official Steam Reviews API** (public, no key,
no signup). Reproduce it with `python fetch_steam.py`.

## 1. Source
- **Endpoint:** `https://store.steampowered.com/appreviews/<appid>?json=1`
  (Valve's public "Get App Reviews" endpoint — [docs](https://partner.steamgames.com/doc/store/getreviews)).
- **No authentication / no rate-limit key.** We self-throttle to ~5 requests/second.

## 2. Which games (the variety)
We build a broad, modern pool of games and merge:
1. **SteamSpy `top100in2weeks`** — games with the most players right now (keeps it *current/modern*).
2. **SteamSpy `all` (page 0)** — the most-owned games (adds *breadth* across genres).
3. **A curated seed list of modern, divisive titles** — e.g. **EA SPORTS FC 25/24** (formerly FIFA),
   Call of Duty, Overwatch 2, eFootball, Battlefield 2042, Diablo IV, Starfield — these guarantee a
   real **negative** class.

We cap the pool at **~250 games** (`MAX_GAMES`).

## 3. Which reviews
For each game we request reviews of **each polarity separately** so the dataset is balanced:
- `review_type=positive` and `review_type=negative`
- `filter=recent` (newest first), `language=english`, `purchase_type=all`
- `PER_TYPE` reviews per game per polarity (default **28**) — kept small on purpose so the data is
  spread across *many* games rather than dominated by a few.
- Paginated via the response `cursor`.

## 4. Relevance filter (2024–2026)
We keep only reviews **created on or after 2024-01-01** (`MIN_TS`), so the dataset reflects recent,
relevant opinions. Because `filter=recent` returns newest-first, we stop paging a game once its
reviews fall before the cutoff.

## 5. Label
Steam has **no star rating** — instead each review carries `voted_up`:
- `voted_up = true`  → **Positive (1)**  (recommended)
- `voted_up = false` → **Negative (0)**  (not recommended)

## 6. Cleaning & schema
- De-duplicate by `recommendationid`; drop reviews shorter than 5 characters; strip newlines.
- Output `steam_reviews.csv` with columns:
  `review_id, product_name, review_title, review_text, text, rating, sentiment, label, review_date, source_url, category`
  (matching the project's standard schema; `rating` is a nominal 5/1 derived from the label since Steam has no stars).

## 7. Balancing & split (modeling)
`run_steam.py` then equalizes the two classes (down-sample the larger), and uses a **stratified 80/20**
train/test split for both the TF-IDF + Logistic Regression baseline and the fine-tuned DistilBERT.

## Ethics
The Steam Reviews API is a public, documented endpoint intended for this kind of access; we use no
login, collect only public review text, and throttle our requests. This is **API-based collection**
(not HTML scraping of a site that forbids it).
