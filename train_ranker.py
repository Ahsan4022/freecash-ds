"""
Step 3a: Offer ranking model.

For every (user, offer) pair, predict P(complete): the chance the user completes
the offer after seeing it. Then rank offers by EXPECTED REVENUE:
    expected revenue = P(complete) x what the advertiser pays

Key idea - position bias:
    Offers at the top of the list get more clicks just because they are at the top.
    So we give the model the list position as a feature, and when we RANK we
    ask "how likely would this offer be completed if it were shown at position 1?"
    That way the model judges offers fairly, not by where they happened to be shown.
"""
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ---------- 1. Build training data: one row per impression ----------
con = duckdb.connect("freecash.duckdb", read_only=True)
df = con.execute("""
    WITH imp AS (
        SELECT user_id, offer_id, event_ts, event_date, list_position
        FROM stg_events WHERE event_type = 'impression'
    ),
    clk AS (
        SELECT user_id, offer_id, list_position, event_ts
        FROM stg_events WHERE event_type = 'click'
    ),
    cmp AS (
        SELECT user_id, offer_id, list_position, event_ts
        FROM stg_events WHERE event_type = 'complete'
    ),
    -- A click happens within 60 seconds of the impression it belongs to
    imp_clk AS (
        SELECT i.*, c.event_ts AS click_ts
        FROM imp i
        LEFT JOIN clk c
          ON c.user_id = i.user_id AND c.offer_id = i.offer_id
         AND c.list_position = i.list_position
         AND c.event_ts BETWEEN i.event_ts AND i.event_ts + INTERVAL 60 SECOND
    ),
    -- A completion can come days later, so we link it to the MOST RECENT
    -- matching click before it (ASOF JOIN), so each completion counts only once
    cmp_link AS (
        SELECT c.user_id, c.offer_id, c.event_ts AS click_ts
        FROM cmp m
        ASOF JOIN clk c
          ON c.user_id = m.user_id AND c.offer_id = m.offer_id
         AND c.list_position = m.list_position
         AND m.event_ts >= c.event_ts
    ),
    labelled AS (
        SELECT ic.* EXCLUDE (click_ts),
               EXISTS (
                   SELECT 1 FROM cmp_link l
                   WHERE l.user_id = ic.user_id AND l.offer_id = ic.offer_id
                     AND l.click_ts = ic.click_ts
               ) AS completed
        FROM imp_clk ic
    )
    SELECT l.*,
           u.country, u.platform, u.acquisition_channel, u.age_group,
           o.offer_type, o.category, o.difficulty, o.user_reward_usd,
           o.advertiser_payout_usd, o.est_minutes
    FROM labelled l
    JOIN raw_users  u USING (user_id)
    JOIN raw_offers o USING (offer_id)
""").df()
con.close()

# Optional: add the exploration data from explore.py  (run:  py train_ranker.py --explore)
USE_EXPLORE = "--explore" in sys.argv
if USE_EXPLORE:
    con = duckdb.connect("freecash.duckdb", read_only=True)
    u = con.execute("SELECT user_id, country, platform, acquisition_channel, age_group FROM raw_users").df()
    o = con.execute("""SELECT offer_id, offer_type, category, difficulty, user_reward_usd,
                              advertiser_payout_usd, est_minutes FROM raw_offers""").df()
    con.close()
    ex = pd.read_csv("explore_log.csv", parse_dates=["event_date"]).merge(u, on="user_id").merge(o, on="offer_id")
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = pd.concat([df, ex[[c for c in df.columns if c in ex.columns]]], ignore_index=True)
    print("Using logged data + exploration data")

cat_cols = ["country", "platform", "acquisition_channel", "age_group", "offer_type", "category"]
num_cols = ["difficulty", "user_reward_usd", "advertiser_payout_usd", "est_minutes", "list_position"]
features = cat_cols + num_cols
for c in cat_cols:
    df[c] = df[c].astype("category")
df["completed"] = df["completed"].astype(int)

print(f"Impressions: {len(df):,}   completion rate: {df['completed'].mean():.2%}")
print(f"Offers that were ever shown: {df['offer_id'].nunique()} of 300\n")

# ---------- 2. Time-based split ----------
dates = np.sort(df["event_date"].unique())
cutoff = dates[int(len(dates) * 0.7)]
train = df[df["event_date"] < cutoff].copy()
test = df[df["event_date"] >= cutoff].copy()

# Offer quality feature: each offer's completion rate in the TRAINING period.
# Offers with little data are pulled towards the overall average ("smoothing"),
# so one lucky completion doesn't make an offer look amazing.
prior = train["completed"].mean()
stats = train.groupby("offer_id")["completed"].agg(["sum", "count"])
offer_rate = ((stats["sum"] + prior * 200) / (stats["count"] + 200)).to_dict()
for part in (train, test):
    part["offer_rate"] = part["offer_id"].map(offer_rate).fillna(prior)
features = features + ["offer_rate"]

# ---------- 3. Train ----------
model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", max_iter=300, learning_rate=0.05, random_state=42
)
model.fit(train[features], train["completed"])

# ---------- 4. Evaluate ----------
# Baseline: every offer gets its average completion rate from the training period
base_pred = test["offer_rate"]
model_pred = model.predict_proba(test[features])[:, 1]

print("AUC (0.5 = random, 1.0 = perfect):")
print(f"  Baseline (offer's average rate): {roc_auc_score(test['completed'], base_pred):.3f}")
print(f"  Model:                           {roc_auc_score(test['completed'], model_pred):.3f}\n")

# ---------- 5. Show the position bias the model learned ----------
sample = test.sample(20000, random_state=0)
print("Average predicted completion chance if the SAME offers were shown at position:")
for pos in [1, 3, 5, 10]:
    p = model.predict_proba(sample[features].assign(list_position=pos))[:, 1].mean()
    print(f"  position {pos:>2}: {p:.2%}")
print()

# ---------- 6. What did the OLD system actually show users? ----------
con = duckdb.connect("freecash.duckdb", read_only=True)
all_offers = con.execute("SELECT offer_type FROM raw_offers").df()
con.close()
shown = df.drop_duplicates("offer_id")
print("Offer types: all 300 offers vs. the offers users actually saw")
print(pd.DataFrame({
    "all offers": all_offers["offer_type"].value_counts(normalize=True),
    "ever shown": shown["offer_type"].value_counts(normalize=True),
}).fillna(0).round(2).to_string(), "\n")

# ---------- 7. Retrain on ALL data and save it for the A/B test ----------
# The time split above is only for honest evaluation. For the model we actually
# use, we train on everything, so it also learns from the most recent data
# (for the --explore version, that's where all the exploration data is!).
prior = df["completed"].mean()
stats = df.groupby("offer_id")["completed"].agg(["sum", "count"])
offer_rate = ((stats["sum"] + prior * 200) / (stats["count"] + 200)).to_dict()
df["offer_rate"] = df["offer_id"].map(offer_rate)
model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", max_iter=300, learning_rate=0.05, random_state=42
)
model.fit(df[features], df["completed"])
out_file = "ranker_v2.joblib" if USE_EXPLORE else "ranker.joblib"
joblib.dump({"model": model, "features": features, "cat_cols": cat_cols,
             "offer_rate": offer_rate, "prior": prior,
             "dtypes": {c: df[c].dtype for c in cat_cols}}, out_file)
print(f"Saved model to {out_file}")