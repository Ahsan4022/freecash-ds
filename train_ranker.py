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
train = df[df["event_date"] < cutoff]
test = df[df["event_date"] >= cutoff]

# ---------- 3. Train ----------
model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", max_iter=300, learning_rate=0.05, random_state=42
)
model.fit(train[features], train["completed"])

# ---------- 4. Evaluate ----------
# Baseline: every offer gets its average completion rate from the training period
offer_rate = train.groupby("offer_id", observed=True)["completed"].mean()
base_pred = test["offer_id"].map(offer_rate).fillna(train["completed"].mean())
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

# ---------- 7. Save the model for the A/B test ----------
joblib.dump({"model": model, "features": features, "cat_cols": cat_cols,
             "dtypes": {c: df[c].dtype for c in cat_cols}}, "ranker.joblib")
print("Saved model to ranker.joblib")