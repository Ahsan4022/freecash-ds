"""
Step 3b: Simulate a 14-day A/B test.

  Control   (A): the current ranking - highest advertiser payout first
  Treatment (B): our model - highest EXPECTED revenue first (P(complete) x payout)

Every user is randomly assigned to A or B (50/50). We then simulate how they
behave, using the same hidden "real world" rules as generate_data.py.
Output: ab_results.csv with one row per user.
"""
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd

# Which model to test, e.g.:  py ab_test.py ranker_v2.joblib
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "ranker.joblib"
OUT_FILE = MODEL_FILE.replace("ranker", "ab_results").replace(".joblib", ".csv")
DAYS = 14
LIST_SIZE = 10
rng = np.random.default_rng(2026)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------- 1. Load users, offers and the hidden truth ----------
con = duckdb.connect("freecash.duckdb", read_only=True)
users = con.execute("SELECT * FROM raw_users ORDER BY user_id").df()
offers = con.execute("SELECT * FROM raw_offers ORDER BY offer_id").df()
con.close()
users = users.merge(pd.read_csv("data/_truth_users.csv"), on="user_id")
offers = offers.merge(pd.read_csv("data/_truth_offers.csv"), on="offer_id")

# ---------- 2. Random assignment ----------
users["group"] = np.where(rng.random(len(users)) < 0.5, "A_control", "B_model")

# ---------- 3. Model scores for every (user, offer) pair ----------
saved = joblib.load(MODEL_FILE)
model, features = saved["model"], saved["features"]
pairs = users[["user_id", "country", "platform", "acquisition_channel", "age_group"]].merge(
    offers[["offer_id", "offer_type", "category", "difficulty", "user_reward_usd",
            "advertiser_payout_usd", "est_minutes"]], how="cross")
pairs["list_position"] = 1                      # judge every offer as if shown at the top
pairs["offer_rate"] = pairs["offer_id"].map(saved["offer_rate"]).fillna(saved["prior"])
for c, dt in saved["dtypes"].items():
    # Values the model never saw in training (e.g. offer types that were never shown)
    # become "missing" - the model has no information about them.
    pairs[c] = pd.Categorical(pairs[c], categories=dt.categories)
p_complete = model.predict_proba(pairs[features])[:, 1]
exp_revenue = (p_complete * pairs["advertiser_payout_usd"].to_numpy()).reshape(len(users), len(offers))

# ---------- 4. Simulate ----------
type_idx = offers["offer_type"].map({"game": 0, "app": 1, "survey": 2}).to_numpy()
attract = offers["_attractiveness"].to_numpy()
diff = offers["difficulty"].to_numpy()
payout = offers["advertiser_payout_usd"].to_numpy()
reward = offers["user_reward_usd"].to_numpy()
pos = np.arange(1, LIST_SIZE + 1)

rows = []
for i, u in enumerate(users.to_dict("records")):
    eng = u["_engagement"]
    prefs = np.array([u["_pref_game"], u["_pref_app"], u["_pref_survey"]])
    s = dict(user_id=u["user_id"], group=u["group"], sessions=0, impressions=0,
             clicks=0, completions=0, revenue_usd=0.0, rewards_usd=0.0)
    for day in range(DAYS):
        if rng.random() >= min(0.9, 0.35 * eng):            # not active today
            continue
        for _ in range(rng.poisson(0.5 + eng)):
            if u["group"] == "A_control":
                score = payout * rng.uniform(0.6, 1.4, len(payout))
            else:
                score = exp_revenue[i] * rng.uniform(0.8, 1.2, len(payout))
            shown = np.argsort(-score)[:LIST_SIZE]
            # Hidden real-world behaviour (same rules as generate_data.py)
            logit = (-2.2 + attract[shown] + prefs[type_idx[shown]]
                     + 0.4 * np.log(eng) - 0.25 * (pos - 1))
            clicked = rng.random(LIST_SIZE) < sigmoid(logit)
            done_p = sigmoid(1.0 - 3.5 * diff[shown] + 0.6 * np.log(eng))
            completed = clicked & (rng.random(LIST_SIZE) < done_p)
            s["sessions"] += 1
            s["impressions"] += LIST_SIZE
            s["clicks"] += int(clicked.sum())
            s["completions"] += int(completed.sum())
            s["revenue_usd"] += float(payout[shown][completed].sum())
            s["rewards_usd"] += float(reward[shown][completed].sum())
    rows.append(s)

results = pd.DataFrame(rows).merge(users[["user_id", "age_group"]], on="user_id")
results.to_csv(OUT_FILE, index=False)
print(results.groupby("group")[["sessions", "clicks", "completions", "revenue_usd"]].sum().round(0))
print(f"\nTested {MODEL_FILE}. Saved {OUT_FILE}. Now run:  py ab_analysis.py {OUT_FILE}")