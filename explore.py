"""
Step 3c: Fix the feedback loop with EXPLORATION.

Problem: the old ranking (highest payout first) only ever showed 36 of 300 offers,
all games. So there is no data on apps or surveys, and no model can learn about them.

Fix: run the old ranking for 14 more days, but in 20% of sessions show 10 RANDOM
offers instead. This costs a little revenue short-term, but collects data on
every offer. Output: explore_log.csv (one row per impression, with the outcome).
"""
import duckdb
import numpy as np
import pandas as pd

DAYS = 14
LIST_SIZE = 10
EXPLORE_RATE = 0.20
rng = np.random.default_rng(7)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


con = duckdb.connect("freecash.duckdb", read_only=True)
users = con.execute("SELECT * FROM raw_users ORDER BY user_id").df()
offers = con.execute("SELECT * FROM raw_offers ORDER BY offer_id").df()
last_day = con.execute("SELECT MAX(event_date) FROM stg_events").fetchone()[0]
con.close()
users = users.merge(pd.read_csv("data/_truth_users.csv"), on="user_id")
offers = offers.merge(pd.read_csv("data/_truth_offers.csv"), on="offer_id")

type_idx = offers["offer_type"].map({"game": 0, "app": 1, "survey": 2}).to_numpy()
attract = offers["_attractiveness"].to_numpy()
diff = offers["difficulty"].to_numpy()
payout = offers["advertiser_payout_usd"].to_numpy()
offer_ids = offers["offer_id"].to_numpy()
pos = np.arange(1, LIST_SIZE + 1)

log = []
for u in users.to_dict("records"):
    eng = u["_engagement"]
    prefs = np.array([u["_pref_game"], u["_pref_app"], u["_pref_survey"]])
    for day in range(DAYS):
        if rng.random() >= min(0.9, 0.35 * eng):
            continue
        date = last_day + pd.Timedelta(days=day + 1)
        for _ in range(rng.poisson(0.5 + eng)):
            explore = rng.random() < EXPLORE_RATE
            if explore:
                shown = rng.choice(len(offers), LIST_SIZE, replace=False)   # random offers
            else:
                shown = np.argsort(-payout * rng.uniform(0.6, 1.4, len(payout)))[:LIST_SIZE]
            logit = (-2.2 + attract[shown] + prefs[type_idx[shown]]
                     + 0.4 * np.log(eng) - 0.25 * (pos - 1))
            clicked = rng.random(LIST_SIZE) < sigmoid(logit)
            done_p = sigmoid(1.0 - 3.5 * diff[shown] + 0.6 * np.log(eng))
            completed = clicked & (rng.random(LIST_SIZE) < done_p)
            for k in range(LIST_SIZE):
                log.append((u["user_id"], offer_ids[shown[k]], date, pos[k],
                            int(completed[k]), "explore" if explore else "payout"))

log = pd.DataFrame(log, columns=["user_id", "offer_id", "event_date", "list_position",
                                 "completed", "policy"])
log.to_csv("explore_log.csv", index=False)

print(f"Logged {len(log):,} impressions over {DAYS} days")
print(f"Offers shown at least once: {log['offer_id'].nunique()} of {len(offers)}\n")
print("Revenue cost of exploring (advertiser payout per impression):")
log = log.merge(offers[["offer_id", "advertiser_payout_usd", "offer_type"]], on="offer_id")
log["revenue"] = log["completed"] * log["advertiser_payout_usd"]
print(log.groupby("policy")["revenue"].mean().round(3).to_string(), "\n")
print("Completion rate by offer type in the EXPLORE sessions (data we never had before):")
print(log[log.policy == "explore"].groupby("offer_type")["completed"].mean()
      .map("{:.2%}".format).to_string())