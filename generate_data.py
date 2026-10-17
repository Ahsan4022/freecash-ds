"""
Synthetic rewards-platform dataset (Freecash-style) for portfolio projects.

Creates four tables:
  users.csv    - one row per user (signup info)
  offers.csv   - games / apps / surveys users can complete for rewards
  events.csv   - impression -> click -> complete funnel, with list position
  cashouts.csv - users withdrawing their earned rewards

Hidden "truth" the models should be able to learn:
  - each user has a latent engagement level (drives activity, churn, completions)
  - users prefer certain offer types (game / app / survey) depending on profile
  - offers lower in the list get fewer clicks (position bias)
  - harder offers pay more but are completed less often

Run:  python generate_data.py --users 20000 --days 60
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_users(n, days, rng, start):
    countries = ["US", "DE", "GB", "FR", "CA", "BR", "IN"]
    country_p = [0.35, 0.15, 0.12, 0.08, 0.08, 0.12, 0.10]
    channels = ["organic", "tiktok", "google_ads", "referral", "meta"]
    channel_p = [0.30, 0.25, 0.20, 0.15, 0.10]

    users = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_date": [start + timedelta(days=int(d)) for d in rng.integers(0, days, n)],
        "country": rng.choice(countries, n, p=country_p),
        "platform": rng.choice(["android", "ios", "web"], n, p=[0.55, 0.30, 0.15]),
        "acquisition_channel": rng.choice(channels, n, p=channel_p),
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], n, p=[0.35, 0.35, 0.18, 0.12]),
    })

    # Latent engagement: referral/organic users are more engaged, paid social less.
    channel_boost = users["acquisition_channel"].map(
        {"organic": 0.3, "referral": 0.5, "google_ads": 0.0, "tiktok": -0.3, "meta": -0.1})
    users["_engagement"] = np.exp(rng.normal(0, 0.6, n) + channel_boost)

    # Type preference: younger users like games, older users like surveys.
    age_game = users["age_group"].map({"18-24": 1.0, "25-34": 0.5, "35-44": 0.0, "45+": -0.5})
    users["_pref_game"] = age_game + rng.normal(0, 0.5, n)
    users["_pref_survey"] = -age_game + rng.normal(0, 0.5, n)
    users["_pref_app"] = rng.normal(0, 0.5, n)
    return users


def make_offers(n, rng):
    types = rng.choice(["game", "app", "survey"], n, p=[0.5, 0.25, 0.25])
    difficulty = rng.uniform(0, 1, n)
    base_pay = np.select([types == "game", types == "app", types == "survey"], [3.0, 1.0, 0.8])
    user_reward = np.round(base_pay * (0.3 + 3 * difficulty ** 2) * rng.uniform(0.8, 1.2, n), 2)
    return pd.DataFrame({
        "offer_id": np.arange(1, n + 1),
        "offer_type": types,
        "category": np.where(types == "game",
                             rng.choice(["puzzle", "casino", "strategy", "casual", "rpg"], n),
                             np.where(types == "app",
                                      rng.choice(["finance", "shopping", "fitness"], n),
                                      "survey")),
        "difficulty": np.round(difficulty, 3),
        "user_reward_usd": user_reward,
        # Advertiser pays Almedia more than the user receives.
        "advertiser_payout_usd": np.round(user_reward * rng.uniform(1.4, 2.2, n), 2),
        "est_minutes": np.round(5 + 600 * difficulty ** 2 * rng.uniform(0.7, 1.3, n)).astype(int),
        "_attractiveness": rng.normal(0, 0.7, n),
    })


def simulate(users, offers, days, rng, start, list_size=10):
    end = start + timedelta(days=days)
    type_idx = offers["offer_type"].map({"game": 0, "app": 1, "survey": 2}).to_numpy()
    attract = offers["_attractiveness"].to_numpy()
    diff = offers["difficulty"].to_numpy()
    reward = offers["user_reward_usd"].to_numpy()
    offer_ids = offers["offer_id"].to_numpy()

    # Baseline ranking used by the "current product": highest advertiser payout first,
    # with some daily shuffling. Your ranking model should beat this later.
    base_score = offers["advertiser_payout_usd"].to_numpy()

    ev_rows, cash_rows = [], []
    # Note: itertuples() renames columns starting with "_", so use plain dicts.
    for r in users.to_dict("records"):
        uid = r["user_id"]
        eng = r["_engagement"]
        prefs = np.array([r["_pref_game"], r["_pref_app"], r["_pref_survey"]])
        day = r["signup_date"]
        balance = 0.0
        churn_p = np.clip(0.12 / eng, 0.02, 0.5)  # daily chance to leave forever
        while day < end:
            if rng.random() < min(0.9, 0.35 * eng):          # active today?
                for _ in range(rng.poisson(0.5 + eng)):       # sessions today
                    ts = day + timedelta(seconds=int(rng.integers(0, 86400)))
                    noisy = base_score * rng.uniform(0.6, 1.4, len(base_score))
                    shown = np.argsort(-noisy)[:list_size]
                    pos = np.arange(1, list_size + 1)
                    logit = (-2.2 + attract[shown] + prefs[type_idx[shown]]
                             + 0.4 * np.log(eng) - 0.25 * (pos - 1))
                    clicked = rng.random(list_size) < sigmoid(logit)
                    done_p = sigmoid(1.0 - 3.5 * diff[shown] + 0.6 * np.log(eng))
                    completed = clicked & (rng.random(list_size) < done_p)
                    for k in range(list_size):
                        oid = offer_ids[shown[k]]
                        ev_rows.append((uid, oid, "impression", ts, pos[k], 0.0))
                        if clicked[k]:
                            ev_rows.append((uid, oid, "click",
                                            ts + timedelta(seconds=int(rng.integers(2, 60))), pos[k], 0.0))
                        if completed[k]:
                            lag = timedelta(minutes=int(rng.integers(5, 60 * 24 * 3)))
                            ev_rows.append((uid, oid, "complete", ts + lag, pos[k], reward[shown[k]]))
                            balance += reward[shown[k]]
                    if balance >= 5 and rng.random() < 0.3:   # cash out at $5+
                        cash_rows.append((uid, ts, round(balance, 2),
                                          rng.choice(["paypal", "crypto", "gift_card"], p=[0.6, 0.15, 0.25])))
                        balance = 0.0
            if rng.random() < churn_p:
                break
            day += timedelta(days=1)

    events = pd.DataFrame(ev_rows, columns=["user_id", "offer_id", "event_type",
                                            "event_ts", "list_position", "reward_usd"])
    events = events[events["event_ts"] < end].sort_values("event_ts").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    cashouts = pd.DataFrame(cash_rows, columns=["user_id", "cashout_ts", "amount_usd", "method"])
    cashouts.insert(0, "cashout_id", np.arange(1, len(cashouts) + 1))
    return events, cashouts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=20000)
    p.add_argument("--offers", type=int, default=300)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    start = datetime(2026, 1, 1)
    os.makedirs(a.out, exist_ok=True)

    users = make_users(a.users, a.days, rng, start)
    offers = make_offers(a.offers, rng)
    events, cashouts = simulate(users, offers, a.days, rng, start)

    # Hidden columns (prefixed "_") are the ground truth: keep them in a separate
    # file so you can check later whether your models recovered them.
    hidden = [c for c in users.columns if c.startswith("_")]
    users.drop(columns=hidden).to_csv(f"{a.out}/users.csv", index=False)
    users[["user_id"] + hidden].to_csv(f"{a.out}/_truth_users.csv", index=False)
    offers.drop(columns=["_attractiveness"]).to_csv(f"{a.out}/offers.csv", index=False)
    offers[["offer_id", "_attractiveness"]].to_csv(f"{a.out}/_truth_offers.csv", index=False)
    events.to_csv(f"{a.out}/events.csv", index=False)
    cashouts.to_csv(f"{a.out}/cashouts.csv", index=False)

    print(f"users: {len(users):,}  offers: {len(offers):,}  "
          f"events: {len(events):,}  cashouts: {len(cashouts):,}")
    print(events["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
