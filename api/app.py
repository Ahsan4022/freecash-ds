"""
Step 4: Offer Ranking API.

Send a user's profile, get back the offers to show them, best first.
Run locally:   py -m uvicorn api.app:app --reload
Then open:     http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
saved = joblib.load(HERE / "model.joblib")        # trained in train_ranker.py --explore
offers = pd.read_csv(HERE / "offers.csv")          # the offer catalogue

app = FastAPI(
    title="Offer Ranking API",
    description=(
        "Ranks offers (games, apps, surveys) for a user of a Freecash-style rewards "
        "platform by expected revenue = P(complete) x advertiser payout. "
        "Part of the project: https://github.com/Ahsan4022/freecash-ds"
    ),
    version="1.0",
)


class UserProfile(BaseModel):
    country: Literal["US", "DE", "GB", "FR", "CA", "BR", "IN"] = "DE"
    platform: Literal["android", "ios", "web"] = "android"
    acquisition_channel: Literal["organic", "tiktok", "google_ads", "referral", "meta"] = "organic"
    age_group: Literal["18-24", "25-34", "35-44", "45+"] = "25-34"
    n: int = Field(10, ge=1, le=50, description="How many offers to return")


@app.get("/")
def home():
    """Quick check that the API is running."""
    return {"status": "ok", "offers_in_catalogue": len(offers), "try_it": "/docs"}


@app.post("/rank")
def rank(user: UserProfile):
    """Return the top-n offers for this user, ranked by expected revenue."""
    pairs = offers.copy()
    for field in ["country", "platform", "acquisition_channel", "age_group"]:
        pairs[field] = getattr(user, field)
    pairs["list_position"] = 1                      # judge every offer as if shown at the top
    pairs["offer_rate"] = pairs["offer_id"].map(saved["offer_rate"]).fillna(saved["prior"])
    for c, dt in saved["dtypes"].items():
        pairs[c] = pd.Categorical(pairs[c], categories=dt.categories)

    p_complete = saved["model"].predict_proba(pairs[saved["features"]])[:, 1]
    ranked = offers.assign(
        p_complete=p_complete.round(4),
        expected_revenue_usd=(p_complete * offers["advertiser_payout_usd"]).round(3),
    ).sort_values("expected_revenue_usd", ascending=False).head(user.n)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    columns = ["rank", "offer_id", "offer_type", "category", "difficulty",
               "user_reward_usd", "advertiser_payout_usd", "p_complete", "expected_revenue_usd"]
    return {
        "user": user.model_dump(exclude={"n"}),
        "offers": ranked[columns].to_dict(orient="records"),
    }