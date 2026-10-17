"""
Step 2: Predict each user's 30-day value (rewards_30d) from their first 3 days.

Approach:
  1. Split users by SIGNUP DATE: train on earlier users, test on later users
     (this mimics real life: you train on the past and predict the future).
  2. Compare against two simple baselines. A model is only useful if it
     beats simple rules.
  3. Measure what matters for the business: can we find the most valuable users early?
"""
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

# ---------- 1. Load the model table built in Step 1 ----------
con = duckdb.connect("freecash.duckdb", read_only=True)
df = con.execute("SELECT * FROM user_first3days").df()
con.close()

cat_cols = ["country", "platform", "acquisition_channel", "age_group"]
num_cols = ["impressions_3d", "clicks_3d", "completions_3d", "active_days_3d", "rewards_3d"]
features = cat_cols + num_cols
target = "rewards_30d"

for c in cat_cols:
    df[c] = df[c].astype("category")
df["rewards_3d"] = df["rewards_3d"].astype(float)
df[target] = df[target].astype(float)

# ---------- 2. Time-based split ----------
dates = np.sort(df["signup_date"].unique())
cutoff = dates[int(len(dates) * 0.7)]          # first 70% of signup days = train
train = df[df["signup_date"] < cutoff]
test = df[df["signup_date"] >= cutoff]
print(f"Train users: {len(train):,}   Test users: {len(test):,}   (split at signup date {pd.Timestamp(cutoff).date()})")
print(f"Users who earned $0 in 30 days: {(df[target] == 0).mean():.0%}\n")

# ---------- 3. Baselines ----------
# Baseline A: predict the average for everyone
pred_mean = np.full(len(test), train[target].mean())

# Baseline B: "30-day value = 3-day value x a fixed multiplier"
k = train[target].sum() / train["rewards_3d"].sum()
pred_mult = test["rewards_3d"] * k

# ---------- 4. The model ----------
# Most users earn $0 and a few earn a lot (very skewed), so we train on
# log(1 + value) and convert back. This stops the few big users from dominating.
model = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    max_iter=300,
    learning_rate=0.05,
    random_state=42,
)
model.fit(train[features], np.log1p(train[target]))
pred_model = np.expm1(model.predict(test[features])).clip(min=0)


# ---------- 5. Evaluate ----------
def top10_capture(pred, actual):
    """If we pick the top 10% of users by prediction,
    what share of ALL 30-day value do they actually hold?"""
    n = max(1, int(len(actual) * 0.10))
    top = np.argsort(-np.asarray(pred))[:n]
    return np.asarray(actual)[top].sum() / np.asarray(actual).sum()


y = test[target].to_numpy()
results = pd.DataFrame({
    "MAE ($, lower=better)": [
        mean_absolute_error(y, pred_mean),
        mean_absolute_error(y, pred_mult),
        mean_absolute_error(y, pred_model),
    ],
    "Top-10% users capture (higher=better)": [
        top10_capture(pred_mean + np.random.default_rng(0).random(len(y)), y),  # = random pick
        top10_capture(pred_mult, y),
        top10_capture(pred_model, y),
    ],
}, index=["A: average / random", "B: 3-day x multiplier", "Model"])
results.loc["Perfect (upper limit)"] = [0.0, top10_capture(y, y)]
print(results.round(3).to_string(), "\n")

# ---------- 6. Which features matter? ----------
imp = permutation_importance(model, test[features], np.log1p(y), n_repeats=5, random_state=42)
importance = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)
print("Feature importance (how much worse the model gets without each feature):")
print(importance.round(3).to_string(), "\n")

# ---------- 7. Check against the hidden truth ----------
# The data generator gave every user a secret 'engagement' level.
# A good model's predictions should line up with it.
truth = pd.read_csv("data/_truth_users.csv")
check = test[["user_id"]].assign(pred=pred_model).merge(truth, on="user_id")
corr = check["pred"].corr(check["_engagement"], method="spearman")
print(f"Correlation of predictions with hidden true engagement: {corr:.2f} (1.0 = perfect)")