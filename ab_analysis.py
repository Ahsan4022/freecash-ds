"""
Step 3b: Analyse the A/B test properly.

  1. Sanity checks: was the split really 50/50? Were the groups similar BEFORE the test?
  2. Main result: revenue per user, with a confidence interval and p-value
  3. CUPED: use each user's pre-test revenue to remove noise -> more precise result
  4. What CAUSED the result: break revenue into its parts
  5. Guardrail: did we just pay users more?
  6. Segments: who benefits most?
"""
import duckdb
import sys
import numpy as np
import pandas as pd
from scipy import stats

ab = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "ab_results.csv")
A, B = "A_control", "B_model"

# Pre-test revenue per user (from the 60 days of logged data) - used for checks and CUPED
con = duckdb.connect("freecash.duckdb", read_only=True)
pre = con.execute("""
    SELECT e.user_id, SUM(o.advertiser_payout_usd) AS pre_revenue
    FROM stg_events e JOIN raw_offers o USING (offer_id)
    WHERE e.event_type = 'complete'
    GROUP BY e.user_id
""").df()
con.close()
ab = ab.merge(pre, on="user_id", how="left").fillna({"pre_revenue": 0})


def compare(metric_a, metric_b, n_boot=2000, seed=0):
    """Difference in means (B - A), bootstrap 95% CI, and Welch t-test p-value."""
    r = np.random.default_rng(seed)
    diffs = [r.choice(metric_b, len(metric_b)).mean() - r.choice(metric_a, len(metric_a)).mean()
             for _ in range(n_boot)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    p = stats.ttest_ind(metric_b, metric_a, equal_var=False).pvalue
    return metric_b.mean() - metric_a.mean(), low, high, p


# ---------- 1. Sanity checks ----------
print("=" * 60, "\n1. SANITY CHECKS\n" + "=" * 60)
n = ab["group"].value_counts()
srm_p = stats.chisquare(n.values).pvalue
print(f"Users per group: A={n[A]:,}  B={n[B]:,}   split check p-value = {srm_p:.2f}")
print("  (p > 0.01 means the 50/50 split worked)")
d, lo, hi, p = compare(ab.loc[ab.group == A, "pre_revenue"], ab.loc[ab.group == B, "pre_revenue"])
print(f"Pre-test revenue per user, B - A: ${d:+.2f}  (95% CI {lo:+.2f} to {hi:+.2f}, p = {p:.2f})")
print("  (before the test, both groups should be the same: CI should include 0)\n")

# ---------- 2. Main result ----------
print("=" * 60, "\n2. MAIN RESULT: revenue per user (14 days)\n" + "=" * 60)
ya, yb = ab.loc[ab.group == A, "revenue_usd"], ab.loc[ab.group == B, "revenue_usd"]
d, lo, hi, p = compare(ya, yb)
print(f"A (old ranking): ${ya.mean():.2f}    B (model): ${yb.mean():.2f}")
print(f"Difference: ${d:+.2f}  ({d / ya.mean():+.1%})   95% CI: {lo:+.2f} to {hi:+.2f}   p = {p:.2g}\n")

# ---------- 3. CUPED ----------
print("=" * 60, "\n3. CUPED (remove noise using pre-test revenue)\n" + "=" * 60)
theta = np.cov(ab["revenue_usd"], ab["pre_revenue"])[0, 1] / ab["pre_revenue"].var()
ab["revenue_cuped"] = ab["revenue_usd"] - theta * (ab["pre_revenue"] - ab["pre_revenue"].mean())
d2, lo2, hi2, p2 = compare(ab.loc[ab.group == A, "revenue_cuped"], ab.loc[ab.group == B, "revenue_cuped"])
print(f"Difference: ${d2:+.2f}   95% CI: {lo2:+.2f} to {hi2:+.2f}")
print(f"Confidence interval is {1 - (hi2 - lo2) / (hi - lo):.0%} narrower than without CUPED")
print("  (same answer, but with fewer users needed -> tests can finish faster)\n")

# ---------- 4. What caused the result? ----------
print("=" * 60, "\n4. WHAT CAUSED IT? revenue/user = sessions x CTR x completion x $/completion\n" + "=" * 60)
g = ab.groupby("group")[["sessions", "impressions", "clicks", "completions", "revenue_usd"]].sum()
users_per_group = ab.groupby("group").size()
parts = pd.DataFrame({
    "sessions per user": g["sessions"] / users_per_group,
    "click rate (CTR)": g["clicks"] / g["impressions"],
    "completions per click": g["completions"] / g["clicks"],
    "$ per completion": g["revenue_usd"] / g["completions"],
}).T
parts["B vs A"] = (parts[B] / parts[A] - 1).map("{:+.1%}".format)
print(parts.round(3).to_string())
print("  Note: the model cannot change how often users open the app, so any")
print("  difference in sessions is just random noise between the groups.\n")

# ---------- 5. Guardrail ----------
print("=" * 60, "\n5. GUARDRAIL: rewards paid to users, and margin\n" + "=" * 60)
ab["margin_usd"] = ab["revenue_usd"] - ab["rewards_usd"]
for m in ["rewards_usd", "margin_usd"]:
    d, lo, hi, p = compare(ab.loc[ab.group == A, m], ab.loc[ab.group == B, m])
    print(f"{m:<12} A=${ab.loc[ab.group == A, m].mean():.2f}  B=${ab.loc[ab.group == B, m].mean():.2f}"
          f"  diff {d:+.2f} (CI {lo:+.2f} to {hi:+.2f})")
print()

# ---------- 6. Segments ----------
print("=" * 60, "\n6. WHO BENEFITS? revenue lift by age group\n" + "=" * 60)
seg = ab.groupby(["age_group", "group"])["revenue_usd"].mean().unstack()
seg["lift"] = (seg[B] / seg[A] - 1).map("{:+.0%}".format)
print(seg.round(2).to_string())