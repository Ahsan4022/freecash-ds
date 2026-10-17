import sys
import duckdb

con = duckdb.connect("freecash.duckdb")

# Each check is a SQL query that counts BAD rows.
# 0 bad rows = PASS. Anything else = FAIL.
checks = {
    "No duplicate event_id": """
        SELECT COUNT(*) FROM (
            SELECT event_id FROM stg_events GROUP BY event_id HAVING COUNT(*) > 1
        )
    """,
    "No missing user_id / offer_id": """
        SELECT COUNT(*) FROM stg_events
        WHERE user_id IS NULL OR offer_id IS NULL
    """,
    "No negative rewards": """
        SELECT COUNT(*) FROM stg_events WHERE reward_usd < 0
    """,
    "Every 'complete' has a 'click' before it": """
        SELECT COUNT(*) FROM stg_events c
        WHERE c.event_type = 'complete'
          AND NOT EXISTS (
              SELECT 1 FROM stg_events k
              WHERE k.event_type = 'click'
                AND k.user_id  = c.user_id
                AND k.offer_id = c.offer_id
                AND k.event_ts <= c.event_ts
          )
    """,
    "No events before the user signed up": """
        SELECT COUNT(*) FROM stg_events e
        JOIN raw_users u ON u.user_id = e.user_id
        WHERE e.event_date < CAST(u.signup_date AS DATE)
    """,
    "Every event belongs to a known user": """
        SELECT COUNT(*) FROM stg_events e
        LEFT JOIN raw_users u ON u.user_id = e.user_id
        WHERE u.user_id IS NULL
    """,
}

failed = 0
for name, sql in checks.items():
    bad_rows = con.execute(sql).fetchone()[0]
    if bad_rows == 0:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  ->  {bad_rows:,} bad rows")
        failed += 1

con.close()

print()
if failed:
    print(f"{failed} check(s) failed. Fix the data before training any model!")
    sys.exit(1)   # exit code 1 = "something went wrong" (useful for automation later)
else:
    print("All checks passed. Data is safe to use.")