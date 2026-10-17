import duckdb

con = duckdb.connect("freecash.duckdb")

# 1) Clean the events and QUARANTINE bad rows instead of using them.
#    Step a: remove exact duplicates (DISTINCT)
#    Step b: if the same event_id was sent twice, keep only the first one
#    Step c: label every bad row with a reason
con.execute("""
    CREATE OR REPLACE TEMP TABLE flagged_events AS
    WITH typed AS (
        SELECT DISTINCT
            event_id,
            user_id,
            offer_id,
            event_type,
            CAST(event_ts AS TIMESTAMP)       AS event_ts,
            CAST(event_ts AS DATE)            AS event_date,
            list_position,
            CAST(reward_usd AS DECIMAL(10,2)) AS reward_usd
        FROM raw_events
    ),
    dedup AS (
        SELECT * FROM typed
        QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_ts) = 1
    )
    SELECT
        d.*,
        CASE
            WHEN u.user_id IS NULL                              THEN 'unknown_user'
            WHEN d.reward_usd < 0                               THEN 'negative_reward'
            WHEN d.event_date < CAST(u.signup_date AS DATE)     THEN 'before_signup'
            WHEN d.event_type = 'complete' AND NOT EXISTS (
                SELECT 1 FROM dedup k
                WHERE k.event_type = 'click'
                  AND k.user_id  = d.user_id
                  AND k.offer_id = d.offer_id
                  AND k.event_ts <= d.event_ts
            )                                                   THEN 'complete_without_click'
        END AS reject_reason
    FROM dedup d
    LEFT JOIN raw_users u ON u.user_id = d.user_id
""")

# Good rows -> stg_events (used by everything downstream)
con.execute("""
    CREATE OR REPLACE TABLE stg_events AS
    SELECT * EXCLUDE (reject_reason) FROM flagged_events
    WHERE reject_reason IS NULL
""")

# Bad rows -> rejected_events (kept so we can investigate them, not silently deleted)
con.execute("""
    CREATE OR REPLACE TABLE rejected_events AS
    SELECT * FROM flagged_events
    WHERE reject_reason IS NOT NULL
""")

print("Rejected rows by reason:")
con.sql("SELECT reject_reason, COUNT(*) AS n FROM rejected_events GROUP BY 1 ORDER BY 2 DESC").show()

# 2) user_daily: one row per user per day
con.execute("""
    CREATE OR REPLACE TABLE user_daily AS
    SELECT
        user_id,
        event_date,
        COUNT(*) FILTER (WHERE event_type = 'impression') AS impressions,
        COUNT(*) FILTER (WHERE event_type = 'click')      AS clicks,
        COUNT(*) FILTER (WHERE event_type = 'complete')   AS completions,
        SUM(reward_usd)                                   AS rewards_usd
    FROM stg_events
    GROUP BY user_id, event_date
""")

# 3) offer_stats: how well each offer performs
#    ctr = click rate (clicks / impressions)
#    completion_rate = completions / clicks
con.execute("""
    CREATE OR REPLACE TABLE offer_stats AS
    SELECT
        o.offer_id,
        o.offer_type,
        o.difficulty,
        o.user_reward_usd,
        COUNT(*) FILTER (WHERE e.event_type = 'impression') AS impressions,
        COUNT(*) FILTER (WHERE e.event_type = 'click')      AS clicks,
        COUNT(*) FILTER (WHERE e.event_type = 'complete')   AS completions,
        ROUND(1.0 * COUNT(*) FILTER (WHERE e.event_type = 'click')
              / NULLIF(COUNT(*) FILTER (WHERE e.event_type = 'impression'), 0), 3) AS ctr,
        ROUND(1.0 * COUNT(*) FILTER (WHERE e.event_type = 'complete')
              / NULLIF(COUNT(*) FILTER (WHERE e.event_type = 'click'), 0), 3)      AS completion_rate
    FROM raw_offers o
    LEFT JOIN stg_events e ON e.offer_id = o.offer_id
    GROUP BY ALL
""")

# 4) user_first3days: what each user did in their first 3 days (the model INPUT)
#    plus rewards_30d = what they earned in their first 30 days (what we PREDICT).
#    We only keep users who signed up at least 30 days before the data ends,
#    otherwise we can't know their full 30-day value yet.
con.execute("""
    CREATE OR REPLACE TABLE user_first3days AS
    WITH u AS (
        SELECT user_id, CAST(signup_date AS DATE) AS signup_date,
               country, platform, acquisition_channel, age_group
        FROM raw_users
    ),
    e AS (
        SELECT e.user_id, e.event_type, e.event_date, e.reward_usd,
               date_diff('day', u.signup_date, e.event_date) AS day_n
        FROM stg_events e
        JOIN u ON u.user_id = e.user_id
    )
    SELECT
        u.*,
        COUNT(*) FILTER (WHERE e.day_n < 3 AND e.event_type = 'impression') AS impressions_3d,
        COUNT(*) FILTER (WHERE e.day_n < 3 AND e.event_type = 'click')      AS clicks_3d,
        COUNT(*) FILTER (WHERE e.day_n < 3 AND e.event_type = 'complete')   AS completions_3d,
        COUNT(DISTINCT e.event_date) FILTER (WHERE e.day_n < 3)             AS active_days_3d,
        COALESCE(SUM(e.reward_usd) FILTER (WHERE e.day_n < 3), 0)           AS rewards_3d,
        COALESCE(SUM(e.reward_usd) FILTER (WHERE e.day_n < 30), 0)          AS rewards_30d
    FROM u
    LEFT JOIN e ON e.user_id = u.user_id
    WHERE u.signup_date + INTERVAL 30 DAY <= (SELECT MAX(event_date) FROM stg_events)
    GROUP BY ALL
""")

for t in ["stg_events", "user_daily", "offer_stats", "user_first3days"]:
    print(f"{t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:,} rows")

print("\nTop 5 offers by click rate:")
con.sql("SELECT offer_id, offer_type, difficulty, ctr, completion_rate FROM offer_stats ORDER BY ctr DESC LIMIT 5").show()

print("Sample of the model table:")
con.sql("SELECT user_id, acquisition_channel, clicks_3d, completions_3d, rewards_3d, rewards_30d FROM user_first3days ORDER BY rewards_30d DESC LIMIT 5").show()

con.close()