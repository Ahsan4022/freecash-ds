import duckdb

# Simulates real-world data problems by inserting bad rows into raw_events.
# To get clean data back at any time, just run:  py load.py
con = duckdb.connect("freecash.duckdb")

# Problem 1: the app sent 1,000 events twice (exact copies)
con.execute("""
    INSERT INTO raw_events
    SELECT * FROM raw_events USING SAMPLE reservoir(1000 ROWS) REPEATABLE (1)
""")

# Problem 2: 300 events were re-sent a few seconds later, with the SAME event_id
con.execute("""
    INSERT INTO raw_events
    SELECT event_id, user_id, offer_id, event_type,
           event_ts + INTERVAL 5 SECOND, list_position, reward_usd
    FROM raw_events USING SAMPLE reservoir(300 ROWS) REPEATABLE (2)
""")

# Problem 3: a bug wrote 30 completions with NEGATIVE rewards
con.execute("""
    INSERT INTO raw_events
    SELECT event_id + 30000000, user_id, offer_id, 'complete',
           event_ts, list_position, -5.00
    FROM (SELECT * FROM raw_events WHERE event_type = 'complete')
    USING SAMPLE reservoir(30 ROWS) REPEATABLE (3)
""")

# Problem 4: possible fraud - 50 completions where the user never clicked the offer
con.execute("""
    INSERT INTO raw_events
    SELECT event_id + 40000000, user_id, offer_id, 'complete',
           event_ts + INTERVAL 1 MINUTE, list_position, 5.00
    FROM (SELECT * FROM raw_events WHERE event_type = 'impression')
    USING SAMPLE reservoir(50 ROWS) REPEATABLE (4)
""")

# Problem 5: 20 events from a user that doesn't exist
con.execute("""
    INSERT INTO raw_events
    SELECT event_id + 50000000, 999999, offer_id, event_type,
           event_ts, list_position, reward_usd
    FROM raw_events USING SAMPLE reservoir(20 ROWS) REPEATABLE (5)
""")

# Problem 6: 40 events with a wrong date (before the user even signed up)
con.execute("""
    INSERT INTO raw_events
    SELECT event_id + 60000000, user_id, offer_id, event_type,
           TIMESTAMP '2025-12-01 12:00:00', list_position, reward_usd
    FROM raw_events USING SAMPLE reservoir(40 ROWS) REPEATABLE (6)
""")

print("raw_events now has", f"{con.execute('SELECT COUNT(*) FROM raw_events').fetchone()[0]:,}", "rows")
print("Bad data inserted. Now run:  py transform.py  and then  py checks.py")
con.close()