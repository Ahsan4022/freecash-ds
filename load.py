import duckdb

# Create (or open) a database file called freecash.duckdb
con = duckdb.connect("freecash.duckdb")

# Load each CSV file into its own table, named raw_users, raw_offers, etc.
for t in ["users", "offers", "events", "cashouts"]:
    con.execute(f"CREATE OR REPLACE TABLE raw_{t} AS SELECT * FROM read_csv_auto('data/{t}.csv')")
    rows = con.execute(f"SELECT COUNT(*) FROM raw_{t}").fetchone()[0]
    print(f"raw_{t}: {rows:,} rows")

# First real SQL question: how many users completed at least one offer?
result = con.execute("""
    SELECT COUNT(DISTINCT user_id)
    FROM raw_events
    WHERE event_type = 'complete'
""").fetchone()[0]
print(f"Users who completed at least one offer: {result:,}")

con.close()