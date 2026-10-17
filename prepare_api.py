"""
Step 4: Copy everything the API needs into the api/ folder.

The API must work on its own (e.g. on a server without our database),
so it gets its own copy of the trained model and the offer catalogue.
"""
import os
import shutil

import duckdb

os.makedirs("api", exist_ok=True)
shutil.copy("ranker_v2.joblib", "api/model.joblib")

con = duckdb.connect("freecash.duckdb", read_only=True)
con.execute("COPY (SELECT * FROM raw_offers ORDER BY offer_id) TO 'api/offers.csv' (HEADER)")
con.close()

print("Copied model to api/model.joblib and offers to api/offers.csv")