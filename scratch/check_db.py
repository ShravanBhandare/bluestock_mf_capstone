import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/bluestock_mf.db')
cursor = conn.cursor()

# Get all tables
tables = [t[0] for t in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables in database:")
for t in tables:
    cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f" - {t}: {cols}")

# Check distinct cities to see how we might classify them into tiers
cursor.execute("SELECT DISTINCT city FROM dim_investor LIMIT 20")
cities = [c[0] for c in cursor.fetchall()]
print(f"Sample cities: {cities}")

conn.close()
