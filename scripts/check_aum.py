import sqlite3
import pandas as pd

conn = sqlite3.connect("data/db/bluestock_mf.db")
df_aum = pd.read_sql("SELECT date_id, fund_house, aum FROM fact_aum", conn)
df_aum['year'] = pd.to_datetime(df_aum['date_id']).dt.year

print("Yearly AUM stats:")
print(df_aum.groupby(['year', 'fund_house'])['aum'].last().reset_index().tail(15))
conn.close()
