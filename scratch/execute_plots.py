import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import pathlib
import yaml

# Path setup
base_dir = pathlib.Path(__file__).resolve().parent.parent
with open(base_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
db_path = base_dir / config["database"]["db_path"]
engine = create_engine(f"sqlite:///{db_path}")

# 1. Generate Daily Returns Distribution Chart
df_nav = pd.read_sql("""
    SELECT n.date_id, n.amfi_code, f.fund_name, n.nav 
    FROM fact_nav n 
    JOIN dim_fund f ON n.amfi_code = f.amfi_code 
    ORDER BY date_id
""", engine)
df_nav["date_id"] = pd.to_datetime(df_nav["date_id"])

# Compute daily returns per scheme
df_nav["daily_return"] = df_nav.groupby("amfi_code")["nav"].pct_change()
df_returns = df_nav.dropna().copy()
all_returns = df_returns["daily_return"].values

# Custom styling for matplotlib
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#090d16'
plt.rcParams['axes.facecolor'] = '#090d16'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['axes.edgecolor'] = '#1e293b'
plt.rcParams['grid.color'] = '#1e293b'

# Plot return distribution histogram
plt.figure(figsize=(10, 5))
sns.histplot(all_returns, bins=100, kde=True, color="#38bdf8")
plt.title("Distribution of Daily Mutual Fund Returns (All 40 Schemes)", fontsize=12, pad=10, weight='bold', color="#f8fafc")
plt.xlabel("Daily Return", color="#94a3b8")
plt.ylabel("Frequency", color="#94a3b8")
plt.xlim(-0.05, 0.05)
plt.grid(True, linestyle="--", alpha=0.1)
plt.tight_layout()
plt.savefig(base_dir / "reports/daily_returns_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved daily_returns_distribution.png successfully.")

# 2. Generate Benchmark Comparison Chart (Top 5 Funds vs Nifty 50 and Nifty 100)
df_scorecard = pd.read_csv(base_dir / "fund_scorecard.csv")
top_5_amfi = df_scorecard.head(5)["AMFI Code"].tolist()

# Load 3-year period (2023-06-01 to 2026-05-31)
start_date = "2023-06-01"
end_date = "2026-05-31"

df_bench_data = pd.read_sql("SELECT * FROM benchmark_data ORDER BY date_id", engine)
df_bench_data["date_id"] = pd.to_datetime(df_bench_data["date_id"])

df_nav_top5 = df_nav[(df_nav["amfi_code"].isin(top_5_amfi)) & (df_nav["date_id"] >= start_date) & (df_nav["date_id"] <= end_date)].copy()
df_nifty = df_bench_data[(df_bench_data["benchmark_name"].isin(["NIFTY50", "NIFTY100"])) & (df_bench_data["date_id"] >= start_date) & (df_bench_data["date_id"] <= end_date)].copy()

# Pivot and merge
df_nav_pivot = df_nav_top5.pivot(index="date_id", columns="fund_name", values="nav").sort_index()
df_bench_pivot = df_nifty.pivot(index="date_id", columns="benchmark_name", values="value").sort_index()
df_merged_chart = pd.merge(df_nav_pivot, df_bench_pivot, left_index=True, right_index=True, how="inner")

# Normalize (Base 100)
df_normalized = (df_merged_chart / df_merged_chart.iloc[0]) * 100

# Plot
plt.figure(figsize=(12, 6.5))
colors = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]

for i, fund_col in enumerate(df_nav_pivot.columns):
    # Shorten names for the legend
    short_name = fund_col.replace("Regular Plan - Growth", "").replace("- Growth", "").strip()[:25] + "..."
    plt.plot(df_normalized.index, df_normalized[fund_col], label=short_name, color=colors[i], linewidth=1.8)

plt.plot(df_normalized.index, df_normalized["NIFTY50"], label="Nifty 50 Index (Bench)", color="#ef4444", linestyle="--", linewidth=2.0)
plt.plot(df_normalized.index, df_normalized["NIFTY100"], label="Nifty 100 Index (Bench)", color="#94a3b8", linestyle="-.", linewidth=2.0)

plt.title("Benchmark Comparison: Top 5 Funds vs Nifty 50 & Nifty 100 (3-Year Period)", fontsize=13, pad=15, weight='bold', color="#f8fafc")
plt.xlabel("Date", fontsize=11, color="#94a3b8")
plt.ylabel("Normalized Performance (Base 100)", fontsize=11, color="#94a3b8")
plt.legend(loc="upper left", fontsize=9, framealpha=0.9)
plt.grid(True, linestyle="--", alpha=0.1)
plt.tight_layout()

# Save figures
plt.savefig(base_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(base_dir / "reports/benchmark_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved benchmark_comparison.png successfully.")
