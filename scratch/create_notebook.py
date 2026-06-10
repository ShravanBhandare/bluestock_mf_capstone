import json
import pathlib
import subprocess

# Paths
base_dir = pathlib.Path("D:/New folder/bluestock_mf_capstone")
notebook_path = base_dir / "Performance_Analytics.ipynb"
notebook_copy_path = base_dir / "notebooks/04_performance_analytics.ipynb"

# Define cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Performance Analytics Engine\n",
            "This notebook calculates and validates key risk and return metrics for all 40 mutual fund schemes:\n",
            "1. **Daily Return Distribution** validation\n",
            "2. **CAGR Comparisons** for 1yr, 3yr, and 5yr periods\n",
            "3. **Sharpe Ratio** ($R_f = 6.5\%$) and **Sortino Ratio** rankings\n",
            "4. **Alpha and Beta** computations via OLS linear regressions against Nifty 100 returns\n",
            "5. **Maximum Drawdown** and the worst drawdown date ranges\n",
            "6. **Composite Rank-Based Scorecard** (0–100)\n",
            "7. **Benchmark Comparison Chart** (top 5 funds vs Nifty 50 and Nifty 100) with tracking error annotations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import scipy.stats as stats\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sqlalchemy import create_engine\n",
            "import pathlib\n",
            "\n",
            "base_dir = pathlib.Path(\"D:/New folder/bluestock_mf_capstone\")\n",
            "db_path = base_dir / \"data/db/bluestock_mf.db\"\n",
            "engine = create_engine(f\"sqlite:///{db_path}\")\n",
            "print(\"Libraries imported and database connection established!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Daily Return Distribution Validation\n",
            "We query all daily NAV values, calculate the daily returns per fund, and check the overall return distribution statistics (Mean, Std Dev, Skewness, Kurtosis)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_nav = pd.read_sql(\"\"\"\n",
            "    SELECT n.date_id, n.amfi_code, f.fund_name, n.nav \n",
            "    FROM fact_nav n \n",
            "    JOIN dim_fund f ON n.amfi_code = f.amfi_code \n",
            "    ORDER BY date_id\n",
            "\"\"\", engine)\n",
            "df_nav[\"date_id\"] = pd.to_datetime(df_nav[\"date_id\"])\n",
            "\n",
            "# Compute daily returns per scheme\n",
            "df_nav[\"daily_return\"] = df_nav.groupby(\"amfi_code\")[\"nav\"].pct_change()\n",
            "df_returns = df_nav.dropna().copy()\n",
            "\n",
            "print(f\"Total daily return data points: {len(df_returns)}\")\n",
            "\n",
            "# Calculate distribution stats\n",
            "all_returns = df_returns[\"daily_return\"].values\n",
            "mean_ret = np.mean(all_returns)\n",
            "std_ret = np.std(all_returns)\n",
            "skew_ret = stats.skew(all_returns)\n",
            "kurt_ret = stats.kurtosis(all_returns)\n",
            "\n",
            "print(\"Daily Returns Distribution Statistics:\")\n",
            "print(f\"  Mean: {mean_ret:.6f}\")\n",
            "print(f\"  Std Dev: {std_ret:.6f} (Annualized Volatility: {std_ret*np.sqrt(252)*100:.2f}%)\")\n",
            "print(f\"  Skewness: {skew_ret:.4f} (Negative skewness indicates longer left tail)\")\n",
            "print(f\"  Kurtosis: {kurt_ret:.4f} (Kurtosis > 0 indicates fat-tailed distribution)\")\n",
            "\n",
            "# Plot histogram\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.histplot(all_returns, bins=100, kde=True, color=\"skyblue\")\n",
            "plt.title(\"Distribution of Daily Mutual Fund Returns (All 40 Schemes)\", fontsize=12, pad=10, weight='bold')\n",
            "plt.xlabel(\"Daily Return\")\n",
            "plt.ylabel(\"Frequency\")\n",
            "plt.xlim(-0.05, 0.05)\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
            "plt.savefig(\"reports/daily_returns_distribution.png\", dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. CAGR Comparison Table (1yr, 3yr, 5yr)\n",
            "Retrieve and display the calculated CAGR statistics across all 40 funds sorted by 3-year return."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_perf = pd.read_sql(\"\"\"\n",
            "    SELECT f.fund_name, f.category, p.* \n",
            "    FROM fact_performance p \n",
            "    JOIN dim_fund f ON p.amfi_code = f.amfi_code\n",
            "\"\"\", engine)\n",
            "\n",
            "cagr_table = df_perf[[\"fund_name\", \"category\", \"cagr_1y\", \"cagr_3y\", \"cagr_5y\"]].copy()\n",
            "cagr_table.columns = [\"Scheme Name\", \"Category\", \"1Yr CAGR\", \"3Yr CAGR\", \"5Yr CAGR\"]\n",
            "\n",
            "# Format percentages\n",
            "for col in [\"1Yr CAGR\", \"3Yr CAGR\", \"5Yr CAGR\"]:\n",
            "    cagr_table[col] = cagr_table[col].apply(lambda x: f\"{x*100:.2f}%\" if pd.notna(x) else \"N/A\")\n",
            "\n",
            "cagr_table = cagr_table.sort_values(\"3Yr CAGR\", ascending=False).reset_index(drop=True)\n",
            "pd.set_option('display.max_rows', 50)\n",
            "cagr_table"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Sharpe and Sortino Ratio Rankings\n",
            "Rank all 40 schemes on risk-adjusted ratios (Sharpe and Sortino) using the repo rate proxy $R_f = 6.5\%$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ratios_table = df_perf[[\"fund_name\", \"category\", \"sharpe_ratio\", \"sortino_ratio\"]].copy()\n",
            "ratios_table.columns = [\"Scheme Name\", \"Category\", \"Sharpe Ratio\", \"Sortino Ratio\"]\n",
            "ratios_table[\"Sharpe Rank\"] = ratios_table[\"Sharpe Ratio\"].rank(ascending=False, method=\"min\").astype(int)\n",
            "ratios_table[\"Sortino Rank\"] = ratios_table[\"Sortino Ratio\"].rank(ascending=False, method=\"min\").astype(int)\n",
            "ratios_table = ratios_table.sort_values(\"Sharpe Ratio\", ascending=False).reset_index(drop=True)\n",
            "ratios_table"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. OLS Regressions vs Nifty 100 (Alpha & Beta)\n",
            "Display the OLS regression slope (Beta) and annualized intercept (Alpha) computed against Nifty 100 daily returns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "reg_table = df_perf[[\"fund_name\", \"category\", \"beta\", \"alpha\"]].copy()\n",
            "reg_table.columns = [\"Scheme Name\", \"Category\", \"Beta (Nifty 100)\", \"Alpha (Nifty 100, Annualized)\"]\n",
            "reg_table[\"Alpha (Nifty 100, Annualized)\"] = reg_table[\"Alpha (Nifty 100, Annualized)\"].apply(lambda x: f\"{x*100:.2f}%\")\n",
            "reg_table = reg_table.sort_values(\"Alpha (Nifty 100, Annualized)\", ascending=False).reset_index(drop=True)\n",
            "reg_table"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Maximum Drawdown & Worst Drawdown Date Ranges\n",
            "Identify the worst drawdown period details (start date, trough date, and recovery status/date) for each fund."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_dd_ranges = pd.read_csv(\"reports/worst_drawdown_ranges.csv\")\n",
            "df_dd_ranges = df_dd_ranges.rename(columns={\n",
            "    \"fund_name\": \"Scheme Name\",\n",
            "    \"max_drawdown_pct\": \"Max Drawdown %\",\n",
            "    \"worst_drawdown_start\": \"Drawdown Start\",\n",
            "    \"worst_drawdown_trough\": \"Trough Date\",\n",
            "    \"worst_drawdown_recovery\": \"Recovery Date\"\n",
            "})\n",
            "df_dd_ranges = df_dd_ranges.drop(columns=[\"amfi_code\"]).sort_values(\"Max Drawdown %\", ascending=False).reset_index(drop=True)\n",
            "df_dd_ranges"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Composite Scorecard Rankings (0–100)\n",
            "Display the scorecard rankings generated based on the formula:\n",
            "$$\\text{Score} = 30\\% \\times \\text{CAGR Rank} + 25\\% \\times \\text{Sharpe Rank} + 20\\% \\times \\text{Alpha Rank} + 15\\% \\times \\text{Expense Rank (inverse)} + 10\\% \\times \\text{Drawdown Rank (inverse)}$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_scorecard = pd.read_csv(\"fund_scorecard.csv\")\n",
            "df_scorecard"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Benchmark Comparison Chart & Tracking Error\n",
            "Plot the normalized NAV paths of the top 5 funds from the scorecard against the Nifty 50 and Nifty 100 indices over a 3-year timeline, and compute their annualized tracking errors."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "top_5_amfi = df_scorecard.head(5)[\"AMFI Code\"].tolist()\n",
            "top_5_names = df_scorecard.head(5)[\"Scheme Name\"].tolist()\n",
            "\n",
            "# Load 3-year period (2023-06-01 to 2026-05-31)\n",
            "start_date = \"2023-06-01\"\n",
            "end_date = \"2026-05-31\"\n",
            "\n",
            "df_bench_data = pd.read_sql(\"SELECT * FROM benchmark_data ORDER BY date_id\", engine)\n",
            "df_bench_data[\"date_id\"] = pd.to_datetime(df_bench_data[\"date_id\"])\n",
            "\n",
            "df_nav_top5 = df_nav[(df_nav[\"amfi_code\"].isin(top_5_amfi)) & (df_nav[\"date_id\"] >= start_date) & (df_nav[\"date_id\"] <= end_date)].copy()\n",
            "df_nifty = df_bench_data[(df_bench_data[\"benchmark_name\"].isin([\"NIFTY50\", \"NIFTY100\"])) & (df_bench_data[\"date_id\"] >= start_date) & (df_bench_data[\"date_id\"] <= end_date)].copy()\n",
            "\n",
            "# Pivot and merge\n",
            "df_nav_pivot = df_nav_top5.pivot(index=\"date_id\", columns=\"fund_name\", values=\"nav\").sort_index()\n",
            "df_bench_pivot = df_nifty.pivot(index=\"date_id\", columns=\"benchmark_name\", values=\"value\").sort_index()\n",
            "\n",
            "df_merged_chart = pd.merge(df_nav_pivot, df_bench_pivot, left_index=True, right_index=True, how=\"inner\")\n",
            "\n",
            "# Normalize (Base 100)\n",
            "df_normalized = (df_merged_chart / df_merged_chart.iloc[0]) * 100\n",
            "\n",
            "# Plot\n",
            "plt.figure(figsize=(12, 6.5))\n",
            "colors = [\"#10b981\", \"#3b82f6\", \"#f59e0b\", \"#ec4899\", \"#8b5cf6\"]\n",
            "\n",
            "for i, fund_col in enumerate(df_nav_pivot.columns):\n",
            "    plt.plot(df_normalized.index, df_normalized[fund_col], label=fund_col[:32]+\"...\", color=colors[i], linewidth=1.8)\n",
            "\n",
            "plt.plot(df_normalized.index, df_normalized[\"NIFTY50\"], label=\"Nifty 50 Index (Bench)\", color=\"#ef4444\", linestyle=\"--\", linewidth=2.0)\n",
            "plt.plot(df_normalized.index, df_normalized[\"NIFTY100\"], label=\"Nifty 100 Index (Bench)\", color=\"#94a3b8\", linestyle=\"-.\", linewidth=2.0)\n",
            "\n",
            "plt.title(\"Benchmark Comparison: Top 5 Funds vs Nifty 50 & Nifty 100 (3-Year Period)\", fontsize=13, pad=15, weight='bold')\n",
            "plt.xlabel(\"Date\", fontsize=11)\n",
            "plt.ylabel(\"Normalized Performance (Base 100)\", fontsize=11)\n",
            "plt.legend(loc=\"upper left\", fontsize=9, framealpha=0.9)\n",
            "plt.grid(True, linestyle=\"--\", alpha=0.3)\n",
            "\n",
            "# Save figures\n",
            "plt.savefig(\"benchmark_comparison.png\", dpi=300, bbox_inches=\"tight\")\n",
            "plt.savefig(\"reports/benchmark_comparison.png\", dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Calculate tracking errors\n",
            "tracking_errors = []\n",
            "for fund_name in df_nav_pivot.columns:\n",
            "    fund_ret = df_merged_chart[fund_name].pct_change().dropna()\n",
            "    nifty50_ret = df_merged_chart[\"NIFTY50\"].pct_change().dropna()\n",
            "    nifty100_ret = df_merged_chart[\"NIFTY100\"].pct_change().dropna()\n",
            "    \n",
            "    # Standard deviation of active return annualized\n",
            "    te_n50 = np.std(fund_ret - nifty50_ret) * np.sqrt(252)\n",
            "    te_n100 = np.std(fund_ret - nifty100_ret) * np.sqrt(252)\n",
            "    \n",
            "    tracking_errors.append({\n",
            "        \"Scheme Name\": fund_name,\n",
            "        \"Tracking Error vs Nifty 50 (Ann)\": f\"{te_n50*100:.2f}%\",\n",
            "        \"Tracking Error vs Nifty 100 (Ann)\": f\"{te_n100*100:.2f}%\"\n",
            "    })\n",
            "\n",
            "df_te = pd.DataFrame(tracking_errors)\n",
            "df_te"
        ]
    }
]

# Write IPYNB structure
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write Performance_Analytics.ipynb
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

# Write 04_performance_analytics.ipynb
with open(notebook_copy_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print("Created Performance_Analytics.ipynb in root and notebooks/ folders successfully!")
