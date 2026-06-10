import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import pathlib
import yaml
import logging
import scipy.stats as stats

# Configure logging
base_dir = pathlib.Path(__file__).resolve().parent.parent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(base_dir / "reports/metrics_computation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_all_metrics():
    """
    Computes all standard performance, attribution, risk, and features statistics.
    
    1. Loads dim_fund, fact_nav, and benchmark_data from SQLite database.
    2. Computes CAGR (1y, 3y, 5y), Sharpe/Sortino ratios, Beta and Alpha from OLS regressions.
    3. Traces worst peak-to-trough historical drawdowns and recoveries.
    4. Evaluates a rank-based scorecard ranking system.
    5. Saves outputs to reports/ (scorecard, alpha_beta, worst drawdown CSVs).
    6. Reloads fact_performance and fact_features SQLite database stores.
    """
    logger.info("Initializing Upgraded Performance Metrics Engine...")
    
    # Path configuration
    with open(base_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_path = base_dir / config["database"]["db_path"]
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}. Please run ETL first.")
        return
        
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Load required tables
    df_funds = pd.read_sql("SELECT * FROM dim_fund", engine)
    df_nav = pd.read_sql("SELECT * FROM fact_nav ORDER BY date_id", engine)
    df_bench = pd.read_sql("SELECT * FROM benchmark_data ORDER BY date_id", engine)
    df_scheme_perf = pd.read_sql("SELECT * FROM scheme_performance", engine)
    
    # Date formatting
    df_nav["date_id"] = pd.to_datetime(df_nav["date_id"])
    df_bench["date_id"] = pd.to_datetime(df_bench["date_id"])
    
    # Risk-free rate (6.5% standard from config)
    rf_annual = config["financials"]["risk_free_rate"]
    logger.info(f"Using Risk-Free Rate (Rf) = {rf_annual*100:.2f}%")
    
    # Load Nifty 100 for daily OLS regressions
    df_nifty100 = df_bench[df_bench["benchmark_name"] == "NIFTY100"].copy().sort_values("date_id")
    df_nifty100["ret_nifty"] = df_nifty100["value"].pct_change()
    
    performance_records = []
    features_list = []
    drawdown_records = []
    
    # Pre-calculated 5Y return fallback map
    fallback_5yr_map = dict(zip(df_scheme_perf["amfi_code"], df_scheme_perf["return_5yr_pct"] / 100.0))
    morningstar_map = dict(zip(df_scheme_perf["amfi_code"], df_scheme_perf["morningstar_rating"].fillna(3).astype(int)))
    risk_grade_map = dict(zip(df_scheme_perf["amfi_code"], df_scheme_perf["risk_grade"]))
    
    for _, fund in df_funds.iterrows():
        amfi = fund["amfi_code"]
        fund_name = fund["fund_name"]
        risk_category = fund["risk_category"]
        raw_bench = fund["benchmark"]
        
        morningstar_rating = morningstar_map.get(amfi, 3)
        risk_grade = risk_grade_map.get(amfi, risk_category)
        fallback_5yr = fallback_5yr_map.get(amfi, None)
            
        logger.info(f"Calculating metrics for: {fund_name} (AMFI: {amfi})...")
        
        # Filter NAV data for this fund
        f_nav = df_nav[df_nav["amfi_code"] == amfi].copy().sort_values("date_id")
        if len(f_nav) < 2:
            logger.warning(f"  Warning: Insufficient NAV data for {fund_name}")
            continue
            
        # 1. Compute Daily Returns
        f_nav["ret_fund"] = f_nav["nav"].pct_change()
        
        # Merge with Nifty 100 daily returns to align dates for OLS regression
        merged_nifty = pd.merge(f_nav, df_nifty100, on="date_id", suffixes=('_fund', '_nifty'))
        merged_nifty = merged_nifty.dropna()
        
        # Align with the mapped benchmark for rolling features
        benchmark_mapping = {
            'NIFTY 100 TRI': 'NIFTY100',
            'BSE 250 SmallCap TRI': 'BSE_SMALLCAP',
            'CRISIL Dynamic Gilt Index': 'CRISIL_GILT',
            'NIFTY Midcap 150 TRI': 'NIFTY_MIDCAP150',
            'CRISIL Short Term Bond Index': 'CRISIL_GILT',
            'NIFTY 500 TRI': 'NIFTY500',
            'CRISIL Liquid Fund AI Index': 'CRISIL_LIQUID',
            'NIFTY 50 TRI': 'NIFTY50',
            'NIFTY Midcap 50 TRI': 'NIFTY_MIDCAP150',
            'NIFTY Large Midcap 250 TRI': 'NIFTY500'
        }
        bench_name = benchmark_mapping.get(raw_bench, "NIFTY50")
        b_data = df_bench[df_bench["benchmark_name"] == bench_name].copy().sort_values("date_id")
        merged = pd.merge(f_nav, b_data, on="date_id", suffixes=('_fund', '_bench'))
        merged["ret_bench"] = merged["value"].pct_change()
        merged = merged.dropna()
        
        if len(merged) < 2:
            logger.warning(f"  Warning: No overlapping date points for fund {fund_name}")
            continue
            
        n_days = len(merged)
        
        # 2. CAGR Calculation Helper
        def calc_cagr(df, col, days):
            """
            Calculates compound annualized growth rate for a given dataframe column.
            
            Args:
                df (pd.DataFrame): Dataframe containing historical values.
                col (str): Column name (e.g. 'nav').
                days (int): Number of trading days in window.
                
            Returns:
                float: Annualized CAGR rate as decimal fraction.
            """
            if len(df) < 2:
                return 0.0
            beg_val = df[col].values[0]
            end_val = df[col].values[-1]
            if beg_val <= 0 or end_val <= 0:
                return 0.0
            return (end_val / beg_val) ** (252 / days) - 1
            
        cagr_1y = calc_cagr(merged.tail(252), "nav", min(len(merged), 252))
        cagr_3y = calc_cagr(merged.tail(756), "nav", min(len(merged), 756))
        
        if len(merged) >= 1260:
            cagr_5y = calc_cagr(merged.tail(1260), "nav", 1260)
        else:
            cagr_5y = fallback_5yr if fallback_5yr is not None else calc_cagr(merged, "nav", len(merged))
            
        # 3. Sharpe Ratio (using 3Y CAGR and daily returns std)
        fund_vol_daily = merged["ret_fund"].std()
        fund_vol_annual = fund_vol_daily * np.sqrt(252)
        
        if fund_vol_annual > 0:
            sharpe = (cagr_3y - rf_annual) / fund_vol_annual
        else:
            sharpe = 0.0
            
        # 4. Sortino Ratio (using 3Y CAGR and downside standard deviation)
        negative_returns = merged[merged["ret_fund"] < 0]["ret_fund"]
        if len(negative_returns) > 0:
            downside_dev_daily = np.sqrt(np.mean(negative_returns ** 2))
            downside_dev_annual = downside_dev_daily * np.sqrt(252)
        else:
            downside_dev_annual = 0.0
            
        if downside_dev_annual > 0:
            sortino = (cagr_3y - rf_annual) / downside_dev_annual
        else:
            sortino = 0.0
            
        # 5. OLS Regression Alpha & Beta vs NIFTY100 daily returns
        if len(merged_nifty) >= 10:
            slope, intercept, r_value, pvalue, stderr = stats.linregress(merged_nifty["ret_nifty"], merged_nifty["ret_fund"])
            beta_nifty = float(slope)
            alpha_nifty = float(intercept * 252)
        else:
            beta_nifty = 1.0
            alpha_nifty = 0.0
            
        # 6. Max Drawdown & Worst Drawdown Date Range
        nav_series = merged["nav"].values
        date_series = merged["date_id"].values
        peaks = np.maximum.accumulate(nav_series)
        drawdowns = np.where(peaks > 0, (nav_series - peaks) / peaks, 0.0)
        max_drawdown = float(np.abs(drawdowns.min()))
        
        if len(drawdowns) > 0 and max_drawdown > 0:
            trough_idx = drawdowns.argmin()
            trough_date = pd.to_datetime(date_series[trough_idx]).strftime("%Y-%m-%d")
            
            # Find the peak date leading to the trough
            peak_val = peaks[trough_idx]
            peak_idx = np.where(nav_series[:trough_idx+1] == peak_val)[0][-1]
            peak_date = pd.to_datetime(date_series[peak_idx]).strftime("%Y-%m-%d")
            
            # Find recovery date (first date after trough where NAV >= peak_val)
            recovery_slice = np.where(nav_series[trough_idx:] >= peak_val)[0]
            if len(recovery_slice) > 0:
                recovery_date = pd.to_datetime(date_series[trough_idx + recovery_slice[0]]).strftime("%Y-%m-%d")
            else:
                recovery_date = "Unrecovered"
        else:
            peak_date = "N/A"
            trough_date = "N/A"
            recovery_date = "N/A"
            
        performance_records.append({
            "amfi_code": amfi,
            "risk_level": risk_category,
            "benchmark_name": bench_name,
            "cagr_1y": round(float(cagr_1y), 4),
            "cagr_3y": round(float(cagr_3y), 4),
            "cagr_5y": round(float(cagr_5y), 4) if cagr_5y is not None else None,
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "beta": round(float(beta_nifty), 4),
            "alpha": round(float(alpha_nifty), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "morningstar_rating": morningstar_rating,
            "risk_grade": risk_grade
        })
        
        drawdown_records.append({
            "amfi_code": amfi,
            "fund_name": fund_name,
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "worst_drawdown_start": peak_date,
            "worst_drawdown_trough": trough_date,
            "worst_drawdown_recovery": recovery_date
        })
        
        # Precompute daily rolling features for the Feature Store (fact_features)
        merged["rolling_vol_7d"] = merged["ret_fund"].rolling(7).std() * np.sqrt(252)
        merged["rolling_vol_30d"] = merged["ret_fund"].rolling(30).std() * np.sqrt(252)
        
        rolling_mean_30d = merged["ret_fund"].rolling(30).mean() * 252
        merged["rolling_sharpe_30d"] = (rolling_mean_30d - rf_annual) / merged["rolling_vol_30d"]
        
        rolling_cov_30d = merged["ret_fund"].rolling(30).cov(merged["ret_bench"])
        rolling_var_30d = merged["ret_bench"].rolling(30).var()
        merged["rolling_beta_30d"] = rolling_cov_30d / rolling_var_30d
        
        rolling_bench_mean_30d = merged["ret_bench"].rolling(30).mean() * 252
        merged["rolling_alpha_30d"] = rolling_mean_30d - (rf_annual + merged["rolling_beta_30d"] * (rolling_bench_mean_30d - rf_annual))
        
        peaks_series = merged["nav"].cummax()
        merged["rolling_drawdown"] = (merged["nav"] - peaks_series) / peaks_series
        
        fund_features = merged[[
            "amfi_code", "date_id", "ret_fund", "rolling_vol_7d", "rolling_vol_30d", 
            "rolling_sharpe_30d", "rolling_beta_30d", "rolling_alpha_30d", "rolling_drawdown"
        ]].copy()
        
        fund_features.columns = [
            "amfi_code", "date_id", "daily_return", "rolling_vol_7d", "rolling_vol_30d", 
            "rolling_sharpe_30d", "rolling_beta_30d", "rolling_alpha_30d", "rolling_drawdown"
        ]
        
        fund_features = fund_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fund_features["date_id"] = fund_features["date_id"].dt.strftime("%Y-%m-%d")
        features_list.append(fund_features)
        
    df_perf_results = pd.DataFrame(performance_records)
    
    # 7. Compute Rank-based Scorecard
    df_score = pd.merge(df_perf_results, df_funds[["amfi_code", "expense_ratio", "fund_name", "category"]], on="amfi_code")
    
    # Percentile ranks (higher is better for CAGR, Sharpe, Alpha; lower is better for Expense, Drawdown)
    df_score["cagr_rank"] = df_score["cagr_3y"].rank(pct=True)
    df_score["sharpe_rank"] = df_score["sharpe_ratio"].rank(pct=True)
    df_score["alpha_rank"] = df_score["alpha"].rank(pct=True)
    df_score["expense_rank"] = df_score["expense_ratio"].rank(pct=True, ascending=False)
    df_score["drawdown_rank"] = df_score["max_drawdown"].rank(pct=True, ascending=False)
    
    # Composite Score = 30% Return + 25% Sharpe + 20% Alpha + 15% Expense + 10% Drawdown
    w = config["scorecard"]["weights"]
    df_score["Score"] = (
        df_score["cagr_rank"] * w.get("cagr", 0.30) +
        df_score["sharpe_rank"] * w.get("sharpe", 0.25) +
        df_score["alpha_rank"] * w.get("alpha", 0.20) +
        df_score["expense_rank"] * w.get("expense", 0.15) +
        df_score["drawdown_rank"] * w.get("drawdown", 0.10)
    ) * 100
    df_score["Score"] = df_score["Score"].round(2)
    df_score = df_score.sort_values("Score", ascending=False).reset_index(drop=True)
    df_score["Rank"] = range(1, len(df_score) + 1)
    
    # Save scorecard deliverables to root and reports folder
    export_scorecard = df_score[[
        "Rank", "amfi_code", "fund_name", "category", "cagr_3y", 
        "sharpe_ratio", "alpha", "max_drawdown", "expense_ratio", "Score"
    ]].rename(columns={
        "amfi_code": "AMFI Code",
        "fund_name": "Scheme Name",
        "category": "Category",
        "cagr_3y": "3Yr CAGR",
        "sharpe_ratio": "Sharpe Ratio",
        "alpha": "Alpha",
        "max_drawdown": "Max Drawdown",
        "expense_ratio": "Expense Ratio",
        "Score": "Composite Score"
    })
    export_scorecard.to_csv(base_dir / "reports/fund_scorecard.csv", index=False)
    logger.info("Generated fund_scorecard.csv deliverables in reports.")
    
    # Save Alpha & Beta OLS regression deliverables
    export_alpha_beta = df_score[[
        "amfi_code", "fund_name", "category", "beta", "alpha"
    ]].rename(columns={
        "amfi_code": "AMFI Code",
        "fund_name": "Scheme Name",
        "category": "Category",
        "beta": "Beta (Nifty 100 Regressed)",
        "alpha": "Alpha (Nifty 100 Regressed)"
    })
    export_alpha_beta.to_csv(base_dir / "reports/alpha_beta.csv", index=False)
    logger.info("Generated alpha_beta.csv deliverables in reports.")
    
    # Save worst drawdown date ranges as a lookup table
    df_drawdowns = pd.DataFrame(drawdown_records)
    df_drawdowns.to_csv(base_dir / "reports/worst_drawdown_ranges.csv", index=False)
    logger.info("Generated worst_drawdown_ranges.csv lookup table.")
    
    # Load fact_performance back to the DB (it matches the schema columns exactly)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fact_performance"))
    df_perf_results.to_sql("fact_performance", engine, if_exists="append", index=False)
    logger.info("Database fact_performance table successfully reloaded with regression metrics!")
    
    # Save Feature Store tables
    if features_list:
        df_features_all = pd.concat(features_list, ignore_index=True)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_features"))
        df_features_all.to_sql("fact_features", engine, if_exists="append", index=False)
        logger.info("Precomputed rolling features successfully loaded into fact_features store!")
        
    logger.info("Upgraded Performance Metrics Engine completed successfully!")

def get_daily_returns(engine, amfi_code):
    """
    Helper function to query and calculate daily percentage returns for a fund.
    
    Args:
        engine (sqlalchemy.engine.Engine): SQLalchemy database engine connection.
        amfi_code (int): AMFI code of target fund.
        
    Returns:
        pd.DataFrame: Dataframe with date_id, nav, and daily_return.
    """
    query = f"SELECT date_id, nav FROM fact_nav WHERE amfi_code = {amfi_code} ORDER BY date_id"
    df = pd.read_sql(query, engine)
    df["date_id"] = pd.to_datetime(df["date_id"])
    df["daily_return"] = df["nav"].pct_change()
    return df.dropna()

def calculate_var_cvar(daily_returns_col, confidence=0.95):
    """
    Calculates historical Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
    
    Args:
        daily_returns_col (pd.Series): Series of historical daily percentage returns.
        confidence (float): Confidence level for risk estimation (default 0.95).
        
    Returns:
        tuple: (VaR 95%, CVaR 95%) positive values representing potential loss.
    """
    if len(daily_returns_col) < 5:
        return 0.0, 0.0
    var_percentile = (1 - confidence) * 100
    var = np.percentile(daily_returns_col, var_percentile)
    cvar = daily_returns_col[daily_returns_col <= var].mean()
    return float(np.abs(var)), float(np.abs(cvar))

if __name__ == "__main__":
    compute_all_metrics()
