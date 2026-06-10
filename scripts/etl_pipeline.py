"""
Bluestock Mutual Fund ETL (Extract, Transform, Load) Pipeline.

This script extracts raw mutual fund datasets (CSVs), cleans and transforms them 
into dimension and fact tables, and loads them into a SQLite database using a Star Schema.
"""

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import pathlib
import datetime
import logging

# Set up logging configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [ETL] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def run_etl():
    """
    Executes the complete ETL lifecycle.
    
    1. Extracts raw CSV files from data/raw/
    2. Standardizes categories, handles forward-filling of NAVs, and formats columns.
    3. Initializes SQL schema and loads cleaned datasets into SQLite database.
    """
    logger.info("Starting Bluestock Mutual Fund ETL Pipeline (Manager Datasets)...")
    
    # Path configuration
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data/raw"
    processed_dir = base_dir / "data/processed"
    db_dir = base_dir / "data/db"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old database file to prevent UNIQUE constraint failures
    db_path = db_dir / "bluestock_mf.db"
    if db_path.exists():
        try:
            db_path.unlink()
            logger.info("Cleared existing database file for a clean rebuild.")
        except Exception as e:
            logger.warning(f"Could not delete old database file: {e}")
            
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    
    # ----------------------------------------------------
    # EXTRACT PHASE
    # ----------------------------------------------------
    logger.info("Extracting manager raw datasets...")
    
    try:
        df_fund_master_raw = pd.read_csv(raw_dir / "01_fund_master.csv")
        df_nav_history_raw = pd.read_csv(raw_dir / "02_nav_history.csv")
        df_aum_raw = pd.read_csv(raw_dir / "03_aum_by_fund_house.csv")
        df_sip_raw = pd.read_csv(raw_dir / "04_monthly_sip_inflows.csv")
        df_cat_inflow_raw = pd.read_csv(raw_dir / "05_category_inflows.csv")
        df_folio_raw = pd.read_csv(raw_dir / "06_industry_folio_count.csv")
        df_scheme_perf_raw = pd.read_csv(raw_dir / "07_scheme_performance.csv")
        df_txs_raw = pd.read_csv(raw_dir / "08_investor_transactions.csv")
        df_holdings_raw = pd.read_csv(raw_dir / "09_portfolio_holdings.csv")
        df_bench_raw = pd.read_csv(raw_dir / "10_benchmark_indices.csv")
    except Exception as e:
        logger.error(f"Failed to read raw CSV files: {e}")
        raise e
        
    logger.info("Extraction successful. All 10 manager datasets loaded.")

    # ----------------------------------------------------
    # TRANSFORM PHASE
    # ----------------------------------------------------
    logger.info("Transforming and cleaning datasets...")
    
    # 1. Clean and Standardize Fund Master
    df_fund_master = df_fund_master_raw.drop_duplicates(subset=["amfi_code"]).copy()
    
    # Standardize Categories by combining Category and Sub-category
    df_fund_master["category"] = df_fund_master["category"] + " " + df_fund_master["sub_category"]
    
    # Rename expense ratio column to match database schema
    df_fund_master = df_fund_master.rename(columns={
        "scheme_name": "fund_name",
        "expense_ratio_pct": "expense_ratio"
    })
    
    # Keep only relevant columns for dim_fund
    fund_cols = ["amfi_code", "fund_house", "fund_name", "category", "sub_category", "plan", "launch_date", "benchmark", "expense_ratio", "risk_category"]
    df_fund_master = df_fund_master[fund_cols]
    
    # Save dim_fund
    df_fund_master.to_csv(processed_dir / "dim_fund.csv", index=False)
    
    # 2. Generate Calendar Dimension (dim_date)
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2026, 5, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_date = pd.DataFrame({
        "date_id": dates.strftime("%Y-%m-%d"),
        "date": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.dayofweek,
        "is_weekend": (dates.dayofweek >= 5).astype(int)
    })
    df_date.to_csv(processed_dir / "dim_date.csv", index=False)
    
    # 3. Create Investor Dimension from Transactions CSV
    df_investor = df_txs_raw[[
        "investor_id", "gender", "age_group", "annual_income_lakh", "city", "state", "kyc_status"
    ]].drop_duplicates(subset=["investor_id"]).copy()
    
    df_investor.to_csv(processed_dir / "dim_investor.csv", index=False)
    
    # 4. Clean NAV History: Full Calendar Reindexing and Forward-Filling
    df_nav_history_raw["date"] = pd.to_datetime(df_nav_history_raw["date"])
    df_nav_clean_list = []
    
    for amfi in df_nav_history_raw["amfi_code"].unique():
        sub_nav = df_nav_history_raw[df_nav_history_raw["amfi_code"] == amfi].copy()
        sub_nav = sub_nav.drop_duplicates(subset=["date"])
        sub_nav = sub_nav.set_index("date")
        
        # Reindex to full calendar range (handles weekend and holiday missing NAV data points)
        sub_nav = sub_nav.reindex(dates)
        
        # Forward fill and backward fill values
        sub_nav["nav"] = sub_nav["nav"].ffill().bfill()
        sub_nav["amfi_code"] = amfi
        
        sub_nav = sub_nav.reset_index().rename(columns={"index": "date"})
        sub_nav["date"] = sub_nav["date"].dt.strftime("%Y-%m-%d")
        df_nav_clean_list.append(sub_nav)
        
    df_nav = pd.concat(df_nav_clean_list, ignore_index=True)
    df_nav = df_nav[df_nav["nav"] > 0]
    df_nav = df_nav.rename(columns={"date": "date_id"})
    df_nav.to_csv(processed_dir / "fact_nav.csv", index=False)
    
    # 5. Clean Benchmark Indices: Reindex & Forward Fill
    df_bench_raw["date"] = pd.to_datetime(df_bench_raw["date"])
    df_bench_clean_list = []
    
    for idx_name in df_bench_raw["index_name"].unique():
        sub_bench = df_bench_raw[df_bench_raw["index_name"] == idx_name].copy()
        sub_bench = sub_bench.drop_duplicates(subset=["date"])
        sub_bench = sub_bench.set_index("date")
        
        # Reindex
        sub_bench = sub_bench.reindex(dates)
        sub_bench["close_value"] = sub_bench["close_value"].ffill().bfill()
        sub_bench["index_name"] = idx_name
        
        sub_bench = sub_bench.reset_index().rename(columns={"index": "date"})
        sub_bench["date"] = sub_bench["date"].dt.strftime("%Y-%m-%d")
        df_bench_clean_list.append(sub_bench)
        
    df_bench = pd.concat(df_bench_clean_list, ignore_index=True)
    df_bench = df_bench.rename(columns={
        "date": "date_id",
        "close_value": "value",
        "index_name": "benchmark_name"
    })
    df_bench.to_csv(processed_dir / "benchmark_data.csv", index=False)
    
    # 6. Clean Transactions
    df_txs = df_txs_raw.copy()
    # Generate unique transaction ID since raw file doesn't have one
    df_txs["transaction_id"] = [f"TX_{idx+1:06d}" for idx in range(len(df_txs))]
    df_txs = df_txs.rename(columns={
        "transaction_date": "date_id",
        "amount_inr": "amount"
    })
    
    # Filter columns to match fact_transactions schema
    tx_cols = ["transaction_id", "investor_id", "amfi_code", "date_id", "transaction_type", "amount"]
    df_txs = df_txs[tx_cols]
    df_txs = df_txs[df_txs["amount"] > 0]
    df_txs.to_csv(processed_dir / "fact_transactions.csv", index=False)
    
    # 7. Clean AUM: Reindex & Forward Fill to Monthly Frequency
    df_aum_raw["date"] = pd.to_datetime(df_aum_raw["date"])
    aum_dates = pd.date_range(start="2022-01-31", end="2026-05-31", freq="ME")
    aum_clean_list = []
    
    for amc in df_aum_raw["fund_house"].unique():
        sub_aum = df_aum_raw[df_aum_raw["fund_house"] == amc].copy()
        sub_aum = sub_aum.drop_duplicates(subset=["date"])
        sub_aum = sub_aum.set_index("date")
        
        # Reindex to all month-ends
        sub_aum = sub_aum.reindex(aum_dates)
        sub_aum["aum_crore"] = sub_aum["aum_crore"].ffill().bfill()
        sub_aum["num_schemes"] = sub_aum["num_schemes"].ffill().bfill().astype(int)
        sub_aum["fund_house"] = amc
        
        sub_aum = sub_aum.reset_index().rename(columns={"index": "date"})
        sub_aum["date"] = sub_aum["date"].dt.strftime("%Y-%m-%d")
        aum_clean_list.append(sub_aum)
        
    df_aum = pd.concat(aum_clean_list, ignore_index=True)
    df_aum = df_aum.rename(columns={
        "date": "date_id",
        "aum_crore": "aum"
    })
    df_aum = df_aum[["date_id", "fund_house", "aum", "num_schemes"]]
    df_aum = df_aum[df_aum["aum"] > 0]
    df_aum.to_csv(processed_dir / "fact_aum.csv", index=False)
    
    # 8. Clean SIP
    df_sip = df_sip_raw.rename(columns={
        "sip_inflow_crore": "sip_inflow",
        "active_sip_accounts_crore": "active_sip_accounts",
        "new_sip_accounts_lakh": "new_sip_accounts",
        "sip_aum_lakh_crore": "sip_aum",
        "yoy_growth_pct": "yoy_growth"
    })
    df_sip.to_csv(processed_dir / "sip_inflows.csv", index=False)
    
    # 9. Clean Category Inflows
    df_cat_inflow = df_cat_inflow_raw.rename(columns={
        "net_inflow_crore": "net_inflow"
    })
    df_cat_inflow.to_csv(processed_dir / "category_inflows.csv", index=False)
    
    # 10. Clean Folio Count
    df_folio = df_folio_raw.copy()
    df_folio.to_csv(processed_dir / "folio_count.csv", index=False)
    
    # 11. Clean Holdings
    df_holdings = df_holdings_raw.rename(columns={
        "weight_pct": "weightage",
        "market_value_cr": "market_value",
        "current_price_inr": "current_price"
    })
    df_holdings = df_holdings[df_holdings["weightage"] > 0]
    df_holdings.to_csv(processed_dir / "portfolio_holdings.csv", index=False)
    
    # 12. Clean Scheme Performance Static Metadata
    df_scheme_perf = df_scheme_perf_raw.copy()
    df_scheme_perf.to_csv(processed_dir / "scheme_performance.csv", index=False)
    
    logger.info("Transformations and validation checks complete.")

    # ----------------------------------------------------
    # LOAD PHASE
    # ----------------------------------------------------
    logger.info("Loading clean datasets into SQLite Star Schema database...")
    
    # Read schema.sql to initialize database tables
    sql_schema_path = base_dir / "sql/schema.sql"
    with open(sql_schema_path, "r") as f:
        schema_sql = f.read()
        
    with engine.connect() as conn:
        # Clear existing tables to allow schema rebuild if file is locked
        try:
            conn.execute(text("PRAGMA foreign_keys = OFF;"))
            r = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"))
            tables = [row[0] for row in r.fetchall()]
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {table};"))
            logger.info("Cleared existing tables from database for a clean rebuild.")
        except Exception as e:
            logger.warning(f"Could not drop existing tables: {e}")
            
        raw_conn = conn.connection
        cursor = raw_conn.cursor()
        cursor.executescript(schema_sql)
        raw_conn.commit()
        logger.info("Database schema successfully initialized from schema.sql")
        
    # Append loads into database tables
    df_fund_master.to_sql("dim_fund", engine, if_exists="append", index=False)
    df_date.to_sql("dim_date", engine, if_exists="append", index=False)
    df_investor.to_sql("dim_investor", engine, if_exists="append", index=False)
    
    df_nav.to_sql("fact_nav", engine, if_exists="append", index=False)
    df_txs.to_sql("fact_transactions", engine, if_exists="append", index=False)
    df_aum.to_sql("fact_aum", engine, if_exists="append", index=False)
    
    df_sip.to_sql("sip_inflows", engine, if_exists="append", index=False)
    df_cat_inflow.to_sql("category_inflows", engine, if_exists="append", index=False)
    df_folio.to_sql("folio_count", engine, if_exists="append", index=False)
    df_holdings.to_sql("portfolio_holdings", engine, if_exists="append", index=False)
    df_bench.to_sql("benchmark_data", engine, if_exists="append", index=False)
    df_scheme_perf.to_sql("scheme_performance", engine, if_exists="append", index=False)
    
    logger.info(f"Data successfully loaded. Database resides at: {db_path.resolve()}")
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_etl()
