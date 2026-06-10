-- Database schema for Bluestock Mutual Fund Analytics Platform
-- Star Schema Structure (Updated for manager raw datasets)

-- Turn on foreign keys support in SQLite
PRAGMA foreign_keys = ON;

-- DIMENSION TABLES

-- 1. Dim Fund
CREATE TABLE IF NOT EXISTS dim_fund (
    amfi_code INTEGER PRIMARY KEY,
    fund_house TEXT NOT NULL,
    fund_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT NOT NULL,
    plan TEXT NOT NULL,
    launch_date TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    expense_ratio REAL NOT NULL, -- maps to expense_ratio_pct
    risk_category TEXT NOT NULL
);

-- 2. Dim Date (Generated calendar)
CREATE TABLE IF NOT EXISTS dim_date (
    date_id TEXT PRIMARY KEY, -- 'YYYY-MM-DD'
    date TEXT NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL, -- 0-6 (Monday-Sunday)
    is_weekend INTEGER NOT NULL -- 0 or 1
);

-- 3. Dim Investor
CREATE TABLE IF NOT EXISTS dim_investor (
    investor_id TEXT PRIMARY KEY, -- 'INVXXXXXX'
    gender TEXT NOT NULL,
    age_group TEXT NOT NULL,
    annual_income_lakh REAL NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    kyc_status TEXT NOT NULL
);


-- FACT TABLES

-- 1. Fact NAV (Daily historical NAV)
CREATE TABLE IF NOT EXISTS fact_nav (
    nav_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amfi_code INTEGER NOT NULL,
    date_id TEXT NOT NULL,
    nav REAL NOT NULL,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);

-- 2. Fact Transactions (Investor activity)
CREATE TABLE IF NOT EXISTS fact_transactions (
    transaction_id TEXT PRIMARY KEY, -- TX_XXXXXX
    investor_id TEXT NOT NULL,
    amfi_code INTEGER NOT NULL,
    date_id TEXT NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('SIP', 'Lumpsum', 'Redemption')),
    amount REAL NOT NULL,
    FOREIGN KEY (investor_id) REFERENCES dim_investor(investor_id),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);

-- 3. Fact AUM (Monthly Assets Under Management by Fund House)
CREATE TABLE IF NOT EXISTS fact_aum (
    aum_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_id TEXT NOT NULL,
    fund_house TEXT NOT NULL,
    aum REAL NOT NULL, -- in Crores
    num_schemes INTEGER NOT NULL,
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);

-- 4. Fact Performance (Calculated fund statistics)
CREATE TABLE IF NOT EXISTS fact_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amfi_code INTEGER NOT NULL,
    risk_level TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    cagr_1y REAL,
    cagr_3y REAL,
    cagr_5y REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    beta REAL,
    alpha REAL,
    max_drawdown REAL,
    morningstar_rating INTEGER,
    risk_grade TEXT,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);


-- SUPPORTING TABLES

-- Portfolio Holdings (Fund constituent holdings)
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amfi_code INTEGER NOT NULL,
    stock_symbol TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    sector TEXT NOT NULL,
    weightage REAL NOT NULL, -- weight_pct
    market_value REAL, -- market_value_cr
    current_price REAL, -- current_price_inr
    portfolio_date TEXT NOT NULL,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);

-- SIP Inflows
CREATE TABLE IF NOT EXISTS sip_inflows (
    sip_id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL, -- 'YYYY-MM'
    sip_inflow REAL NOT NULL, -- in Crores
    active_sip_accounts REAL NOT NULL, -- in Crores
    new_sip_accounts REAL NOT NULL, -- in Lakhs
    sip_aum REAL NOT NULL, -- in Lakh Crores
    yoy_growth REAL
);

-- Category Inflows
CREATE TABLE IF NOT EXISTS category_inflows (
    inflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    category TEXT NOT NULL,
    net_inflow REAL NOT NULL -- in Crores
);

-- Folio Count
CREATE TABLE IF NOT EXISTS folio_count (
    folio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    total_folios_crore REAL NOT NULL,
    equity_folios_crore REAL NOT NULL,
    debt_folios_crore REAL NOT NULL,
    hybrid_folios_crore REAL NOT NULL,
    others_folios_crore REAL NOT NULL
);

-- Benchmark Data (Daily Benchmark Values)
CREATE TABLE IF NOT EXISTS benchmark_data (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_name TEXT NOT NULL,
    date_id TEXT NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);

-- Scheme Performance (Staging table for static data)
CREATE TABLE IF NOT EXISTS scheme_performance (
    amfi_code INTEGER PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    fund_house TEXT NOT NULL,
    category TEXT NOT NULL,
    plan TEXT NOT NULL,
    return_1yr_pct REAL,
    return_3yr_pct REAL,
    return_5yr_pct REAL,
    benchmark_3yr_pct REAL,
    alpha REAL,
    beta REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    std_dev_ann_pct REAL,
    max_drawdown_pct REAL,
    aum_crore REAL,
    expense_ratio_pct REAL,
    morningstar_rating INTEGER,
    risk_grade TEXT,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_fact_nav_amfi_date ON fact_nav(amfi_code, date_id);
CREATE INDEX IF NOT EXISTS idx_fact_transactions_investor ON fact_transactions(investor_id);
CREATE INDEX IF NOT EXISTS idx_fact_transactions_amfi ON fact_transactions(amfi_code);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_amfi ON portfolio_holdings(amfi_code);
CREATE INDEX IF NOT EXISTS idx_benchmark_data_name_date ON benchmark_data(benchmark_name, date_id);

-- 5. Fact Features (Feature Store)
CREATE TABLE IF NOT EXISTS fact_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    amfi_code INTEGER NOT NULL,
    date_id TEXT NOT NULL,
    daily_return REAL,
    rolling_vol_7d REAL,
    rolling_vol_30d REAL,
    rolling_sharpe_30d REAL,
    rolling_beta_30d REAL,
    rolling_alpha_30d REAL,
    rolling_drawdown REAL,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
);

-- 6. Investor Segments (ML Clustering)
CREATE TABLE IF NOT EXISTS investor_segments (
    investor_id TEXT PRIMARY KEY,
    segment_id INTEGER,
    segment_name TEXT,
    avg_sip_amount REAL,
    annual_income REAL,
    transaction_frequency INTEGER,
    FOREIGN KEY (investor_id) REFERENCES dim_investor(investor_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_features_amfi_date ON fact_features(amfi_code, date_id);

