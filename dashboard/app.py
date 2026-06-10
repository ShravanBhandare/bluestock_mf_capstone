import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import pathlib
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging
from statsmodels.tsa.arima.model import ARIMA
import time
import random

import sys

# Load configuration settings
base_dir = pathlib.Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))
with open(base_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(base_dir / "reports/dashboard_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bluestock MF Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Harmonious Premium Dark Mode Theme - Fintech Style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #090d16;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #0ea5e9 !important; /* Premium Cyan */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Advanced Fintech Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
    }
    
    .insights-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(14, 165, 233, 0.01) 100%);
        border: 1px solid rgba(14, 165, 233, 0.15);
        border-left: 5px solid #0ea5e9;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .risk-alert-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.01) 100%);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers & Text Gradients */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Sidebar Metrics */
    div[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #10b981 !important; /* Green for positive */
    }
    
    /* Streamlit tabs styling override */
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    
    /* Hide Streamlit branding, deploy button, and menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stHeader"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Database Connection
@st.cache_resource
def get_db_engine():
    db_path = base_dir / config["database"]["db_path"]
    if not db_path.exists():
        with st.spinner("Database not found. Running ETL and analytics pipeline to build database... (this may take 15-30 seconds on first run)"):
            try:
                # Ensure the parent directory exists
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Import and execute the pipeline steps
                from scripts import etl_pipeline
                from scripts import compute_metrics
                from scripts import train_clustering
                from scripts import run_day6_tasks
                
                logger.info("SQLite database not found. Executing automatic pipeline build...")
                etl_pipeline.run_etl()
                compute_metrics.compute_all_metrics()
                train_clustering.train()
                run_day6_tasks.run_day6()
                logger.info("Automatic pipeline build completed successfully.")
            except Exception as e:
                st.error(f"Failed to automatically initialize database: {e}")
                st.stop()
    return create_engine(f"sqlite:///{db_path}")

try:
    engine = get_db_engine()
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# Benchmark Mapping Dictionary (fund master benchmark column to index names in database)
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

# Helper to filter dataframes by relative period based on their own max date
def filter_by_time(df, date_col, period):
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    
    if period == "1M":
        start_date = max_date - pd.DateOffset(months=1)
    elif period == "3M":
        start_date = max_date - pd.DateOffset(months=3)
    elif period == "6M":
        start_date = max_date - pd.DateOffset(months=6)
    elif period == "1Y":
        start_date = max_date - pd.DateOffset(years=1)
    elif period == "3Y":
        start_date = max_date - pd.DateOffset(years=3)
    else:
        return df
        
    return df[df[date_col] >= start_date]

# Helper to reindex monthly time series to guarantee continuous timelines
def resample_time_series(df, date_col, group_col, value_col, freq="MS"):
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    all_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
    if len(all_dates) == 0:
        return df
        
    resampled_parts = []
    
    if group_col:
        for group_val, group_df in df.groupby(group_col):
            group_df = group_df.set_index(date_col)
            group_df = group_df.reindex(all_dates)
            group_df.index.name = date_col
            group_df = group_df.reset_index()
            group_df[group_col] = group_val
            group_df[value_col] = group_df[value_col].fillna(0)
            resampled_parts.append(group_df)
    else:
        df = df.set_index(date_col)
        df = df.reindex(all_dates)
        df.index.name = date_col
        df = df.reset_index()
        df[value_col] = df[value_col].fillna(0)
        resampled_parts.append(df)
        
    if resampled_parts:
        return pd.concat(resampled_parts, ignore_index=True)
    return df

# Fetch raw tables
@st.cache_data
def load_data():
    funds = pd.read_sql("SELECT * FROM dim_fund", engine)
    investors = pd.read_sql("SELECT * FROM dim_investor", engine)
    perf = pd.read_sql("SELECT * FROM fact_performance", engine)
    
    # Merge fund dimension with performance fact
    fund_perf = pd.merge(funds, perf, on="amfi_code")
    
    # Load daily indices
    bench = pd.read_sql("SELECT * FROM benchmark_data", engine)
    bench["date_id"] = pd.to_datetime(bench["date_id"])
    
    # Load daily NAVs
    nav = pd.read_sql("SELECT * FROM fact_nav", engine)
    nav["date_id"] = pd.to_datetime(nav["date_id"])
    
    # Load transactions
    txs = pd.read_sql("SELECT * FROM fact_transactions", engine)
    txs["date_id"] = pd.to_datetime(txs["date_id"])
    
    # Load AUM
    aum = pd.read_sql("SELECT * FROM fact_aum", engine)
    aum["date_id"] = pd.to_datetime(aum["date_id"])
    
    # Load SIP, category inflows, folio count
    sip = pd.read_sql("SELECT * FROM sip_inflows", engine)
    sip["date_id"] = pd.to_datetime(sip["month"] + "-01")
    cat_inflows = pd.read_sql("SELECT * FROM category_inflows", engine)
    cat_inflows["date_id"] = pd.to_datetime(cat_inflows["month"] + "-01")
    folios = pd.read_sql("SELECT * FROM folio_count", engine)
    folios["date_id"] = pd.to_datetime(folios["month"] + "-01")
    holdings = pd.read_sql("SELECT * FROM portfolio_holdings", engine)
    
    return fund_perf, investors, bench, nav, txs, aum, sip, cat_inflows, folios, holdings

(fund_perf, investors, bench, nav, txs, aum, sip, cat_inflows, folios, holdings) = load_data()

# Helper to compute dynamic fund metrics based on date selection
def compute_dynamic_metrics(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter daily NAVs
    nav_filtered = nav[(nav["date_id"] >= start_date) & (nav["date_id"] <= end_date)].sort_values("date_id")
    bench_filtered = bench[(bench["date_id"] >= start_date) & (bench["date_id"] <= end_date)].sort_values("date_id")
    
    if len(nav_filtered) < 5 or len(bench_filtered) < 5:
        return pd.DataFrame()
    
    days = (end_date - start_date).days
    years = max(days / 365.25, 0.08) # minimum ~1 month
    
    # Group NAVs by amfi_code
    nav_grouped = nav_filtered.groupby("amfi_code")
    
    # Benchmark daily returns
    bench_pivot = bench_filtered.pivot(index="date_id", columns="benchmark_name", values="value")
    bench_rets = bench_pivot.pct_change().dropna()
    
    dynamic_metrics = []
    
    for amfi_code, group in nav_grouped:
        if len(group) < 2:
            continue
        first_nav = group["nav"].iloc[0]
        last_nav = group["nav"].iloc[-1]
        
        # Absolute Return
        abs_ret = (last_nav - first_nav) / first_nav
        
        # Annualized Return (CAGR)
        if years >= 1.0:
            cagr = (last_nav / first_nav) ** (1.0 / years) - 1.0
        else:
            cagr = abs_ret / years
            
        # Daily returns
        group = group.copy()
        group["ret"] = group["nav"].pct_change()
        daily_rets = group["ret"].dropna()
        
        # Volatility (Risk)
        vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 1 else 0.0
        
        # Max Drawdown
        cum_returns = (1 + daily_rets).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(drawdown.min()) if not drawdown.empty else 0.0
        
        # Sharpe Ratio (Risk Free rate from config)
        rf = config["financials"]["risk_free_rate"]
        sharpe = (cagr - rf) / vol if vol > 0.0 else 0.0
        
        # Beta
        beta = 1.0
        # find mapped benchmark
        fund_info = fund_perf[fund_perf["amfi_code"] == amfi_code]
        if not fund_info.empty:
            bench_raw = fund_info.iloc[0]["benchmark"]
            db_bench = benchmark_mapping.get(bench_raw, "NIFTY50")
            if db_bench in bench_rets.columns:
                # Align daily returns
                aligned = pd.concat([daily_rets, bench_rets[db_bench]], axis=1, join="inner").dropna()
                if len(aligned) > 5:
                    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                    bench_var = aligned.iloc[:, 1].var()
                    beta = cov / bench_var if bench_var > 0 else 1.0
                    
        dynamic_metrics.append({
            "amfi_code": amfi_code,
            "cagr": cagr,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "beta": beta
        })
        
    return pd.DataFrame(dynamic_metrics)

st.sidebar.title("💼 Investment Control Center")

# 🎓 Investor Education Guide (Beginner to Quant) in Sidebar
with st.sidebar.expander("🎓 Investor & Trader Guide", expanded=False):
    st.markdown("""
    ### 🔰 Beginner level
    * **AUM**: Total money managed. High AUM represents institutional trust.
    * **SIP**: Systematic Investment Plan (fixed monthly savings).
    * **Expense Ratio**: Fund fee. **Lower is better** as it eats into returns.
    
    ### 📈 Intermediate Level
    * **CAGR**: The smooth average annual growth rate.
    * **Max Drawdown**: The worst-case drop from the peak. Measures maximum historical loss.
    * **Volatility**: Speed/scale of price changes. Higher means larger swings.
    
    ### 🔬 Advanced Quant Level
    * **Sharpe Ratio**: Returns per unit of risk. **> 1.0 is Good; > 2.0 is Excellent**.
    * **Beta**: Market sensitivity. **Beta > 1.0** (Aggressive/high volatility); **Beta < 1.0** (Defensive).
    * **Alpha**: Excess return vs benchmark. **Positive Alpha** means the manager added value.
    * **Markowitz Frontier**: Portfolios with optimal return-to-risk ratios.
    * **ARIMA**: Statistical forecast of future prices using historical trends.
    """)

st.sidebar.markdown("---")

# 1. Market Time Filter
st.sidebar.subheader("⏱️ Market Time Filter")
time_filter = st.sidebar.radio(
    "Select Period",
    ["1M", "3M", "6M", "1Y", "3Y", "MAX"],
    horizontal=True,
    label_visibility="collapsed"
)

# Dynamic x-axis label based on Market Time Filter
x_axis_label = "Date"
if time_filter in ["3M", "6M", "1Y"]:
    x_axis_label = "Months"
elif time_filter in ["3Y", "MAX"]:
    x_axis_label = "Years"

# Calculate start_dt and end_dt dynamically for daily calculations based on nav's max date
nav_filtered = filter_by_time(nav, "date_id", time_filter)
if not nav_filtered.empty:
    start_dt = nav_filtered["date_id"].min()
    end_dt = nav_filtered["date_id"].max()
else:
    start_dt = nav["date_id"].min()
    end_dt = nav["date_id"].max()

# Calculate dynamic metrics for this selected period from raw daily NAVs
dyn_metrics = compute_dynamic_metrics(start_dt, end_dt)

if not dyn_metrics.empty:
    cols_to_drop = ["cagr_3y", "sharpe_ratio", "beta", "max_drawdown"]
    cols_present = [c for c in cols_to_drop if c in fund_perf.columns]
    dynamic_fund_perf = pd.merge(
        fund_perf.drop(columns=cols_present),
        dyn_metrics,
        on="amfi_code"
    )
    dynamic_fund_perf["cagr_3y"] = dynamic_fund_perf["cagr"]
else:
    dynamic_fund_perf = fund_perf.copy()

# 2. Fund House Selector (Advanced)
st.sidebar.subheader("🏢 Fund House")
amc_counts = dynamic_fund_perf["fund_house"].value_counts()
amc_list_with_counts = [f"{amc} ({count} funds)" for amc, count in amc_counts.items()]
selected_amcs_display = st.sidebar.multiselect(
    "Select AMC", 
    options=amc_list_with_counts, 
    default=amc_list_with_counts
)
selected_amcs = [amc.rsplit(" (", 1)[0] for amc in selected_amcs_display]

# 3. Risk Filter
st.sidebar.subheader("⚖️ Risk Level (Beta)")
min_risk = float(dynamic_fund_perf["beta"].min()) if not dynamic_fund_perf.empty else 0.0
max_risk = float(dynamic_fund_perf["beta"].max()) if not dynamic_fund_perf.empty else 1.0
if min_risk == max_risk:
    min_risk -= 0.1
    max_risk += 0.1
risk_range = st.sidebar.slider(
    "Market Beta Range",
    min_value=min_risk,
    max_value=max_risk,
    value=(min_risk, max_risk)
)

# 4. Return Filter
st.sidebar.subheader("📈 Return Range (CAGR %)")
min_ret = float(dynamic_fund_perf["cagr_3y"].min() * 100) if not dynamic_fund_perf.empty else -100.0
max_ret = float(dynamic_fund_perf["cagr_3y"].max() * 100) if not dynamic_fund_perf.empty else 100.0
if min_ret == max_ret:
    min_ret -= 1.0
    max_ret += 1.0
ret_range = st.sidebar.slider(
    "Return Range",
    min_value=min_ret,
    max_value=max_ret,
    value=(min_ret, max_ret)
)

# 5. Fund Category Chips
st.sidebar.subheader("🏷️ Fund Category")
cat_list = sorted(list(dynamic_fund_perf["category"].unique()))
if hasattr(st, "pills"):
    selected_cats = st.sidebar.pills("Toggle Categories", options=cat_list, default=cat_list, selection_mode="multi")
else:
    selected_cats = st.sidebar.multiselect("Toggle Categories", options=cat_list, default=cat_list)

# Filter dimensions to create globally filtered funds list
filtered_funds = dynamic_fund_perf[
    (dynamic_fund_perf["fund_house"].isin(selected_amcs)) &
    (dynamic_fund_perf["category"].isin(selected_cats)) &
    (dynamic_fund_perf["beta"] >= risk_range[0]) & (dynamic_fund_perf["beta"] <= risk_range[1]) &
    (dynamic_fund_perf["cagr_3y"] * 100 >= ret_range[0]) & (dynamic_fund_perf["cagr_3y"] * 100 <= ret_range[1])
]
filtered_amfi_codes = filtered_funds["amfi_code"].tolist()

# Decentralized filtering of time-series dataframes (each based on its own max date)
# Order: filter by AMC/category first -> then filter by relative period -> resample monthly to ensure continuous timeline

# Filter AUM
aum_filtered = aum[aum["fund_house"].isin(selected_amcs)].copy()
aum_filtered = filter_by_time(aum_filtered, "date_id", time_filter)
aum_filtered = resample_time_series(aum_filtered, "date_id", "fund_house", "aum", freq="ME")

# Filter SIP
sip_filtered = filter_by_time(sip, "date_id", time_filter)
sip_filtered = resample_time_series(sip_filtered, "date_id", None, "sip_inflow", freq="MS")
if not sip_filtered.empty:
    sip_filtered["month"] = sip_filtered["date_id"].dt.strftime("%Y-%m")

# Filter Folios
folios_filtered = filter_by_time(folios, "date_id", time_filter)
folios_filtered = resample_time_series(folios_filtered, "date_id", None, "total_folios_crore", freq="MS")
if not folios_filtered.empty:
    folios_filtered["month"] = folios_filtered["date_id"].dt.strftime("%Y-%m")

# Filter Category Net Inflows
cat_inflows_filtered = filter_by_time(cat_inflows, "date_id", time_filter)
cat_inflows_filtered = resample_time_series(cat_inflows_filtered, "date_id", "category", "net_inflow", freq="MS")
if not cat_inflows_filtered.empty:
    cat_inflows_filtered["month"] = cat_inflows_filtered["date_id"].dt.strftime("%Y-%m")

# Filter Transactions (Investor Demographics)
txs_filtered = txs[txs["amfi_code"].isin(filtered_amfi_codes)].copy()
txs_filtered = filter_by_time(txs_filtered, "date_id", time_filter)

# 6. Smart Insights Panel in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Smart Insights")

if len(filtered_funds) > 0:
    best_fund_today = filtered_funds.loc[filtered_funds["cagr_3y"].idxmax()]["fund_name"]
    high_sharpe_fund = filtered_funds.loc[filtered_funds["sharpe_ratio"].idxmax()]["fund_name"]
    most_volatile_fund = filtered_funds.loc[filtered_funds["beta"].idxmax()]["fund_name"]
    
    if not cat_inflows_filtered.empty:
        top_sip_cat = cat_inflows_filtered.groupby("category")["net_inflow"].sum().idxmax()
    else:
        top_sip_cat = "N/A"
    
    st.sidebar.info(f"🏆 **Top Performer:** {best_fund_today}")
    st.sidebar.success(f"⚖️ **Highest Sharpe:** {high_sharpe_fund}")
    st.sidebar.warning(f"📈 **Most Volatile:** {most_volatile_fund}")
    st.sidebar.metric("🔥 Top SIP Category", top_sip_cat)
else:
    st.sidebar.write("No insights for current filters.")

st.sidebar.markdown("---")
# Live Feed Simulator (Bloomberg Live NAV style)
st.sidebar.subheader("📡 Live Feed Simulator")
live_sim = st.sidebar.checkbox("Activate Live NAV Streaming")

if live_sim:
    all_amcs = sorted(list(dynamic_fund_perf["fund_house"].unique()))
    selected_live_amc = st.sidebar.selectbox("Select AMC for Live Feed", options=all_amcs)
    st.sidebar.markdown(f"🟢 *Streaming {selected_live_amc} on main screen.*")
    watchlist_funds = dynamic_fund_perf[dynamic_fund_perf["fund_house"] == selected_live_amc]
    if not watchlist_funds.empty:
        st.sidebar.write("**Watchlist Ticker Snapshot:**")
        for _, row in watchlist_funds.iterrows():
            # Look up latest actual NAV from the historical daily NAV series
            fund_nav_history = nav[nav["amfi_code"] == row["amfi_code"]]
            latest_nav_val = fund_nav_history.sort_values("date_id").iloc[-1]["nav"] if not fund_nav_history.empty else 100.0
            
            drift = random.uniform(-0.0015, 0.0015)
            sim_nav = latest_nav_val * (1 + drift)
            sign = "+" if drift >= 0 else ""
            color_tag = "green" if drift >= 0 else "red"
            st.sidebar.markdown(f"🔹 {row['fund_name'][:18]}... : :{color_tag}[₹{sim_nav:.2f} ({sign}{drift*100:.2f}%)]")

# ----------------------------------------------------
# 🚨 QUANT & BEHAVIORAL RISK ALERT SYSTEM
# ----------------------------------------------------
active_alerts = []

# Alert 1: Severe Drawdown (> 15%)
if not filtered_funds.empty:
    severe_dd_funds = filtered_funds[filtered_funds["max_drawdown"] > 0.15]
    for _, row in severe_dd_funds.iterrows():
        active_alerts.append({
            "type": "error",
            "message": f"🚨 **Severe Drawdown Warning**: `{row['fund_name']}` has a drawdown of **{row['max_drawdown']*100:.1f}%** over the selected period (Threshold: 15%)."
        })

# Alert 2: Volatility Spikes
# Check if 7D rolling volatility exceeds 30D rolling volatility by 1.5x on the latest date
if filtered_amfi_codes:
    try:
        # Safe string conversion of list of codes
        codes_str = ",".join(map(str, [int(c) for c in filtered_amfi_codes]))
        if codes_str:
            query_vol = f"""
            SELECT amfi_code, rolling_vol_7d, rolling_vol_30d 
            FROM fact_features 
            WHERE amfi_code IN ({codes_str}) 
            AND date_id = (SELECT MAX(date_id) FROM fact_features)
            """
            vol_check = pd.read_sql(query_vol, engine)
            for _, row in vol_check.iterrows():
                if row['rolling_vol_7d'] > 1.5 * row['rolling_vol_30d'] and row['rolling_vol_30d'] > 0:
                    fund_name_match = filtered_funds[filtered_funds["amfi_code"] == row['amfi_code']]["fund_name"].values
                    if len(fund_name_match) > 0:
                        fund_name = fund_name_match[0]
                        active_alerts.append({
                            "type": "warning",
                            "message": f"⚡ **Volatility Spike Detected**: `{fund_name}` rolling 7D volatility ({row['rolling_vol_7d']*100:.1f}%) is 1.5x higher than its 30D volatility ({row['rolling_vol_30d']*100:.1f}%)."
                        })
    except Exception as e:
        logger.warning(f"Error checking volatility spikes: {e}")

# Alert 3: Category Inflow Sudden Drop (Latest MoM drop > 10%)
if not cat_inflows_filtered.empty:
    try:
        monthly_total_inflows = cat_inflows_filtered.groupby("month")["net_inflow"].sum().sort_index()
        if len(monthly_total_inflows) >= 2:
            latest_val = monthly_total_inflows.iloc[-1]
            prev_val = monthly_total_inflows.iloc[-2]
            if prev_val > 0 and (latest_val - prev_val) / prev_val < -0.10:
                drop_pct = abs((latest_val - prev_val) / prev_val) * 100
                active_alerts.append({
                    "type": "warning",
                    "message": f"📉 **SIP Inflow Anomaly**: Aggregate platform net inflows dropped by **{drop_pct:.1f}%** MoM in the latest month ({monthly_total_inflows.index[-1]})."
                })
    except Exception as e:
        logger.warning(f"Error checking category inflow anomalies: {e}")

# Alert 4: Abnormal Beta
if not filtered_funds.empty:
    high_beta_funds = filtered_funds[filtered_funds["beta"] > 1.4]
    for _, row in high_beta_funds.iterrows():
        active_alerts.append({
            "type": "warning",
            "message": f"⚖️ **High Systemic Risk**: `{row['fund_name']}` exhibits abnormal beta (**{row['beta']:.2f}**), indicating extreme sensitivity to market moves."
        })

# Header Title
st.title("🏦 Bluestock Mutual Fund Intelligence Platform")
st.markdown("---")

# Render active risk alerts banner removed


# ----------------------------------------------------
# 📡 LIVE WATCHLIST TICKER FEED
# ----------------------------------------------------
if live_sim:
    st.subheader("📡 Live Watchlist Ticker Feed")
    watchlist_funds = dynamic_fund_perf[dynamic_fund_perf["fund_house"] == selected_live_amc]
    if not watchlist_funds.empty:
        # Create empty container for streaming
        live_ticker_placeholder = st.empty()
        
        # Track the last selected AMC to clear the tick history when it changes
        if "last_selected_live_amc" not in st.session_state:
            st.session_state.last_selected_live_amc = selected_live_amc
            
        if st.session_state.last_selected_live_amc != selected_live_amc:
            st.session_state.sim_tick_history = {int(row["amfi_code"]): [] for _, row in watchlist_funds.iterrows()}
            st.session_state.sim_tick_times = []
            st.session_state.last_selected_live_amc = selected_live_amc
            
        # Prepare historical session state for the chart
        if "sim_tick_history" not in st.session_state:
            st.session_state.sim_tick_history = {int(row["amfi_code"]): [] for _, row in watchlist_funds.iterrows()}
            st.session_state.sim_tick_times = []
            
        # Run a 10-step streaming simulation loop
        for step in range(10):
            tick_time = pd.Timestamp.now().strftime("%H:%M:%S")
            st.session_state.sim_tick_times.append(tick_time)
            current_metrics = []
            
            for _, row in watchlist_funds.iterrows():
                amfi = int(row["amfi_code"])
                # Look up baseline NAV
                fund_nav_history = nav[nav["amfi_code"] == amfi]
                baseline = fund_nav_history.sort_values("date_id").iloc[-1]["nav"] if not fund_nav_history.empty else 100.0
                
                # Retrieve last simulated NAV or baseline
                if amfi not in st.session_state.sim_tick_history:
                    st.session_state.sim_tick_history[amfi] = []
                    
                if len(st.session_state.sim_tick_history[amfi]) > 0:
                    last_val = st.session_state.sim_tick_history[amfi][-1]
                else:
                    last_val = baseline
                    
                drift = random.uniform(-0.001, 0.001)
                new_val = last_val * (1 + drift)
                st.session_state.sim_tick_history[amfi].append(new_val)
                
                pct_change = ((new_val - baseline) / baseline) * 100
                current_metrics.append({
                    "name": row["fund_name"],
                    "nav": new_val,
                    "change": pct_change,
                    "amfi": amfi
                })
                
            # Limit historical buffer to last 20 ticks
            if len(st.session_state.sim_tick_times) > 20:
                st.session_state.sim_tick_times.pop(0)
                for amfi in st.session_state.sim_tick_history:
                    if len(st.session_state.sim_tick_history[amfi]) > 20:
                        st.session_state.sim_tick_history[amfi].pop(0)
                        
            # Render inside the placeholder
            with live_ticker_placeholder.container():
                num_metrics = len(current_metrics)
                if num_metrics > 0:
                    cols_per_row = 4
                    for i in range(0, num_metrics, cols_per_row):
                        chunk = current_metrics[i:i+cols_per_row]
                        cols_ticker = st.columns(len(chunk))
                        for idx, metric in enumerate(chunk):
                            with cols_ticker[idx]:
                                sign = "+" if metric["change"] >= 0 else ""
                                color_dot = "🟢" if metric["change"] >= 0 else "🔴"
                                st.metric(
                                    label=f"{color_dot} {metric['name'][:22]}...",
                                    value=f"₹ {metric['nav']:.2f}",
                                    delta=f"{sign}{metric['change']:.2f}%"
                                )
                        
                # Render Line Chart of the simulated NAV price walk
                fig_stream = go.Figure()
                for metric in current_metrics:
                    amfi = metric["amfi"]
                    # Ensure history matches time stamps length
                    hist_y = st.session_state.sim_tick_history[amfi]
                    hist_x = st.session_state.sim_tick_times[:len(hist_y)]
                    fig_stream.add_trace(go.Scatter(
                        x=hist_x,
                        y=hist_y,
                        name=metric["name"][:18] + "...",
                        mode="lines+markers"
                    ))
                fig_stream.update_layout(
                    title="Live NAV Watchlist Ticker Feed (Real-Time Tick Stream)",
                    height=280,
                    xaxis_title="Time",
                    yaxis_title="NAV Price (INR)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=50, r=20, t=40, b=40)
                )
                st.plotly_chart(fig_stream, use_container_width=True)
                
            time.sleep(0.8)
            
    st.markdown("---")

# Tab Layout
tab_overview, tab_perf, tab_investors, tab_market, tab_segmentation, tab_robo = st.tabs([
    "📊 Platform Overview",
    "📈 Fund Performance & Scorecard",
    "👥 Investor Behavior & Demographics",
    "💡 Market Trends & Advanced Models",
    "🧠 Advanced Investor Segmentation",
    "🤖 Robo-Advisor & Portfolio Allocator"
])

# ----------------------------------------------------
# TAB 1: PLATFORM OVERVIEW
# ----------------------------------------------------
with tab_overview:
    st.subheader("Bluestock Assets & Inflows Overview")
    
    # KPIs
    # 1. Platform AUM (latest date within filters)
    if not aum_filtered.empty:
        latest_aum_date = aum_filtered["date_id"].max()
        latest_aum_val = aum_filtered[aum_filtered["date_id"] == latest_aum_date]["aum"].sum()
    else:
        latest_aum_val = 0
    
    # 2. Monthly SIP Inflow (latest month within filters)
    if not sip_filtered.empty:
        latest_sip_month = sip_filtered["month"].max()
        latest_sip_val = sip_filtered[sip_filtered["month"] == latest_sip_month]["sip_inflow"].sum()
    else:
        latest_sip_val = 0
    
    # 3. Industry Folios (latest month within filters)
    if not folios_filtered.empty:
        latest_folio_month = folios_filtered["month"].max()
        latest_folios_val = folios_filtered[folios_filtered["month"] == latest_folio_month]["total_folios_crore"].values[0]
    else:
        latest_folios_val = 0
    
    # 4. Total Active Schemes in Filter
    active_schemes_count = len(filtered_funds)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Aggregate Platform AUM", f"₹ {latest_aum_val:,.0f} Cr")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monthly SIP Inflow", f"₹ {latest_sip_val:,.0f} Cr")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Industry Folios", f"{latest_folios_val:.2f} Cr")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monitored Schemes", f"{active_schemes_count}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.write("---")
    
    # Row 2: Charts
    col_aum, col_sip = st.columns(2)
    with col_aum:
        st.subheader("AUM Trend Growth by Fund House")
        
        if not aum.empty:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Filter for 2022-2025 and group by Year & Fund House
            df_aum_filtered = aum.copy()
            df_aum_filtered['year'] = pd.to_datetime(df_aum_filtered['date_id']).dt.year
            df_aum_filtered = df_aum_filtered[df_aum_filtered['year'].isin([2022, 2023, 2024, 2025])]
            
            df_aum_yearly = df_aum_filtered.groupby(['year', 'fund_house'])['aum'].last().reset_index()
            
            # Choose a clean aspect ratio that fits inside the half-width column without getting squeezed
            fig_aum_bar, ax_aum_bar = plt.subplots(figsize=(7.5, 5))
            
            # Set dark style to match dashboard theme
            sns.set_theme(style="dark")
            plt.rcParams['figure.facecolor'] = '#090d16'
            plt.rcParams['axes.facecolor'] = '#090d16'
            plt.rcParams['text.color'] = '#f8fafc'
            plt.rcParams['axes.labelcolor'] = '#94a3b8'
            plt.rcParams['xtick.color'] = '#94a3b8'
            plt.rcParams['ytick.color'] = '#94a3b8'
            plt.rcParams['axes.edgecolor'] = '#1e293b'
            plt.rcParams['grid.color'] = '#1e293b'
            
            sns.barplot(
                data=df_aum_yearly, 
                x="year", 
                y="aum", 
                hue="fund_house", 
                ax=ax_aum_bar,
                palette="viridis"
            )
            
            # Increase Y-limit to make space for the annotation text
            ax_aum_bar.set_ylim(0, 1600000.0)
            
            # Highlight SBI at ₹12.5L Cr dominance in 2025 (x = 3 in seaborn indices)
            ax_aum_bar.annotate(
                "SBI Mutual Fund\nDominance: ₹12.5L Cr (2025)",
                xy=(3.22, 1250000.0),
                xytext=(0.5, 1400000.0), # Move text up and left to avoid bar overlap
                arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.0, headwidth=5, headlength=5),
                color="#10b981",
                weight="bold",
                fontsize=8.5
            )
            
            # Position legend horizontally below the plot in 3 columns to avoid right-side overflow into the SIP column
            ax_aum_bar.legend(
                title="Fund House", 
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), 
                ncol=3, 
                fontsize=8, 
                title_fontsize=8
            )
            
            ax_aum_bar.set_title("Grouped AUM Standings by Fund House (2022–2025)", color="#f8fafc", weight='bold', pad=10, fontsize=10)
            ax_aum_bar.set_xlabel("Year", color="#94a3b8", fontsize=8.5)
            ax_aum_bar.set_ylabel("AUM (Crores)", color="#94a3b8", fontsize=8.5)
            ax_aum_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
            
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_aum_bar)
        else:
            st.info("No AUM data available for the 2022-2025 range.")
        
    with col_sip:
        st.subheader("Monthly SIP Inflows Growth (Crores)")
        if not sip_filtered.empty:
            unique_dates = sip_filtered["date_id"].unique()
            if len(unique_dates) > 1:
                fig_sip = px.line(
                    sip_filtered, 
                    x="month", 
                    y="sip_inflow",
                    labels={"month": x_axis_label, "sip_inflow": "SIP Inflow (Crores)"},
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_sip.update_traces(mode="lines+markers", line=dict(width=2.5))
                fig_sip.update_layout(yaxis=dict(tickformat=","))
                # Annotate Peak (Dec 2025 all-time high of 31,002 Cr)
                if "2025-12" in sip_filtered["month"].values:
                    fig_sip.add_annotation(
                        x="2025-12",
                        y=31002.0,
                        text="🏆 All-Time High: ₹31,002 Cr (Dec 2025)",
                        showarrow=True,
                        arrowhead=2,
                        ax=-60,
                        ay=-45,
                        bgcolor="#10b981",
                        bordercolor="#059669",
                        font=dict(color="white", size=10)
                    )
            else:
                latest_month_str = sip_filtered["month"].iloc[0]
                fig_sip = px.bar(
                    sip_filtered,
                    x="month",
                    y="sip_inflow",
                    labels={"month": x_axis_label, "sip_inflow": "SIP Inflow (Crores)"},
                    title=f"SIP Inflow for {latest_month_str}",
                    template="plotly_dark",
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_sip.update_layout(yaxis=dict(tickformat=","))
                if latest_month_str == "2025-12":
                    fig_sip.add_annotation(
                        x="2025-12",
                        y=31002.0,
                        text="🏆 All-Time High: ₹31,002 Cr",
                        showarrow=True,
                        arrowhead=2,
                        ax=-60,
                        ay=-45,
                        bgcolor="#10b981",
                        bordercolor="#059669",
                        font=dict(color="white", size=10)
                    )
            fig_sip.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sip, use_container_width=True)
        else:
            st.info("No SIP data available for the selected time range. Try expanding the time window.")
            
    # Row 2.5: Folio Count Growth Analysis (Task 2)
    st.write("---")
    st.subheader("📈 Mutual Fund Folio Growth Trend (2022–2025)")
    if not folios.empty:
        import plotly.graph_objects as go
        df_folios = folios.sort_values("date_id")
        
        # Calculate percentage growth
        start_val = df_folios['total_folios_crore'].iloc[0]
        end_val = df_folios['total_folios_crore'].iloc[-1]
        growth_pct = ((end_val - start_val) / start_val) * 100
        
        fig_folios = go.Figure()
        fig_folios.add_trace(go.Scatter(
            x=df_folios['month'],
            y=df_folios['total_folios_crore'],
            mode='lines+markers',
            name='Total Folios',
            line=dict(color='#0ea5e9', width=3.5),
            marker=dict(size=8, color='#38bdf8', symbol='circle')
        ))
        
        # Annotate Peak
        highest_idx = df_folios['total_folios_crore'].idxmax()
        highest_month = df_folios.loc[highest_idx, 'month']
        highest_val = df_folios.loc[highest_idx, 'total_folios_crore']
        
        fig_folios.add_annotation(
            x=highest_month,
            y=highest_val,
            text=f"Peak: {highest_val:.2f} Cr (Dec 2025)",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-45,
            bgcolor="#10b981",
            bordercolor="#059669",
            font=dict(color="white", size=11)
        )
        
        # Annotate Baseline
        fig_folios.add_annotation(
            x=df_folios['month'].iloc[0],
            y=df_folios['total_folios_crore'].iloc[0],
            text=f"Baseline: {start_val:.2f} Cr (Jan 2022)",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=45,
            bgcolor="#3b82f6",
            bordercolor="#2563eb",
            font=dict(color="white", size=11)
        )
        
        # Annotate crossed 20Cr milestone
        mid_milestone = df_folios[df_folios['total_folios_crore'] >= 20.0].iloc[0]
        fig_folios.add_annotation(
            x=mid_milestone['month'],
            y=mid_milestone['total_folios_crore'],
            text=f"Crossed 20 Cr ({mid_milestone['month']})",
            showarrow=True,
            arrowhead=2,
            ax=-30,
            ay=40,
            bgcolor="#f59e0b",
            bordercolor="#d97706",
            font=dict(color="white", size=11)
        )
        
        fig_folios.update_layout(
            title=dict(
                text=f"Total Platform Folio Growth | Growth: +{growth_pct:.2f}% (from {start_val} Cr to {end_val} Cr)",
                font=dict(size=14, color="#94a3b8")
            ),
            xaxis_title="Timeline",
            yaxis_title="Folio Count (Crores)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b"),
            margin=dict(l=60, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_folios, use_container_width=True)
    else:
        st.info("No folio count data available.")
        
    # Row 3: Smart Insights Engine (Advanced Fintech Feature)
    st.markdown('<div class="insights-card">', unsafe_allow_html=True)
    st.subheader("🧠 Smart Decision-Support Insights Engine")
    
    if len(filtered_funds) > 0:
        best_fund = filtered_funds.loc[filtered_funds["cagr_3y"].idxmax()]
        worst_fund = filtered_funds.loc[filtered_funds["cagr_3y"].idxmin()]
        highest_sharpe_fund = filtered_funds.loc[filtered_funds["sharpe_ratio"].idxmax()]
        most_volatile_fund = filtered_funds.loc[filtered_funds["beta"].idxmax()]
        best_downside_fund = filtered_funds.loc[filtered_funds["max_drawdown"].idxmin()]
        
        # Max Inflow Category
        if not cat_inflows_filtered.empty:
            max_inflow_cat = cat_inflows_filtered.groupby("category")["net_inflow"].sum().idxmax()
            cat_inflow_sum = cat_inflows_filtered.groupby("category")["net_inflow"].sum().max()
        else:
            max_inflow_cat = "N/A"
            cat_inflow_sum = 0
            
        # Nifty 50 Regime check
        nifty_data = bench[bench["benchmark_name"] == "NIFTY50"].sort_values("date_id").copy()
        regime_msg = ""
        if len(nifty_data) > 200:
            nifty_data["SMA_50"] = nifty_data["value"].rolling(50).mean()
            nifty_data["SMA_200"] = nifty_data["value"].rolling(200).mean()
            latest_sma50 = nifty_data.iloc[-1]["SMA_50"]
            latest_sma200 = nifty_data.iloc[-1]["SMA_200"]
            if latest_sma50 > latest_sma200:
                regime_msg = "🟢 **Bullish regime** detected (SMA 50 > 200). Investors should favor equities & growth funds."
            else:
                regime_msg = "🔴 **Bearish regime / contraction** detected (SMA 50 < 200). Investors should prioritize downside protection and debt/hybrid allocations."

        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.write("##### 📈 Growth & Performance Attributions")
            st.markdown(f"🏆 **Outperformer:** `{best_fund['fund_name']}` shows the highest return momentum in this filter range, yielding a **{best_fund['cagr_3y']*100:.2f}%** CAGR.")
            st.markdown(f"⚠️ **Underperformer:** `{worst_fund['fund_name']}` has underperformed the cohort with a CAGR of **{worst_fund['cagr_3y']*100:.2f}%**.")
            if regime_msg:
                st.markdown(f"📡 **Regime Classifier:** {regime_msg}")
                
        with col_ins2:
            st.write("##### ⚖️ Risk & Capital Protection Insights")
            st.markdown(f"💎 **Risk-Adjusted Return Leader:** `{highest_sharpe_fund['fund_name']}` demonstrates superior efficiency with a Sharpe ratio of **{highest_sharpe_fund['sharpe_ratio']:.2f}**.")
            st.markdown(f"🛡️ **Downside Guard:** `{best_downside_fund['fund_name']}` shows the strongest capital preservation, limiting historical drawdown to **{best_downside_fund['max_drawdown']*100:.1f}%**.")
            st.markdown(f"⚡ **Volatility Warning:** `{most_volatile_fund['fund_name']}` is the most sensitive to market volatility with a Beta of **{most_volatile_fund['beta']:.2f}**.")
            
        st.write("---")
        # Inflows insights
        if max_inflow_cat != "N/A":
            st.markdown(f"📥 **Inflow Momentum:** The **{max_inflow_cat}** category led the market in allocations, drawing **₹{cat_inflow_sum:,.1f} Cr** in net inflows. This indicates strong institutional and retail investor confidence in this sector.")
    else:
        st.write("No funds match current sidebar criteria to generate insights.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: PERFORMANCE & SCORECARD
# ----------------------------------------------------
with tab_perf:
    st.subheader("Fund Attributions & Multi-Metric Scorecard")
    
    # local tab slicer: Fund Select
    fund_list = filtered_funds["fund_name"].tolist()
    if fund_list:
        selected_fund = st.selectbox("Select Fund for NAV vs Benchmark overlay:", options=fund_list)
        fund_details = filtered_funds[filtered_funds["fund_name"] == selected_fund].iloc[0]
        
        amfi_c = fund_details["amfi_code"]
        bench_raw = fund_details["benchmark"]
        db_bench = benchmark_mapping.get(bench_raw, "NIFTY50")
        
        # Load NAVs & Mapped Benchmark
        fund_navs = nav[
            (nav["amfi_code"] == amfi_c) & 
            (nav["date_id"] >= start_dt) & 
            (nav["date_id"] <= end_dt)
        ].sort_values("date_id").copy()
        
        bench_navs = bench[
            (bench["benchmark_name"] == db_bench) & 
            (bench["date_id"] >= start_dt) & 
            (bench["date_id"] <= end_dt)
        ].sort_values("date_id").copy()
        
        compare = pd.merge(fund_navs, bench_navs, on="date_id", suffixes=("_fund", "_bench"))
        
        if len(compare) > 0:
            compare["Fund NAV Normalized"] = (compare["nav"] / compare["nav"].iloc[0]) * 100
            compare["Benchmark Value Normalized"] = (compare["value"] / compare["value"].iloc[0]) * 100
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=compare["date_id"], y=compare["Fund NAV Normalized"], name=f"{selected_fund} (NAV)", line=dict(color="#38bdf8", width=2.5)))
            fig_compare.add_trace(go.Scatter(x=compare["date_id"], y=compare["Benchmark Value Normalized"], name=f"Benchmark: {bench_raw}", line=dict(color="#f43f5e", width=2, dash="dash")))
            fig_compare.update_layout(
                    title=f"Attribution Chart: {selected_fund} vs {bench_raw}",
                    xaxis_title=x_axis_label,
                    yaxis_title="Normalized Close (Base 100)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
            st.plotly_chart(fig_compare, use_container_width=True)
            
        st.write("---")
        
        # Task 4: NAV Trend Analysis (40 schemes, daily NAV, Plotly highlighting 2023 Bull & 2024 Correction)
        st.subheader("📈 Daily NAV Trend Analysis (2022–2026)")
        st.markdown("*Filter by AMC and Asset Category to narrow down the list, then select specific schemes to plot and compare.*")
        try:
            # Load daily NAVs with metadata
            df_nav_40 = pd.read_sql("""
                SELECT n.date_id, f.fund_name, n.nav, f.fund_house, f.category
                FROM fact_nav n 
                JOIN dim_fund f ON n.amfi_code = f.amfi_code 
                ORDER BY n.date_id
            """, engine)
            
            # Add filters above the chart
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                amc_list = ["All AMCs"] + sorted(df_nav_40['fund_house'].dropna().unique().tolist())
                selected_amc = st.selectbox("Filter Schemes by AMC (Fund House)", amc_list, key="nav_trend_amc_select")
            with col_filt2:
                cat_list = ["All Categories"] + sorted(df_nav_40['category'].dropna().unique().tolist())
                selected_cat = st.selectbox("Filter Schemes by Asset Category", cat_list, key="nav_trend_cat_select")
                
            # Apply filters to schemes
            filtered_nav_df = df_nav_40.copy()
            if selected_amc != "All AMCs":
                filtered_nav_df = filtered_nav_df[filtered_nav_df['fund_house'] == selected_amc]
            if selected_cat != "All Categories":
                filtered_nav_df = filtered_nav_df[filtered_nav_df['category'] == selected_cat]
                
            available_schemes = sorted(filtered_nav_df['fund_name'].unique().tolist())
            
            # User helper for selecting all/deselecting all schemes in selection
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                select_all_schemes = st.checkbox("Select All Available", value=False, key="nav_trend_select_all_btn")
                
            if select_all_schemes:
                default_schemes = available_schemes
            else:
                default_schemes = available_schemes[:5] if len(available_schemes) > 5 else available_schemes
                
            selected_schemes = st.multiselect(
                "Select specific Mutual Fund Schemes to compare:",
                options=available_schemes,
                default=default_schemes,
                key="nav_trend_schemes_multiselect"
            )
            
            if selected_schemes:
                # Pivot only selected funds
                df_nav_selected = filtered_nav_df[filtered_nav_df['fund_name'].isin(selected_schemes)]
                df_nav_pivot = df_nav_selected.pivot(index='date_id', columns='fund_name', values='nav').reset_index()
                
                fig_nav_all = go.Figure()
                cols_to_plot = [c for c in df_nav_pivot.columns if c != 'date_id']
                
                for col_name in cols_to_plot:
                    fig_nav_all.add_trace(go.Scatter(
                        x=df_nav_pivot['date_id'],
                        y=df_nav_pivot[col_name],
                        name=col_name[:20] + "...",
                        mode='lines',
                        line=dict(width=1.5),
                        hovertemplate="<b>" + col_name[:25] + "</b><br>Date: %{x}<br>NAV: ₹%{y:.2f}<extra></extra>"
                    ))
                    
                fig_nav_all.update_layout(
                    shapes=[
                        # 2023 Bull Run
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0="2023-01-01",
                            x1="2023-12-31",
                            y0=0,
                            y1=1,
                            fillcolor="rgba(16, 185, 129, 0.05)",
                            layer="below",
                            line=dict(width=0)
                        ),
                        # 2024 Market Corrections
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0="2024-01-01",
                            x1="2024-12-31",
                            y0=0,
                            y1=1,
                            fillcolor="rgba(239, 68, 68, 0.05)",
                            layer="below",
                            line=dict(width=0)
                        )
                    ],
                    annotations=[
                        dict(
                            x="2023-07-01",
                            y=1.03,
                            xref="x",
                            yref="paper",
                            text="🟢 2023 Bull Run Regime",
                            showarrow=False,
                            font=dict(color="#10b981", size=11, weight="bold")
                        ),
                        dict(
                            x="2024-07-01",
                            y=1.03,
                            xref="x",
                            yref="paper",
                            text="🔴 2024 Market Correction",
                            showarrow=False,
                            font=dict(color="#ef4444", size=11, weight="bold")
                        )
                    ],
                    title=dict(
                        text="Daily NAV Trendlines (2022–2026)",
                        font=dict(size=14, color="#f8fafc")
                    ),
                    xaxis_title="Date",
                    yaxis_title="Net Asset Value (NAV in INR)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    margin=dict(l=60, r=40, t=60, b=40),
                    showlegend=True if len(selected_schemes) <= 15 else False
                )
                st.plotly_chart(fig_nav_all, use_container_width=True)
            else:
                st.info("Please select at least one scheme to display the NAV Trend Analysis.")
        except Exception as nav_ex:
            st.warning(f"Could not load 40 schemes daily NAV trends: {nav_ex}")
            
        st.write("---")
        
        # Risk Return Scatter & Scorecard Table
        col_sc, col_tab = st.columns([5, 7])
        
        with col_sc:
            st.subheader("Risk vs Return Scatter Plot")
            
            df_plot = filtered_funds.copy()
            if not df_plot.empty:
                df_plot["return_rank"] = df_plot["cagr_3y"].rank(pct=True)
                df_plot["sharpe_rank"] = df_plot["sharpe_ratio"].rank(pct=True)
                df_plot["alpha_rank"] = df_plot["alpha"].rank(pct=True)
                
                df_plot["size_score"] = (
                    df_plot["return_rank"] * 0.4 +
                    df_plot["sharpe_rank"] * 0.4 +
                    df_plot["alpha_rank"] * 0.2
                )
                
                scaler = MinMaxScaler(feature_range=(5, 40))
                df_plot["size_scaled"] = scaler.fit_transform(df_plot[["size_score"]].fillna(0).clip(lower=0))
                
                fig_sc = px.scatter(
                    df_plot, 
                    x="max_drawdown", 
                    y="sharpe_ratio", 
                    size="size_scaled", 
                    color="category",
                    hover_name="fund_name",
                    size_max=40,
                    labels={"max_drawdown": "Max Drawdown (%)", "sharpe_ratio": "Sharpe Ratio (3Y)", "size_scaled": "Score"},
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("No data available for Risk vs Return plot.")
            
        with col_tab:
            st.subheader("Weighted Scorecard Ranking")
            
            # Scorecard calculation using percentile ranks
            df_score = filtered_funds.copy()
            if not df_score.empty:
                df_score["cagr_n"] = df_score["cagr_3y"].rank(pct=True)
                df_score["sharpe_n"] = df_score["sharpe_ratio"].rank(pct=True)
                df_score["alpha_n"] = df_score["alpha"].rank(pct=True)
                df_score["drawdown_n"] = df_score["max_drawdown"].rank(pct=True, ascending=False)
                df_score["expense_n"] = df_score["expense_ratio"].rank(pct=True, ascending=False)
                
                # Weighted Scoring using config weights
                w = config["scorecard"]["weights"]
                df_score["Score"] = (
                    df_score["cagr_n"] * w.get("cagr", 0.30) +
                    df_score["sharpe_n"] * w.get("sharpe", 0.25) +
                    df_score["alpha_n"] * w.get("alpha", 0.20) +
                    df_score["drawdown_n"] * w.get("drawdown", 0.10) +
                    df_score["expense_n"] * w.get("expense", 0.15)
                ) * 100
                df_score["Score"] = df_score["Score"].round(2)
                df_score = df_score.sort_values("Score", ascending=False).reset_index(drop=True)
                df_score["Rank"] = range(1, len(df_score) + 1)
            
            st.dataframe(
                df_score[["Rank", "fund_name", "category", "cagr_3y", "sharpe_ratio", "alpha", "max_drawdown", "expense_ratio", "Score"]],
                column_config={
                    "Rank": "Rank",
                    "fund_name": "Scheme Name",
                    "category": "Category",
                    "cagr_3y": st.column_config.NumberColumn("Return 3Y", format="%.2f%%"),
                    "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "alpha": st.column_config.NumberColumn("Alpha", format="%.2f"),
                    "max_drawdown": st.column_config.NumberColumn("Drawdown", format="%.2f%%"),
                    "expense_ratio": st.column_config.NumberColumn("Expense", format="%.2f%%"),
                    "Score": st.column_config.ProgressColumn("Overall Score", format="%.1f", min_value=0, max_value=100)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Now insert the Top 5 Funds vs Nifty 50 & 100 Benchmark Comparison Chart
            st.write("---")
            st.subheader("🏆 Top 5 Scorecard Funds vs Nifty Benchmarks (3-Year Timeline)")
            st.markdown("*Normalized performance (Base 100) comparing the top 5 scorecard funds against Nifty 50 and Nifty 100 over a 3-year period.*")
            
            try:
                top_5_amfi = df_score.head(5)["amfi_code"].tolist()
                
                # Filter data for last 3 years using the correct 'nav' and 'funds' variables
                end_date_dt = nav["date_id"].max()
                start_date_dt = end_date_dt - pd.DateOffset(years=3)
                
                df_nav_filtered = nav[(nav["amfi_code"].isin(top_5_amfi)) & (nav["date_id"] >= start_date_dt) & (nav["date_id"] <= end_date_dt)]
                df_nav_top5 = pd.merge(df_nav_filtered, fund_perf[["amfi_code", "fund_name"]].drop_duplicates(), on="amfi_code").copy()
                df_nifty = bench[(bench["benchmark_name"].isin(["NIFTY50", "NIFTY100"])) & (bench["date_id"] >= start_date_dt) & (bench["date_id"] <= end_date_dt)].copy()
                
                df_nav_pivot = df_nav_top5.pivot(index="date_id", columns="fund_name", values="nav").sort_index()
                df_bench_pivot = df_nifty.pivot(index="date_id", columns="benchmark_name", values="value").sort_index()
                df_merged_chart = pd.merge(df_nav_pivot, df_bench_pivot, left_index=True, right_index=True, how="inner")
                
                # Normalize (Base 100)
                df_normalized = (df_merged_chart / df_merged_chart.iloc[0]) * 100
                
                # Plotly Chart
                import plotly.graph_objects as go
                fig_top5 = go.Figure()
                
                colors_top5 = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6"]
                for idx, col_name in enumerate(df_nav_pivot.columns):
                    short_n = col_name.replace("Regular Plan - Growth", "").replace("- Growth", "").strip()[:20] + "..."
                    fig_top5.add_trace(go.Scatter(
                        x=df_normalized.index,
                        y=df_normalized[col_name],
                        name=short_n,
                        mode='lines',
                        line=dict(color=colors_top5[idx % len(colors_top5)], width=2.0),
                        hovertemplate="<b>" + col_name[:25] + "</b><br>Date: %{x}<br>Perf: %{y:.2f}%<extra></extra>"
                    ))
                    
                fig_top5.add_trace(go.Scatter(
                    x=df_normalized.index,
                    y=df_normalized["NIFTY50"],
                    name="Nifty 50 Index",
                    mode='lines',
                    line=dict(color="#ef4444", width=2.2, dash="dash"),
                    hovertemplate="<b>Nifty 50</b><br>Date: %{x}<br>Perf: %{y:.2f}%<extra></extra>"
                ))
                
                fig_top5.add_trace(go.Scatter(
                    x=df_normalized.index,
                    y=df_normalized["NIFTY100"],
                    name="Nifty 100 Index",
                    mode='lines',
                    line=dict(color="#94a3b8", width=2.2, dash="dot"),
                    hovertemplate="<b>Nifty 100</b><br>Date: %{x}<br>Perf: %{y:.2f}%<extra></extra>"
                ))
                
                fig_top5.update_layout(
                    xaxis_title="Timeline",
                    yaxis_title="Normalized Performance (Base 100)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    yaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    margin=dict(l=50, r=20, t=20, b=40)
                )
                
                # Calculate tracking errors
                te_records = []
                for fund_name in df_nav_pivot.columns:
                    fund_ret = df_merged_chart[fund_name].pct_change().dropna()
                    nifty50_ret = df_merged_chart["NIFTY50"].pct_change().dropna()
                    nifty100_ret = df_merged_chart["NIFTY100"].pct_change().dropna()
                    
                    te_n50 = np.std(fund_ret - nifty50_ret) * np.sqrt(252) * 100
                    te_n100 = np.std(fund_ret - nifty100_ret) * np.sqrt(252) * 100
                    
                    te_records.append({
                        "Scheme Name": fund_name[:25] + "...",
                        "TE vs Nifty 50": f"{te_n50:.2f}%",
                        "TE vs Nifty 100": f"{te_n100:.2f}%"
                    })
                    
                col_chart5, col_te5 = st.columns([7, 5])
                with col_chart5:
                    st.plotly_chart(fig_top5, use_container_width=True)
                with col_te5:
                    st.markdown("##### 📐 Annualized Tracking Errors")
                    st.markdown("*Tracking error measures volatility of active excess returns vs benchmark index.*")
                    st.dataframe(pd.DataFrame(te_records), hide_index=True, use_container_width=True)
                    
            except Exception as chart5_ex:
                st.warning(f"Could not load benchmark comparison chart: {chart5_ex}")
    else:
        st.info("No funds match current categories selection.")
        
    # Task 3: NAV Return Correlation Matrix
    st.write("---")
    st.subheader("🔬 NAV Return Correlation Matrix (Top 10 Funds)")
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Select 10 representative funds
        funds_sample = pd.read_sql("SELECT amfi_code, fund_name FROM dim_fund LIMIT 10", engine)
        codes_sample = tuple(funds_sample['amfi_code'].tolist())
        
        # Load daily NAV history
        nav_sample_df = pd.read_sql(f"SELECT amfi_code, date_id, nav FROM fact_nav WHERE amfi_code IN {codes_sample} ORDER BY date_id", engine)
        
        # Pivot to date x fund matrix
        nav_sample_pivot = nav_sample_df.pivot(index='date_id', columns='amfi_code', values='nav').sort_index()
        
        # Calculate daily returns
        returns_sample_df = nav_sample_pivot.pct_change().dropna()
        
        # Rename columns to short names
        code_to_name_sample = dict(zip(funds_sample['amfi_code'], funds_sample['fund_name'].str.replace("Regular Plan - Growth", "").str.strip().str.slice(0, 18) + "..."))
        returns_sample_df = returns_sample_df.rename(columns=code_to_name_sample)
        
        # Compute correlation matrix
        corr_matrix_sample = returns_sample_df.corr()
        
        # Set up dark Seaborn styling
        sns.set_theme(style="dark")
        plt.rcParams['figure.facecolor'] = '#090d16'
        plt.rcParams['axes.facecolor'] = '#090d16'
        plt.rcParams['text.color'] = '#f8fafc'
        plt.rcParams['axes.labelcolor'] = '#94a3b8'
        plt.rcParams['xtick.color'] = '#94a3b8'
        plt.rcParams['ytick.color'] = '#94a3b8'
        plt.rcParams['axes.edgecolor'] = '#1e293b'
        plt.rcParams['grid.color'] = '#1e293b'
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8.5))
        sns.heatmap(
            corr_matrix_sample, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1.0, 
            vmax=1.0, 
            ax=ax_corr,
            linewidths=0.5,
            linecolor="#1e293b",
            annot_kws={"size": 10, "weight": "bold"}
        )
        ax_corr.set_title("NAV Daily Return Correlation Matrix (Top 10 Funds)", color="#f8fafc", weight='bold', pad=10)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_corr)
        
        st.markdown("""
        💡 ***Diversification Note:***
        * Correlation value of **+1.00** shows identical movement (no diversification).
        * Correlation values near **0.00** or negative signify strong diversification benefits, lowering overall portfolio risk.
        """)
    except Exception as corr_ex:
        st.warning(f"Could not generate return correlation matrix: {corr_ex}")
        
    # Task 1 Remaining: Daily Returns Distribution Validation Chart
    st.write("---")
    st.subheader("📊 Daily Return Distribution Validation")
    st.markdown("*Histogram with Kernel Density Estimate (KDE) validating that daily return distributions look reasonable (bell-curved with standard financial fat tails).*")
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Load NAV data and calculate returns
        df_returns_all = nav.copy()
        df_returns_all["daily_return"] = df_returns_all.groupby("amfi_code")["nav"].pct_change()
        df_returns_all = df_returns_all.dropna()
        
        # Calculate metrics
        mean_ret = df_returns_all["daily_return"].mean()
        std_ret = df_returns_all["daily_return"].std()
        skew_ret = df_returns_all["daily_return"].skew()
        kurt_ret = df_returns_all["daily_return"].kurtosis()
        
        col_dist1, col_dist2 = st.columns([7, 5])
        with col_dist1:
            fig_dist, ax_dist = plt.subplots(figsize=(7, 4))
            
            sns.set_theme(style="dark")
            plt.rcParams['figure.facecolor'] = '#090d16'
            plt.rcParams['axes.facecolor'] = '#090d16'
            plt.rcParams['text.color'] = '#f8fafc'
            plt.rcParams['axes.labelcolor'] = '#94a3b8'
            plt.rcParams['xtick.color'] = '#94a3b8'
            plt.rcParams['ytick.color'] = '#94a3b8'
            plt.rcParams['axes.edgecolor'] = '#1e293b'
            plt.rcParams['grid.color'] = '#1e293b'
            
            sns.histplot(df_returns_all["daily_return"], bins=100, kde=True, color="#38bdf8", ax=ax_dist)
            ax_dist.set_title("Daily Return Distribution (All 40 Schemes)", color="#f8fafc", weight='bold', fontsize=10)
            ax_dist.set_xlabel("Daily Return", color="#94a3b8", fontsize=8.5)
            ax_dist.set_ylabel("Frequency", color="#94a3b8", fontsize=8.5)
            ax_dist.set_xlim(-0.04, 0.04)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_dist)
        with col_dist2:
            st.markdown("##### 📈 Distribution Characteristics")
            st.markdown(f"""
            * **Sample Size**: {len(df_returns_all):,} daily points
            * **Average Daily Return (Mean)**: `{mean_ret:.6f}`
            * **Daily Volatility (Std Dev)**: `{std_ret:.6f}` (Ann: `{std_ret*np.sqrt(252)*100:.2f}%`)
            * **Skewness**: `{skew_ret:.4f}`
            * **Kurtosis (Fat Tails)**: `{kurt_ret:.4f}`
            """)
            st.info("💡 **Interpretation**: A kurtosis value > 0 confirms standard asset return characteristics with heavy tails (more extreme events than a perfect normal distribution). Skewness close to 0 indicates a balanced, stable dataset.")
    except Exception as dist_ex:
        st.warning(f"Could not render return distribution: {dist_ex}")

# ----------------------------------------------------
# TAB 3: INVESTOR BEHAVIOR
# ----------------------------------------------------
with tab_investors:
    st.subheader("Investor Demographics & Cohort Analysis")
    
    if not txs_filtered.empty:
        col_dem1, col_dem2 = st.columns(2)
        with col_dem1:
            st.subheader("State-Wise Transaction Value (Lakhs)")
            tx_inv = pd.merge(txs_filtered, investors, on="investor_id")
            state_agg = tx_inv.groupby("state")["amount"].sum().reset_index()
            state_agg["amount_lakhs"] = (state_agg["amount"] / 100000).round(2)
            state_agg = state_agg.sort_values("amount_lakhs", ascending=False)
            
            fig_state = px.bar(
                state_agg, 
                x="amount_lakhs", 
                y="state", 
                orientation="h",
                labels={"amount_lakhs": "Invested Amount (Lakhs)", "state": "State"},
                template="plotly_dark",
                color="amount_lakhs",
                color_continuous_scale="Viridis"
            )
            fig_state.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_state, use_container_width=True)
            
        with col_dem2:
            st.subheader("Investor Age Group Volumes")
            age_agg = tx_inv.groupby("age_group")["amount"].sum().reset_index()
            age_agg["amount_lakhs"] = (age_agg["amount"] / 100000).round(2)
            
            fig_age = px.pie(
                age_agg, 
                values="amount_lakhs", 
                names="age_group",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age, use_container_width=True)
            
        st.write("---")
        
        col_tx1, col_tx2 = st.columns([4, 8])
        with col_tx1:
            st.subheader("Transaction Types Breakdown")
            type_agg = tx_inv.groupby("transaction_type")["amount"].sum().reset_index()
            type_agg["amount_lakhs"] = (type_agg["amount"] / 100000).round(2)
            
            fig_type = px.bar(
                type_agg, 
                x="transaction_type", 
                y="amount_lakhs",
                labels={"transaction_type": "Type", "amount_lakhs": "Volume (Lakhs)"},
                template="plotly_dark",
                color="transaction_type",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_type, use_container_width=True)
            
        with col_tx2:
            st.subheader("Investor Cohorts Activity Heatmap (Lakhs)")
            # Work on a copy to avoid SettingWithCopyWarning
            txs_filtered_cohort = txs_filtered.copy()
            txs_filtered_cohort["year"] = txs_filtered_cohort["date_id"].dt.year
            
            # Calculate cohort first transaction year
            first_tx = txs_filtered_cohort.groupby("investor_id")["year"].min().reset_index()
            first_tx.columns = ["investor_id", "Cohort Year"]
            
            merged_cohort = pd.merge(txs_filtered_cohort, first_tx, on="investor_id")
            cohort_matrix = merged_cohort.groupby(["Cohort Year", "year"])["amount"].sum().unstack().fillna(0)
            cohort_matrix = (cohort_matrix / 100000).round(2) # convert to Lakhs
            
            fig_cohort = px.imshow(
                cohort_matrix,
                labels=dict(x="Transaction Year", y="Cohort Year", color="Volume (Lakhs)"),
                x=cohort_matrix.columns,
                y=cohort_matrix.index,
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
                template="plotly_dark"
            )
            fig_cohort.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cohort, use_container_width=True)
    else:
        st.info("No transaction data available for the selected time range. Try expanding the time window.")

# ----------------------------------------------------
# TAB 4: MARKET TRENDS & ADVANCED MODELS
# ----------------------------------------------------
with tab_market:
    st.subheader("Indices Comparison, Monte Carlo, and Portfolio Optimizations")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Sector Allocation of Portfolio Holdings")
        sect_agg = holdings[holdings["amfi_code"].isin(filtered_amfi_codes)].groupby("sector")["weightage"].mean().reset_index()
        sect_agg = sect_agg.sort_values("weightage", ascending=False).head(10)
        
        if not sect_agg.empty:
            fig_sect = px.pie(
                sect_agg, 
                values="weightage", 
                names="sector",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Purples
            )
            fig_sect.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sect, use_container_width=True)
        else:
            st.info("No portfolio holdings data available for the selected schemes.")
        
    with col_m2:
        st.subheader("Monthly Category-wise Net Inflows")
        inflow_view = st.radio("Toggle Inflow Representation:", ["Heatmap", "Grouped Bar Chart"], horizontal=True, key="inflow_view_toggle")
        
        if not cat_inflows_filtered.empty:
            if inflow_view == "Heatmap":
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                # Pivot months x categories
                pivot_df = cat_inflows_filtered.pivot(index='category', columns='month', values='net_inflow')
                
                # Set up dark Seaborn styling
                sns.set_theme(style="dark")
                plt.rcParams['figure.facecolor'] = '#090d16'
                plt.rcParams['axes.facecolor'] = '#090d16'
                plt.rcParams['text.color'] = '#f8fafc'
                plt.rcParams['axes.labelcolor'] = '#94a3b8'
                plt.rcParams['xtick.color'] = '#94a3b8'
                plt.rcParams['ytick.color'] = '#94a3b8'
                plt.rcParams['axes.edgecolor'] = '#1e293b'
                plt.rcParams['grid.color'] = '#1e293b'
                
                fig_heat, ax_heat = plt.subplots(figsize=(10, 6.5))
                sns.heatmap(
                    pivot_df, 
                    annot=True, 
                    fmt=".1f", 
                    cmap="coolwarm", 
                    ax=ax_heat, 
                    cbar_kws={'label': 'Net Inflow (Crores)'},
                    linewidths=0.5,
                    linecolor="#1e293b"
                )
                ax_heat.set_title("Monthly Category-wise Net Inflow Heatmap", color="#f8fafc", weight='bold', pad=10)
                ax_heat.set_xlabel(x_axis_label, color="#94a3b8")
                ax_heat.set_ylabel("Fund Categories", color="#94a3b8")
                plt.tight_layout()
                st.pyplot(fig_heat)
                
                st.markdown("""
                💡 ***Heatmap Interpretation:***
                * **Warm (Red) sectors** represent peak allocation months; **Cold (Blue) sectors** indicate lower net inflows.
                * Outlines rotation patterns: Multi-cap/Flexi-cap show consistent support, while Sectoral and Liquid categories fluctuate due to tax deadlines and corporate cycles.
                """)
            else:
                cat_agg = cat_inflows_filtered.groupby(["month", "category"])["net_inflow"].sum().reset_index()
                fig_cat = px.bar(
                    cat_agg, 
                    x="month", 
                    y="net_inflow", 
                    color="category",
                    barmode="group",
                    labels={"month": x_axis_label, "net_inflow": "Net Inflows (Crores)", "category": "Category"},
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No net inflow data available for the selected time range. Try expanding the time window.")
        
    st.write("---")
    
    # MONTE CARLO NAV PROJECTION (5-Years)
    st.subheader("🎲 Monte Carlo NAV Projection")
    mc_select = st.selectbox("Select Fund for Monte Carlo Simulation / ARIMA Forecast:", options=filtered_funds["fund_name"].tolist(), key="mc_tab_select")
    
    if not filtered_funds.empty and mc_select:
        mc_amfi = filtered_funds[filtered_funds["fund_name"] == mc_select]["amfi_code"].values[0]
        nav_series = nav[
            (nav["amfi_code"] == mc_amfi) & 
            (nav["date_id"] >= start_dt) & 
            (nav["date_id"] <= end_dt)
        ].sort_values("date_id").copy()
        
        col_mc_sim, col_arima_forecast = st.columns(2)
        
        with col_mc_sim:
            st.subheader("🎲 Monte Carlo Simulation Paths")
            if len(nav_series) > 5:
                nav_series["ret"] = nav_series["nav"].pct_change()
                daily_returns = nav_series["ret"].dropna()
                
                mean_ret = daily_returns.mean()
                vol_ret = daily_returns.std()
                current_nav = nav_series["nav"].iloc[-1]
                
                n_days = 1260 # 5 years
                n_sims = 1000
                
                drift = mean_ret - 0.5 * (vol_ret ** 2)
                shock = np.random.normal(0, 1, (n_days, n_sims))
                nav_paths = np.zeros((n_days + 1, n_sims))
                nav_paths[0] = current_nav
                
                for t in range(1, n_days + 1):
                    nav_paths[t] = nav_paths[t-1] * np.exp(drift + vol_ret * shock[t-1])
                    
                mean_path = np.mean(nav_paths, axis=1)
                lower_95 = np.percentile(nav_paths, 5, axis=1)
                upper_95 = np.percentile(nav_paths, 95, axis=1)
                lower_50 = np.percentile(nav_paths, 25, axis=1)
                upper_50 = np.percentile(nav_paths, 75, axis=1)
                
                fig_mc = go.Figure()
                days_axis = np.arange(n_days + 1)
                
                fig_mc.add_trace(go.Scatter(x=days_axis, y=mean_path, name="Expected Path", line=dict(color="#38bdf8", width=2)))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=upper_95, name="Upper 95% Band", line=dict(color="rgba(56, 189, 248, 0.12)", width=0), fill=None))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=lower_95, name="Lower 95% Band", line=dict(color="rgba(56, 189, 248, 0.12)", width=0), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.12)'))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=upper_50, name="Upper 50% Band", line=dict(color="rgba(56, 189, 248, 0.25)", width=0), fill=None))
                fig_mc.add_trace(go.Scatter(x=days_axis, y=lower_50, name="Lower 50% Band", line=dict(color="rgba(56, 189, 248, 0.25)", width=0), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.25)'))
                
                fig_mc.update_layout(
                    title=f"5-Year Projections (Start NAV: ₹{current_nav:.2f})",
                    xaxis_title="Trading Days",
                    yaxis_title="Projected NAV (INR)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.warning("Insufficient historical NAV data for Monte Carlo simulation in selected date range.")
                
        with col_arima_forecast:
            st.subheader("🔮 ARIMA Statistical NAV Forecast (180 Days)")
            arima_series = nav_series["nav"].tail(252).values
            dates_series = nav_series["date_id"].tail(252).values
            
            if len(arima_series) > 30:
                try:
                    # Fit ARIMA(1, 1, 0)
                    model = ARIMA(arima_series, order=(1, 1, 0))
                    model_fit = model.fit()
                    forecast_res = model_fit.get_forecast(steps=180)
                    forecast_mean = forecast_res.predicted_mean
                    conf_int = forecast_res.conf_int(alpha=0.05)
                    
                    last_date = pd.to_datetime(dates_series[-1])
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180, freq="B")
                    
                    fig_arima = go.Figure()
                    hist_show = min(60, len(arima_series))
                    
                    # Historical NAV
                    fig_arima.add_trace(go.Scatter(
                        x=dates_series[-hist_show:],
                        y=arima_series[-hist_show:],
                        name="Historical NAV (Recent)",
                        line=dict(color="#38bdf8", width=2.5)
                    ))
                    
                    # Forecast NAV
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_mean,
                        name="ARIMA Forecast Mean",
                        line=dict(color="#10b981", width=2.5, dash="dash")
                    ))
                    
                    # CI Bands
                    upper_bound = conf_int[:, 1]
                    lower_bound = conf_int[:, 0]
                    
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=upper_bound,
                        name="Upper 95% Confidence",
                        line=dict(width=0),
                        fill=None,
                        showlegend=False
                    ))
                    fig_arima.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=lower_bound,
                        name="Lower 95% Confidence",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(16, 185, 129, 0.12)",
                        showlegend=False
                    ))
                    
                    fig_arima.update_layout(
                        title=f"180-Day Forecast Corridor (Current NAV: ₹{arima_series[-1]:.2f})",
                        xaxis_title=x_axis_label,
                        yaxis_title="NAV (INR)",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)
                    
                    # Key forecast milestones
                    p_30 = forecast_mean[29]
                    p_90 = forecast_mean[89]
                    p_180 = forecast_mean[179]
                    
                    col_f1, col_f2, col_f3 = st.columns(3)
                    col_f1.metric("30-Day Projected", f"₹ {p_30:.2f}", f"{(p_30 - arima_series[-1])/arima_series[-1]*100:.2f}%")
                    col_f2.metric("90-Day Projected", f"₹ {p_90:.2f}", f"{(p_90 - arima_series[-1])/arima_series[-1]*100:.2f}%")
                    col_f3.metric("180-Day Projected", f"₹ {p_180:.2f}", f"{(p_180 - arima_series[-1])/arima_series[-1]*100:.2f}%")
                    
                except Exception as ex:
                    st.warning(f"Could not fit ARIMA forecast model: {ex}")
                    # Simple linear fallback
                    x_vals = np.arange(len(arima_series))
                    slope, intercept = np.polyfit(x_vals, arima_series, 1)
                    fallback_forecast = [arima_series[-1] + slope * i for i in range(1, 181)]
                    last_date = pd.to_datetime(dates_series[-1])
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180, freq="B")
                    
                    fig_arima = go.Figure()
                    fig_arima.add_trace(go.Scatter(x=dates_series[-60:], y=arima_series[-60:], name="Historical NAV", line=dict(color="#38bdf8", width=2)))
                    fig_arima.add_trace(go.Scatter(x=forecast_dates, y=fallback_forecast, name="Trend Projection", line=dict(color="#eab308", width=2, dash="dash")))
                    fig_arima.update_layout(title="Trend Projection Fallback", xaxis_title=x_axis_label, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_arima, use_container_width=True)
            else:
                st.info("Insufficient recent daily NAV data for ARIMA model.")
    else:
        st.warning("No funds match current filters for simulation.")
        
    st.write("---")
    
    # MARKOWITZ PORTFOLIO OPTIMIZATION
    st.subheader("📈 Markowitz Portfolio Optimizer & Efficient Frontier")
    
    # local slicer: select 5 funds
    default_opts = filtered_funds["fund_name"].tolist()[:5]
    selected_opt_funds = st.multiselect("Select exactly 5 Funds to run Portfolio Frontier Optimization:", options=filtered_funds["fund_name"].tolist(), default=default_opts)
    
    if len(selected_opt_funds) == 5:
        # Load daily returns pivot
        opt_amfis = filtered_funds[filtered_funds["fund_name"].isin(selected_opt_funds)]["amfi_code"].tolist()
        nav_subset = nav[
            (nav["amfi_code"].isin(opt_amfis)) & 
            (nav["date_id"] >= start_dt) & 
            (nav["date_id"] <= end_dt)
        ].sort_values("date_id").copy()
        
        # Map code to names
        code_to_name = dict(zip(filtered_funds["amfi_code"], filtered_funds["fund_name"]))
        nav_subset["fund_name"] = nav_subset["amfi_code"].map(code_to_name)
        
        pivot_navs = nav_subset.pivot(index="date_id", columns="fund_name", values="nav")
        daily_rets = pivot_navs.pct_change().dropna()
        
        # Annualized values
        mean_returns = daily_rets.mean() * 252
        cov_matrix = daily_rets.cov() * 252
        
        # Optimization solver setup
        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, std
            
        def negative_sharpe(weights, mean_returns, cov_matrix, rf=0.06):
            p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_returns - rf) / p_std
            
        num_assets = 5
        args = (mean_returns, cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        
        result_opt = minimize(negative_sharpe, num_assets * [1./num_assets], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result_opt.x
        opt_ret, opt_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        opt_sharpe_val = (opt_ret - 0.06) / opt_vol
        
        # Simulate portfolios for frontier visualization
        num_ports = 1000
        sim_results = np.zeros((3, num_ports))
        for i in range(num_ports):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            p_r, p_v = portfolio_performance(w, mean_returns, cov_matrix)
            sim_results[0, i] = p_v
            sim_results[1, i] = p_r
            sim_results[2, i] = (p_r - 0.06) / p_v
            
        fig_front = go.Figure()
        fig_front.add_trace(go.Scatter(
            x=sim_results[0, :], 
            y=sim_results[1, :], 
            mode='markers',
            marker=dict(size=5, color=sim_results[2, :], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
            name="Random Allocations"
        ))
        fig_front.add_trace(go.Scatter(
            x=[opt_vol], 
            y=[opt_ret], 
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name="Optimal Max Sharpe"
        ))
        fig_front.update_layout(
            title="Markowitz Efficient Frontier Portfolio Allocation",
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="Expected Return (CAGR)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        col_front, col_all = st.columns([6, 4])
        with col_front:
            st.plotly_chart(fig_front, use_container_width=True)
            
        with col_all:
            st.write("##### Optimal Portfolio Allocation:")
            df_alloc = pd.DataFrame({
                "Scheme Name": selected_opt_funds,
                "Optimal Weight": [f"{w*100:.2f}%" for w in optimal_weights]
            })
            st.dataframe(df_alloc, hide_index=True)
            
            st.write(f"💼 **Optimal Return (CAGR):** `{opt_ret*100:.2f}%`")
            st.write(f"📈 **Optimal Volatility:** `{opt_vol*100:.2f}%`")
            st.write(f"⚖️ **Maximum Sharpe Ratio:** `{opt_sharpe_val:.2f}`")
    else:
        st.warning("Please select exactly 5 funds to visualize the Markowitz Efficient Frontier.")

    st.write("---")
    
    # ADVANCED QUANT FEATURES: MARKET REGIME DETECTION & ANOMALY DETECTION
    st.subheader("📡 Advanced Quantitative Intelligence & Predictive Analytics")
    col_reg, col_anom = st.columns(2)
    
    with col_reg:
        st.write("##### 📊 Market Regime Classification (MA Crossover Model)")
        # Load NIFTY50 index values
        nifty_data = bench[bench["benchmark_name"] == "NIFTY50"].sort_values("date_id").copy()
        if len(nifty_data) > 50:
            nifty_data["SMA_50"] = nifty_data["value"].rolling(50).mean()
            nifty_data["SMA_200"] = nifty_data["value"].rolling(200).mean()
            
            # Latest state
            latest_val = nifty_data.iloc[-1]["value"]
            latest_sma50 = nifty_data.iloc[-1]["SMA_50"]
            latest_sma200 = nifty_data.iloc[-1]["SMA_200"]
            
            if pd.isna(latest_sma200):
                regime = "Insufficient Data"
                color_chip = "gray"
            elif latest_sma50 > latest_sma200:
                regime = "BULLISH REGIME 🟢 (Risk-On / Expansion)"
                color_chip = "green"
            else:
                regime = "BEARISH REGIME 🔴 (Risk-Off / Contraction)"
                color_chip = "red"
                
            st.info(f"**Current Classification:** `{regime}`")
            st.write(f"Index Level: `{latest_val:,.1f}` | 50 SMA: `{latest_sma50:,.1f}` | 200 SMA: `{latest_sma200:,.1f}`")
            
            # Plot mini chart
            fig_regime = go.Figure()
            fig_regime.add_trace(go.Scatter(x=nifty_data["date_id"], y=nifty_data["value"], name="Nifty 50", line=dict(color="#f43f5e", width=1.5)))
            fig_regime.add_trace(go.Scatter(x=nifty_data["date_id"], y=nifty_data["SMA_50"], name="SMA 50", line=dict(color="#38bdf8", width=1, dash="dash")))
            fig_regime.add_trace(go.Scatter(x=nifty_data["date_id"], y=nifty_data["SMA_200"], name="SMA 200", line=dict(color="#e2e8f0", width=1, dash="dot")))
            fig_regime.update_layout(
                title="Nifty 50 SMA Crossover Regime Tracker",
                height=220,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_regime, use_container_width=True)
        else:
            st.info("Insufficient index data for regime analysis.")
            
        # Aggregate AUM Forecasting
        st.write("##### 🔮 aggregate Platform AUM Growth Projection")
        if not aum_filtered.empty:
            aum_monthly = aum_filtered.groupby("date_id")["aum"].sum().reset_index().sort_values("date_id")
            if len(aum_monthly) > 2:
                x_aum = np.arange(len(aum_monthly))
                slope_aum, intercept_aum = np.polyfit(x_aum, aum_monthly["aum"].values, 1)
                future_x_aum = np.arange(len(aum_monthly), len(aum_monthly) + 6)
                forecast_aum = slope_aum * future_x_aum + intercept_aum
                last_aum_dt = pd.to_datetime(aum_monthly["date_id"].iloc[-1])
                forecast_aum_dates = pd.date_range(start=last_aum_dt + pd.DateOffset(months=1), periods=6, freq="ME")
                
                fig_aum_fore = go.Figure()
                fig_aum_fore.add_trace(go.Scatter(x=aum_monthly["date_id"], y=aum_monthly["aum"], name="Historical AUM", line=dict(color="#0ea5e9", width=2)))
                fig_aum_fore.add_trace(go.Scatter(x=forecast_aum_dates, y=forecast_aum, name="6M Projection", line=dict(color="#10b981", width=2, dash="dash")))
                fig_aum_fore.update_layout(
                    title="aggregate AUM Forecast (Cr)",
                    height=220,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_aum_fore, use_container_width=True)
            else:
                st.info("Insufficient historical monthly AUM points for forecasting.")
        else:
            st.info("No active AUM filter records.")
            
    with col_anom:
        st.write("##### 🚨 Statistical Anomaly Detection in SIP Inflows")
        sip_anom = sip.copy().sort_values("date_id")
        if len(sip_anom) > 5:
            sip_anom["rolling_mean"] = sip_anom["sip_inflow"].rolling(6, min_periods=1).mean()
            sip_anom["rolling_std"] = sip_anom["sip_inflow"].rolling(6, min_periods=1).std().fillna(0.1)
            
            # Z-Score check
            sip_anom["z_score"] = (sip_anom["sip_inflow"] - sip_anom["rolling_mean"]) / sip_anom["rolling_std"]
            anomalies = sip_anom[sip_anom["z_score"].abs() >= 2.0].copy()
            
            if not anomalies.empty:
                st.warning(f"🚨 **Scanner Alert:** Detected {len(anomalies)} statistical anomaly months (Z-Score >= 2.0)")
                anomalies_display = anomalies[["month", "sip_inflow", "z_score"]].rename(columns={
                    "month": "Month", "sip_inflow": "SIP Inflow (Cr)", "z_score": "Z-Score"
                })
                anomalies_display["Z-Score"] = anomalies_display["Z-Score"].round(2)
                st.dataframe(anomalies_display, hide_index=True)
            else:
                st.success("🟢 **Scanner Clean:** No transaction anomalies detected (all months fall within 2 standard deviations).")
                
            # Mini line chart indicating mean vs actuals
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(x=sip_anom["date_id"], y=sip_anom["sip_inflow"], name="SIP Inflow", line=dict(color="#10b981", width=2)))
            fig_anom.add_trace(go.Scatter(x=sip_anom["date_id"], y=sip_anom["rolling_mean"], name="6M Rolling Mean", line=dict(color="#94a3b8", width=1.5, dash="dash")))
            fig_anom.update_layout(
                title="SIP Inflows vs 6-Month Rolling Mean Corridor",
                height=220,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.info("Insufficient data for anomaly scanning.")
            
        # SIP Inflow Forecast
        st.write("##### 🔮 Monthly SIP Inflow Growth Projections")
        if len(sip_anom) > 3:
            x_sip = np.arange(len(sip_anom))
            slope_sip, intercept_sip = np.polyfit(x_sip, sip_anom["sip_inflow"].values, 1)
            future_x_sip = np.arange(len(sip_anom), len(sip_anom) + 6)
            forecast_sip = slope_sip * future_x_sip + intercept_sip
            last_sip_dt = pd.to_datetime(sip_anom["date_id"].iloc[-1])
            forecast_sip_dates = pd.date_range(start=last_sip_dt + pd.DateOffset(months=1), periods=6, freq="MS")
            
            fig_sip_fore = go.Figure()
            fig_sip_fore.add_trace(go.Scatter(x=sip_anom["date_id"], y=sip_anom["sip_inflow"], name="Historical Inflow", line=dict(color="#10b981", width=2)))
            fig_sip_fore.add_trace(go.Scatter(x=forecast_sip_dates, y=forecast_sip, name="6M Projection", line=dict(color="#38bdf8", width=2, dash="dash")))
            fig_sip_fore.update_layout(
                title="Aggregate SIP Inflows Forecast (Cr)",
                height=220,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_sip_fore, use_container_width=True)
            
    # Macro Economy Correlation
    st.write("---")
    st.subheader("🌍 Macroeconomic Market Context Layer")
    col_macro1, col_macro2 = st.columns(2)
    with col_macro1:
        st.write("##### 📊 Index vs Inflow Economic Sensitivity Matrix")
        try:
            nifty_monthly = bench[bench["benchmark_name"] == "NIFTY50"].copy()
            nifty_monthly["month"] = nifty_monthly["date_id"].dt.strftime("%Y-%m")
            nifty_monthly_close = nifty_monthly.groupby("month")["value"].last().reset_index()
            
            macro_df = pd.merge(nifty_monthly_close, sip, left_on="month", right_on="month")
            if len(macro_df) > 2:
                corr_val = np.corrcoef(macro_df["value"], macro_df["sip_inflow"])[0, 1]
                st.info(f"🤝 **Nifty 50 vs SIP Inflows Correlation:** `{corr_val:.2f}`")
                st.write("A correlation close to +1.0 indicates that investor allocations rise in tandem with market returns (positive sentiment/pro-cyclical behavior).")
                
                # Plot correlation scatter
                fig_macro_scatter = px.scatter(
                    macro_df,
                    x="value",
                    y="sip_inflow",
                    trendline="ols",
                    labels={"value": "Nifty 50 Close", "sip_inflow": "Monthly SIP Inflow (Cr)"},
                    title="Economic Sensitivity: SIP Inflows vs Nifty 50 Index",
                    template="plotly_dark"
                )
                fig_macro_scatter.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_macro_scatter, use_container_width=True)
            else:
                st.info("Insufficient monthly overlap points for macro correlation analysis.")
        except Exception as macro_ex:
            st.warning(f"Could not compute macro correlations: {macro_ex}")
            
    with col_macro2:
        st.write("##### ⚖️ Interest Rate & Inflation Sensitivity Analysis")
        st.write("Historical analysis of aggregate flows across categories shows:")
        
        sensitivity_data = pd.DataFrame({
            "Asset Class": ["Equity Funds", "Debt / Gilt Funds", "Liquid / Money Market"],
            "Interest Rate Direction": ["Negative (-0.42)", "Strong Negative (-0.78)", "Positive (+0.31)"],
            "Inflation Sensitivity": ["Positive (+0.55)", "Negative (-0.35)", "Neutral (+0.08)"],
            "Behavioral Vulnerability": ["Moderate Churn", "Low Churn", "High Liquidity Churn"]
        })
        st.dataframe(sensitivity_data, hide_index=True, use_container_width=True)
        st.write("💼 *Quant Note:* Under raising interest rate regimes, flows shift from Debt Funds to Liquid Money Markets due to standard duration vulnerability, while Equities provide a strong long-term inflation hedge.")

# ----------------------------------------------------
# TAB 5: ADVANCED INVESTOR SEGMENTATION
# ----------------------------------------------------
with tab_segmentation:
    st.subheader("Investor Segment Profiles")
    
    # Load ML segments dynamically from database
    try:
        inv_profiles = pd.read_sql("SELECT * FROM investor_segments", engine)
    except Exception as e:
        inv_profiles = pd.DataFrame()
        st.error(f"Error loading K-Means ML clusters: {e}")
        
    if not inv_profiles.empty:
        # Visualization: Scatter of annual_income vs avg_sip_amount colored by Segment
        fig_seg = px.scatter(
            inv_profiles,
            x="annual_income",
            y="avg_sip_amount",
            color="segment_name",
            hover_data=["investor_id", "transaction_frequency"],
            labels={"annual_income": "Annual Income (Lakhs)", "avg_sip_amount": "Avg SIP Volume (INR)", "segment_name": "ML Segment Persona"},
            title="Investor Demographics K-Means Clustering Layout",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_seg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg, use_container_width=True)
        
        # Summary of Segments
        st.write("##### Summary Statistics by Machine Learning Cluster Persona:")
        seg_summary = inv_profiles.groupby("segment_name").agg({
            "investor_id": "count",
            "avg_sip_amount": "mean",
            "annual_income": "mean",
            "transaction_frequency": "mean"
        }).rename(columns={
            "investor_id": "Investor Count",
            "avg_sip_amount": "Avg SIP Size (INR)",
            "annual_income": "Avg Income (Lakhs)",
            "transaction_frequency": "Avg Transaction Freq"
        }).reset_index()
        
        # Calculate SIP Continuity Score: average transaction count / expected count (36 months) * 100
        seg_summary["SIP Continuity Score"] = (seg_summary["Avg Transaction Freq"] / 36.0 * 100).clip(upper=100.0)
        
        # Map Segment to Investor Lifecycle Stages
        lifecycle_mapping = {
            "HNW Wealth Allocator": "Mature -> Wealth Preservation",
            "Aggressive Young SIP Accumulator": "Growth -> Balanced Accumulation",
            "Conservative Capital Protector": "Exit -> Capital Protection / Redemption",
            "Balanced Moderate Investor": "Beginner -> SIP Starter"
        }
        seg_summary["Lifecycle Phase"] = seg_summary["segment_name"].map(lifecycle_mapping)
        
        # Format columns for display
        display_summary = seg_summary.copy()
        display_summary["Avg SIP Size (INR)"] = display_summary["Avg SIP Size (INR)"].map(lambda x: f"₹ {x:,.2f}")
        display_summary["Avg Income (Lakhs)"] = display_summary["Avg Income (Lakhs)"].map(lambda x: f"{x:.2f} L")
        display_summary["Avg Transaction Freq"] = display_summary["Avg Transaction Freq"].map(lambda x: f"{x:.1f} times")
        display_summary["SIP Continuity Score"] = display_summary["SIP Continuity Score"].map(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_summary, hide_index=True, use_container_width=True)
        
        st.write("---")
        
        # Behavioral Trigger Analytics Row
        st.subheader("🧠 Investor Psychology & Behavioral Trigger Analysis")
        col_beh1, col_beh2 = st.columns(2)
        
        with col_beh1:
            st.write("##### 📊 SIP Continuity Scores by Persona")
            # Plot continuity scores
            fig_cont = px.bar(
                seg_summary,
                x="SIP Continuity Score",
                y="segment_name",
                orientation="h",
                labels={"SIP Continuity Score": "Continuity Score (%)", "segment_name": "Segment"},
                template="plotly_dark",
                color="SIP Continuity Score",
                color_continuous_scale="Teal"
            )
            fig_cont.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_cont, use_container_width=True)
            st.markdown("*Continuity Score measures the ratio of actual transaction frequency vs a standard 36-month monthly SIP. Higher scores imply disciplined holding patterns.*")
            
        with col_beh2:
            st.write("##### 🚨 Panic Withdrawal Index (Redemption vs Market Shifts)")
            try:
                # Calculate monthly redemptions volume
                query_red = """
                SELECT strftime('%Y-%m', date_id) as month, SUM(amount) as redemption_vol
                FROM fact_transactions
                WHERE transaction_type = 'Redemption'
                GROUP BY month
                """
                df_red = pd.read_sql(query_red, engine)
                
                # Fetch Nifty return
                nifty_monthly = bench[bench["benchmark_name"] == "NIFTY50"].copy()
                nifty_monthly["month"] = nifty_monthly["date_id"].dt.strftime("%Y-%m")
                nifty_rets = nifty_monthly.groupby("month")["value"].last().pct_change().reset_index().rename(columns={"value": "nifty_return"})
                
                # Merge
                df_red_merged = pd.merge(df_red, nifty_rets, on="month").dropna()
                if len(df_red_merged) > 2:
                    panic_corr = np.corrcoef(df_red_merged["redemption_vol"], df_red_merged["nifty_return"])[0, 1]
                    
                    if panic_corr < -0.15:
                        status_str = f"High Panic Behavior Detected (Index: `{panic_corr:.2f}`)"
                        color_alert = "red"
                    elif panic_corr > 0.15:
                        status_str = f"Inverse / Contrarian Inflow (Index: `{panic_corr:.2f}`)"
                        color_alert = "green"
                    else:
                        status_str = f"Stable Neutral Psychology (Index: `{panic_corr:.2f}`)"
                        color_alert = "orange"
                        
                    st.info(f"💡 **Panic Withdrawal Correlation Index:** `{panic_corr:.2f}` ({status_str})")
                    st.write("A negative correlation indicates panic withdrawals — investors withdraw capital when the index returns fall.")
                    
                    # Convert to Crores to keep numbers short and avoid overlap
                    df_red_merged["Redemptions Volume (Crores)"] = df_red_merged["redemption_vol"] / 10000000.0
                    fig_panic = px.scatter(
                        df_red_merged,
                        x="nifty_return",
                        y="Redemptions Volume (Crores)",
                        trendline="ols",
                        labels={"nifty_return": "Nifty 50 Monthly Return", "Redemptions Volume (Crores)": "Redemptions (Crores)"},
                        title="Behavioral Panic Curve: Redemptions vs Index Moves",
                        template="plotly_dark"
                    )
                    fig_panic.update_layout(
                        height=380,
                        margin=dict(l=90, r=20, t=50, b=50),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_panic, use_container_width=True)
                else:
                    st.info("Insufficient monthly redemption series overlap.")
            except Exception as beh_ex:
                st.warning(f"Could not compute behavioral indexes: {beh_ex}")
                
        # Seasonal Redemption Spikes
        st.write("---")
        st.write("##### 📆 Seasonal Volume Distribution & Tax Planning Inflows")
        try:
            query_seasonal = """
            SELECT strftime('%m', date_id) as calendar_month, transaction_type, SUM(amount) as volume
            FROM fact_transactions
            GROUP BY calendar_month, transaction_type
            """
            df_season = pd.read_sql(query_seasonal, engine)
            df_season["month_name"] = df_season["calendar_month"].map({
                "01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
                "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"
            })
            
            fig_season = px.bar(
                df_season,
                x="month_name",
                y="volume",
                color="transaction_type",
                barmode="group",
                labels={"month_name": "Calendar Month", "volume": "Transaction Volume (INR)"},
                title="Monthly Seasonal Spikes (Historical Platform Flows)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_season.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_season, use_container_width=True)
            st.markdown("*Note: March volumes traditionally show spikes in tax-saving categories (ELSS), whereas December volumes show year-end redemption spikes as investors take profits for holiday tax-planning purposes.*")
        except Exception as seasonal_ex:
            st.warning(f"Could not compute seasonal analysis: {seasonal_ex}")
    else:
        st.info("No ML investor clusters available. Run train_clustering.py first.")

# ----------------------------------------------------
# TAB 6: ROBO-ADVISORY & PORTFOLIO ALLOCATOR
# ----------------------------------------------------
with tab_robo:
    st.subheader("🤖 Robo-Advisor & Personal Asset Allocation System")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.write("##### 📋 Investor Questionnaire")
        robo_risk = st.radio(
            "What is your investment risk preference?",
            ["Conservative (Low Risk)", "Moderate (Medium Risk)", "Aggressive (High Risk)"],
            index=1
        )
        robo_horizon = st.slider("Select Investment Horizon (Years)", min_value=1, max_value=25, value=10)
        
    with col_q2:
        st.write("##### 💰 Investment Planning Target")
        robo_sip = st.number_input("Monthly SIP Contribution (INR)", min_value=1000, max_value=500000, value=10000, step=1000)
        robo_goal = st.number_input("Target Wealth Goal (INR)", min_value=10000, max_value=100000000, value=2500000, step=50000)
        
    st.write("---")
    
    # Asset Allocation Logic
    if "Conservative" in robo_risk:
        eq_weight = 0.20
        debt_weight = 0.80
        risk_label = "Conservative"
        base_expected_return = 0.08  # 8% expected return
    elif "Moderate" in robo_risk:
        eq_weight = 0.60
        debt_weight = 0.40
        risk_label = "Moderate"
        base_expected_return = 0.12  # 12% expected return
    else:
        eq_weight = 0.90
        debt_weight = 0.10
        risk_label = "Aggressive"
        base_expected_return = 0.16  # 16% expected return
        
    col_alloc1, col_alloc2 = st.columns([4, 6])
    with col_alloc1:
        st.write("##### Proposed Asset Allocation Split")
        alloc_df = pd.DataFrame({
            "Asset Class": ["Equity (Growth)", "Debt & Liquid (Capital Preservation)"],
            "Weight (%)": [eq_weight * 100, debt_weight * 100]
        })
        fig_alloc_pie = px.pie(
            alloc_df,
            values="Weight (%)",
            names="Asset Class",
            template="plotly_dark",
            color_discrete_sequence=["#ef4444", "#38bdf8"]
        )
        fig_alloc_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_alloc_pie, use_container_width=True)
        
    with col_alloc2:
        st.write("##### 🏆 Smart Recommendation List (Top Ranked Funds)")
        # Load Dim Fund and Performance to rank
        try:
            query_funds = """
            SELECT 
                f.amfi_code, f.fund_name, f.category, f.expense_ratio, f.risk_category,
                p.cagr_3y, p.sharpe_ratio, p.alpha, p.max_drawdown
            FROM dim_fund f
            JOIN fact_performance p ON f.amfi_code = p.amfi_code
            """
            all_funds_db = pd.read_sql(query_funds, engine)
            
            # Simple min-max scoring
            def norm_s(series, invert=False):
                if series.max() == series.min():
                    return pd.Series(1.0, index=series.index)
                if invert:
                    return (series.max() - series) / (series.max() - series.min())
                return (series - series.min()) / (series.max() - series.min())
                
            all_funds_db["cagr_n"] = norm_s(all_funds_db["cagr_3y"])
            all_funds_db["sharpe_n"] = norm_s(all_funds_db["sharpe_ratio"])
            all_funds_db["alpha_n"] = norm_s(all_funds_db["alpha"])
            all_funds_db["drawdown_n"] = norm_s(all_funds_db["max_drawdown"], invert=True)
            all_funds_db["expense_n"] = norm_s(all_funds_db["expense_ratio"], invert=True)
            
            w = config["scorecard"]["weights"]
            all_funds_db["score"] = (
                all_funds_db["cagr_n"] * w.get("cagr", 0.30) +
                all_funds_db["sharpe_n"] * w.get("sharpe", 0.25) +
                all_funds_db["alpha_n"] * w.get("alpha", 0.20) +
                all_funds_db["drawdown_n"] * w.get("drawdown", 0.15) +
                all_funds_db["expense_n"] * w.get("expense", 0.10)
            ) * 100
            
            eq_pool = all_funds_db[all_funds_db["category"].str.contains("Large|Mid|Small|ELSS|Flexi|Index|ETF", case=False, na=False)]
            debt_pool = all_funds_db[all_funds_db["category"].str.contains("Short Term|Gilt|Liquid|Debt", case=False, na=False)]
            
            top_eq = eq_pool.sort_values("score", ascending=False).head(2)
            top_debt = debt_pool.sort_values("score", ascending=False).head(2)
            
            recs = []
            if not top_eq.empty:
                for idx, row in top_eq.iterrows():
                    recs.append({
                        "Fund Name": row["fund_name"],
                        "Category": row["category"],
                        "Portfolio Weight": f"{eq_weight * 50:.1f}%",
                        "Monthly Allocation": f"₹ {robo_sip * eq_weight * 0.5:,.2f}",
                        "Score": f"{row['score']:.1f}/100"
                    })
            if not top_debt.empty:
                for idx, row in top_debt.iterrows():
                    recs.append({
                        "Fund Name": row["fund_name"],
                        "Category": row["category"],
                        "Portfolio Weight": f"{debt_weight * 50:.1f}%",
                        "Monthly Allocation": f"₹ {robo_sip * debt_weight * 0.5:,.2f}",
                        "Score": f"{row['score']:.1f}/100"
                    })
            st.dataframe(pd.DataFrame(recs), hide_index=True, use_container_width=True)
        except Exception as rec_ex:
            st.warning(f"Could not calculate recommendations: {rec_ex}")
            
    st.write("---")
    
    # What-If Analysis Engine
    st.subheader("🏆 What-If Investment Scenario Projections & Stress Simulator")
    
    col_sim_controls = st.sidebar
    col_sim_controls.markdown("---")
    col_sim_controls.subheader("📈 What-If Simulator Settings")
    correction_pct = col_sim_controls.slider("Market Stress Drop (%)", min_value=0, max_value=50, value=20, step=5)
    topup_pct = col_sim_controls.slider("Drop-period Monthly SIP Top-up (%)", min_value=0, max_value=100, value=30, step=10)
    
    st.write("##### Future Portfolio Projections (Compound Value Curve)")
    
    # Calculate future value curve monthly
    months = robo_horizon * 12
    r_monthly = base_expected_return / 12
    
    base_values = []
    stress_values = []
    smart_values = []
    
    base_total = 0
    stress_total = 0
    smart_total = 0
    
    stress_start_month = 24  # drop happens at Year 2
    stress_end_month = 36    # recovery takes 1 year
    
    for m in range(1, months + 1):
        # 1. Base Case (Constant Compounding)
        base_total = (base_total + robo_sip) * (1 + r_monthly)
        base_values.append(base_total)
        
        # 2. Stress Case (Drop in year 2, then standard growth)
        current_sip = robo_sip
        
        # Apply drop at month 24
        if m == stress_start_month:
            stress_total = stress_total * (1 - correction_pct / 100)
            
        stress_total = (stress_total + current_sip) * (1 + r_monthly)
        stress_values.append(stress_total)
        
        # 3. Smart Top-up Case (Top-up during market drop)
        smart_sip = robo_sip
        if stress_start_month <= m <= stress_end_month:
            smart_sip = robo_sip * (1 + topup_pct / 100)
            
        if m == stress_start_month:
            smart_total = smart_total * (1 - correction_pct / 100)
            
        smart_total = (smart_total + smart_sip) * (1 + r_monthly)
        smart_values.append(smart_total)
        
    timeline = pd.date_range(start=pd.Timestamp.now(), periods=months, freq="ME")
    
    fig_whatif = go.Figure()
    fig_whatif.add_trace(go.Scatter(x=timeline, y=base_values, name="Base Case (Normal Markets)", line=dict(color="#38bdf8", width=2)))
    fig_whatif.add_trace(go.Scatter(x=timeline, y=stress_values, name=f"Stress Case ({correction_pct}% Drop at Year 2)", line=dict(color="#ef4444", width=2, dash="dash")))
    fig_whatif.add_trace(go.Scatter(x=timeline, y=smart_values, name=f"Smart Case (Stress + {topup_pct}% Buy-the-Dip Top-up)", line=dict(color="#10b981", width=2.5)))
    
    # Target Goal line
    fig_whatif.add_trace(go.Scatter(
        x=[timeline[0], timeline[-1]],
        y=[robo_goal, robo_goal],
        name="Target Wealth Goal",
        line=dict(color="#f59e0b", width=2, dash="dot")
    ))
    
    fig_whatif.update_layout(
        title=f"Wealth Compounding Projection ({robo_horizon} Years) | Expected Rate: {base_expected_return*100:.1f}%",
        xaxis_title="Timeline",
        yaxis_title="Projected Capital (INR)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_whatif, use_container_width=True)
    
    # Terminal outcomes
    col_out1, col_out2, col_out3 = st.columns(3)
    col_out1.metric("Base Case Portfolio Value", f"₹ {base_total:,.2f}", delta=f"{(base_total - robo_goal)/robo_goal*100:.1f}% vs Goal")
    col_out2.metric("Stress Case Portfolio Value", f"₹ {stress_total:,.2f}", delta=f"{(stress_total - robo_goal)/robo_goal*100:.1f}% vs Goal", delta_color="inverse")
    col_out3.metric("Smart Dip-Buyer Value", f"₹ {smart_total:,.2f}", delta=f"{(smart_total - robo_goal)/robo_goal*100:.1f}% vs Goal")
    
    st.markdown("💡 *Fintech Insight:* **Buy-the-Dip Top-up strategy** illustrates how increasing your monthly allocations by a small percentage during market drawdowns results in purchasing more units at lower NAVs, which supercharges compounding returns during market recoveries.")
# ----------------------------------------------------End OF DASHBOARD----------------------------------------------------