"""
Machine Learning Clustering Engine for Investor Segmentation.

This script executes K-Means clustering on investor profile and transaction datasets,
segments investors into profiles (Conservative, Aggressive, HNW, Moderate),
and stores the results back to the SQLite analytics database.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
import yaml
import pathlib
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [ML Clustering] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def run_investor_clustering():
    """
    Groups investors into distinct personas using K-Means clustering on client feature profiles.
    
    1. Loads transaction volumes and demographics from SQLite.
    2. Aggregates per-investor metrics (SIP size, frequency, income, age).
    3. Normalizes features and trains a K-Means clustering model.
    4. Categorizes segments into labeled personas (HNW, Aggressive, Balanced, Conservative).
    5. Saves profiles to the SQLite database 'investor_segments' table.
    """
    logger.info("Initializing Machine Learning Investor Clustering Engine (K-Means)...")
    
    # Load configuration settings
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    with open(base_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    db_path = base_dir / config["database"]["db_path"]
    n_clusters = config["ml_settings"]["kmeans_clusters"]
    random_state = config["ml_settings"]["random_state"]
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    # 1. Fetch transaction and investor demographics datasets
    query = """
    SELECT 
        t.investor_id,
        t.amount,
        t.transaction_type,
        i.age_group,
        i.annual_income_lakh
    FROM fact_transactions t
    JOIN dim_investor i ON t.investor_id = i.investor_id
    """
    df = pd.read_sql(query, engine)
    
    if len(df) == 0:
        logger.error("No transaction records found in database. Run ETL first.")
        return
        
    # 2. Extract client feature profiles
    # Age group mapping to numerical midpoints
    age_mapping = {
        "18-25": 21.5,
        "26-35": 30.5,
        "36-45": 40.5,
        "46-55": 50.5,
        "55+": 62.0
    }
    df["age_numeric"] = df["age_group"].map(age_mapping).fillna(35.0)
    
    # Aggregating per investor
    profiles = df.groupby("investor_id").agg({
        "amount": ["count", "mean", "sum"],
        "age_numeric": "first",
        "annual_income_lakh": "first"
    }).reset_index()
    
    profiles.columns = [
        "investor_id", "tx_frequency", "avg_sip_amount", "total_invested", "age", "annual_income"
    ]
    
    # 3. Fit K-Means ML Model
    feature_cols = ["tx_frequency", "avg_sip_amount", "age", "annual_income"]
    X = profiles[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    profiles["segment_id"] = kmeans.fit_predict(X_scaled)
    
    # 4. Generate Data-Driven Persona Labels based on Centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Label mapping rules based on cluster center attributes
    segment_names = {}
    for i in range(n_clusters):
        c_freq, c_size, c_age, c_income = centroids[i]
        
        # High Net-Worth: high income or high average transaction size
        if c_income >= 20.0 or c_size >= 100000:
            segment_names[i] = "HNW Wealth Allocator"
        # Young Aggressive: young age, high transaction frequency (e.g. monthly SIPs)
        elif c_age < 32.0 and c_freq >= 15:
            segment_names[i] = "Aggressive Young SIP Accumulator"
        # Older Conservative: older age, lower transaction sizes
        elif c_age > 45.0 and c_income < 15.0:
            segment_names[i] = "Conservative Capital Protector"
        # Balanced Moderate: everything else
        else:
            segment_names[i] = "Balanced Moderate Investor"
            
    profiles["segment_name"] = profiles["segment_id"].map(segment_names)
    
    # 5. Save results to database
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM investor_segments"))
        
    df_to_save = profiles[[
        "investor_id", "segment_id", "segment_name", "avg_sip_amount", "annual_income", "tx_frequency"
    ]].rename(columns={"tx_frequency": "transaction_frequency"})
    df_to_save.to_sql("investor_segments", engine, if_exists="append", index=False)
    
    logger.info("Investor K-Means clustering completed successfully!")
    
    # Print cluster aggregates to logger
    stats_df = df_to_save.groupby("segment_name").agg({
        "investor_id": "count",
        "avg_sip_amount": "mean",
        "annual_income": "mean",
        "transaction_frequency": "mean"
    })
    logger.info(f"Segment breakdown statistics:\n{stats_df.to_string()}")

def train():
    """
    Entry-point function wrapper for pipeline orchestration.
    """
    run_investor_clustering()

if __name__ == "__main__":
    run_investor_clustering()
