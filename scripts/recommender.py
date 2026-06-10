"""
Bluestock Sharpe-Based Fund Recommendation Engine CLI.

This script acts as a command-line interface to retrieve the top 3 funds 
by Sharpe Ratio within a given risk appetite (Low, Moderate, High).

Usage:
    python scripts/recommender.py Moderate
"""

import os
import sys
import sqlite3
import pandas as pd
import pathlib
import yaml

def get_recommendations(risk_appetite="Moderate"):
    """
    Retrieves the top 3 funds by Sharpe Ratio within the matching risk grade.
    
    Args:
        risk_appetite (str): One of ['Low', 'Moderate', 'High'].
        
    Returns:
        pd.DataFrame: Ranks, fund names, categories, risk grades, Sharpe, and CAGR.
    """
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    
    # Load configuration to get the database path
    config_path = base_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        db_path = base_dir / config["database"]["db_path"]
    else:
        db_path = base_dir / "data/db/bluestock_mf.db"
        
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}. Please run the ETL pipeline first.")
        return None
        
    conn = sqlite3.connect(db_path)
    
    # Query performance and fund categories
    query = """
        SELECT f.fund_name, f.category, f.risk_category, p.sharpe_ratio, p.cagr_3y
        FROM dim_fund f
        JOIN fact_performance p ON f.amfi_code = p.amfi_code
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Map risk appetite input to database risk categories
    risk_mapping = {
        "Low": ["Low"],
        "Moderate": ["Moderate", "Moderately High"],
        "High": ["High", "Very High"]
    }
    
    allowed_categories = risk_mapping.get(risk_appetite, ["Moderate", "Moderately High"])
    
    # Filter and sort
    filtered_df = df[df["risk_category"].isin(allowed_categories)].copy()
    recommended_df = filtered_df.sort_values(by="sharpe_ratio", ascending=False).head(3).copy()
    
    if len(recommended_df) == 0:
        return pd.DataFrame()
        
    # Format CAGR as percentage
    recommended_df["cagr_3y"] = (recommended_df["cagr_3y"] * 100).round(2).astype(str) + "%"
    recommended_df["sharpe_ratio"] = recommended_df["sharpe_ratio"].round(4)
    
    # Add rank index
    recommended_df.insert(0, "Rank", range(1, len(recommended_df) + 1))
    
    return recommended_df

def recommend_funds_simple(risk_appetite="Moderate"):
    """
    Prints the formatted Sharpe-based recommendations to standard output.
    
    Args:
        risk_appetite (str): Risk appetite profile.
    """
    df_rec = get_recommendations(risk_appetite)
    if df_rec is None or df_rec.empty:
        print(f"No funds found matching risk profile: {risk_appetite}")
        return
        
    print(df_rec.to_string(index=False, columns=["Rank", "fund_name", "category", "risk_category", "sharpe_ratio", "cagr_3y"]))

if __name__ == "__main__":
    appetite = "Moderate"
    if len(sys.argv) > 1:
        input_val = sys.argv[1].capitalize()
        if input_val in ["Low", "Moderate", "High"]:
            appetite = input_val
        else:
            print("Usage: python recommender.py [Low/Moderate/High]")
            sys.argv.exit(1)
            
    print(f"======================================================================")
    print(f"Bluestock FinTech Recommendation Engine - Risk Profile: {appetite}")
    print(f"======================================================================")
    recommend_funds_simple(appetite)
    print(f"======================================================================\n")
