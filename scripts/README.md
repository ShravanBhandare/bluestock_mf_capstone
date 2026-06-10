# ⚙️ Automation & Execution Scripts

This folder contains all backend Python automation scripts for the ETL process, quant metric computations, ML training, CLI tools, and notebook builders.

## 📁 File Registry
* **[etl_pipeline.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/etl_pipeline.py)**: Extracts raw CSV files, performs standardizing transformations, reindexes/fills missing values, and builds SQLite dimensions/facts.
* **[compute_metrics.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/compute_metrics.py)**: Computes annualized CAGRs, Sharpe/Sortino ratios, Nifty 100 regressions (OLS Alpha/Beta), worst drawdown date ranges, and loads metrics into `fact_performance` and `fact_features`.
* **[train_clustering.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/train_clustering.py)**: Pulls investor demographics and aggregates transactional features to fit a K-Means model, categorizing clients into ML segment personas.
* **[run_day6_tasks.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/run_day6_tasks.py)**: Computes historical 95% VaR/CVaR, rolling Sharpe, cohorts, HHI sector index values, and compiles `notebooks/05_advanced_analytics.ipynb`.
* **[recommender.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/recommender.py)**: Command-line interface offering Sharpe-based fund recommendations for `Low`, `Moderate`, and `High` appetites.
* **[scheduler.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/scheduler.py)**: Active scheduling script that runs in the background to automatically execute ETL and metrics updates daily at 8 PM.
* **[email_automation.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/email_automation.py)**: Automated executive report mailer attaching generated performance reports to target managers, featuring an offline mock backup log system.
* **[build_and_run_notebooks.py](file:///D:/New%20folder/bluestock_mf_capstone/scripts/build_and_run_notebooks.py)**: Automatically generates and executes all 5 Jupyter Notebooks in sequence using `nbconvert`.
