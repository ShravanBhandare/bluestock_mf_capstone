# 📓 Jupyter Notebooks (Development & Prototype Playgrounds)

This folder contains numbered, step-by-step Jupyter Notebooks illustrating the logical progression of the project's analytical layers.

## 📁 Sequence of Notebooks
1. **[01_data_ingestion.ipynb](file:///D:/New%20folder/bluestock_mf_capstone/notebooks/01_data_ingestion.ipynb)**: Tests request-based REST API ingestion parameters from the open public AMFI API (`mfapi.in`).
2. **[02_data_cleaning.ipynb](file:///D:/New%20folder/bluestock_mf_capstone/notebooks/02_data_cleaning.ipynb)**: Details duplicate removals, category mappings, and the reindexing/forward-filling logic to handle weekends/holidays.
3. **[03_eda_analysis.ipynb](file:///D:/New%20folder/bluestock_mf_capstone/notebooks/03_eda_analysis.ipynb)**: Explores data using 19 distinct visual plots (Seaborn and Plotly), evaluating demographics, inflows, and returns distributions.
4. **[04_performance_analytics.ipynb](file:///D:/New%20folder/bluestock_mf_capstone/notebooks/04_performance_analytics.ipynb)**: Prototyping CAGR tables, Sharpe/Sortino ratios, daily OLS linear regressions against the Nifty 100 benchmark, and peak-to-trough worst drawdowns.
5. **[05_advanced_analytics.ipynb](file:///D:/New%20folder/bluestock_mf_capstone/notebooks/05_advanced_analytics.ipynb)**: Outlines 95% historical Value at Risk (VaR), Conditional VaR (Expected Shortfall), behavioral cohort capital growth, SIP continuity gap checks, and sector concentration indices (HHI).

## ⚙️ Building & Running Notebooks
You can rebuild and execute all notebooks cell-by-cell in one command by running:
```bash
python scripts/build_and_run_notebooks.py
```
This runs `nbconvert` programmatically to execute all cells and save the outputs directly inside the notebooks.
