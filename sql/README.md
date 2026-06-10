# 🗄️ SQL Schema & Analytical Queries Layer

This directory contains the database schema set-up definitions and core analytical SQL queries.

## 📁 Key Files
* **[schema.sql](file:///D:/New%20folder/bluestock_mf_capstone/sql/schema.sql)**: Contains relational table schema definitions (Star Schema) establishing primary/foreign keys, datatypes, and integrity checks for dimension tables (`dim_fund`, `dim_date`, `dim_investor`, `investor_segments`) and fact tables (`fact_nav`, `fact_transactions`, `fact_aum`, `fact_performance`, `fact_features`).
* **[queries.sql](file:///D:/New%20folder/bluestock_mf_capstone/sql/queries.sql)**: Stores foundational analytical queries, including rolling moving averages, transaction aggregations, and windowing functions used to validate database models.
