# 📊 Data Warehouse Layer

This directory acts as the data persistence layer of the platform, organizing raw inputs, processed dimension/fact tables, and the SQLite relational database.

## 📁 Directory Structure
* **`raw/`**: Contains the 10 original synthesized CSV files representing AMC master tables, NAV histories, transaction logs, portfolio holdings, AUM scales, and category inflows.
* **`processed/`**: Stores clean, reindexed, forward-filled CSV files generated during the ETL phase. These align exactly with the database schemas.
* **`db/`**: Holds the main SQLite file **`bluestock_mf.db`**, which stores all dimensional facts and models queried by the analytics engine.
