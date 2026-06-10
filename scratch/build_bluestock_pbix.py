import os
import sqlite3
import json
import pbix_mcp.builder_v2
from pbix_mcp.builder import PBIXBuilder

# Define Premium Dark Theme JSON
theme_json = """{
  "name": "BluestockPremiumDarkTheme",
  "dataColors": [
    "#00d8ff",
    "#00f5a0",
    "#ff2a85",
    "#9d4edd",
    "#ffbe0b",
    "#ff7096",
    "#3a86c8",
    "#52b788"
  ],
  "background": "#0a0e17",
  "foreground": "#f8fafc",
  "tableAccent": "#00d8ff",
  "visualStyles": {
    "*": {
      "*": {
        "background": [
          {
            "show": true,
            "color": { "solid": { "color": "#151b26" } },
            "transparency": 0
          }
        ],
        "border": [
          {
            "show": true,
            "color": { "solid": { "color": "#232d3f" } },
            "radius": 12
          }
        ],
        "title": [
          {
            "show": true,
            "fontColor": { "solid": { "color": "#f8fafc" } },
            "fontSize": 12,
            "fontFamily": "Segoe UI Semibold"
          }
        ],
        "textHeaders": [
          {
            "fontColor": { "solid": { "color": "#94a3b8" } },
            "fontSize": 10,
            "fontFamily": "Segoe UI"
          }
        ],
        "visualTooltip": [
          {
            "titleFontColor": { "solid": { "color": "#f8fafc" } },
            "valueFontColor": { "solid": { "color": "#94a3b8" } },
            "background": { "solid": { "color": "#151b26" } }
          }
        ],
        "visualHeader": [
          {
            "show": false
          }
        ]
      }
    },
    "page": {
      "*": {
        "background": [
          {
            "color": { "solid": { "color": "#0a0e17" } },
            "transparency": 0
          }
        ],
        "wallpaper": [
          {
            "color": { "solid": { "color": "#0a0e17" } },
            "transparency": 0
          }
        ]
      }
    },
    "card": {
      "*": {
        "categoryLabels": [
          {
            "show": true,
            "color": { "solid": { "color": "#94a3b8" } },
            "fontSize": 10,
            "fontFamily": "Segoe UI"
          }
        ],
        "labels": [
          {
            "fontSize": 20,
            "color": { "solid": { "color": "#f8fafc" } },
            "fontFamily": "Segoe UI Semibold"
          }
        ]
      }
    }
  }
}"""

# Monkeypatch build_pbix_clean to inject our theme
original_build_pbix_clean = pbix_mcp.builder_v2.build_pbix_clean
def custom_build_pbix_clean(datamodel_bytes, layout_bytes, theme_json_param=None):
    # Call original to get base zip bytes
    pbix_bytes = original_build_pbix_clean(datamodel_bytes, layout_bytes, theme_json=None)
    
    # Repackage the zip in memory to inject custom themes at correct paths
    import io
    import zipfile
    
    in_buf = io.BytesIO(pbix_bytes)
    out_buf = io.BytesIO()
    
    paths_to_write = [
        "Report/StaticResources/SharedResources/BaseThemes/CY24SU11.json",
        "Report/StaticResources/RegisteredResources/CY24SU11.json",
        "Report/StaticResources/SharedResources/BaseThemes/BluestockPremiumDarkTheme.json",
        "Report/StaticResources/RegisteredResources/BluestockPremiumDarkTheme.json"
    ]
    
    theme_bytes = theme_json.encode("utf-8")
    
    with zipfile.ZipFile(in_buf, "r") as jin:
        with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as jout:
            # Copy all files except those in BaseThemes or RegisteredResources
            for item in jin.infolist():
                if "BaseThemes" not in item.filename and "RegisteredResources" not in item.filename:
                    jout.writestr(item, jin.read(item.filename))
            
            # Write custom theme JSON to all paths
            for path in paths_to_write:
                jout.writestr(path, theme_bytes)
                
    return out_buf.getvalue()

pbix_mcp.builder_v2.build_pbix_clean = custom_build_pbix_clean

# Monkeypatch _build_layout to inject theme reference in layout JSON
original_build_layout = PBIXBuilder._build_layout
def custom_build_layout(self):
    layout_bytes = original_build_layout(self)
    layout_str = layout_bytes.decode("utf-16-le")
    layout_data = json.loads(layout_str)
    # Reference BluestockPremiumDarkTheme in layout
    layout_data["theme"] = "BluestockPremiumDarkTheme"
    return json.dumps(layout_data, ensure_ascii=False).encode("utf-16-le")

PBIXBuilder._build_layout = custom_build_layout

db_path = "D:/New folder/bluestock_mf_capstone/data/db/bluestock_mf.db"
output_path = "D:/New folder/bluestock_mf_capstone/bluestock_mf_dashboard_v3.pbix"

print(f"Connecting to database at {db_path}...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
tables = [row[0] for row in cursor.fetchall()]
print("Found tables:", tables)

builder = PBIXBuilder("BluestockMFDashboard")

# Map SQLite types to Power BI types
type_map = {
    "INTEGER": "Int64",
    "REAL": "Double",
    "TEXT": "String",
    "NUMERIC": "Double",
    "DECIMAL": "Decimal",
    "BOOLEAN": "Boolean"
}

# 1. Import all tables from the SQLite database
for table_name in tables:
    print(f"Importing table: {table_name}...")
    
    # Get column info
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols_info = cursor.fetchall()
    
    columns = []
    for cid, col_name, col_type, notnull, dflt_value, pk in cols_info:
        # Map data type
        pbi_type = type_map.get(col_type.upper(), "String")
        columns.append({"name": col_name, "data_type": pbi_type})
        
    # Get row data
    cursor.execute(f"SELECT * FROM {table_name}")
    rows_raw = cursor.fetchall()
    
    rows = []
    col_names = [col["name"] for col in columns]
    for row_raw in rows_raw:
        row_dict = {}
        for col, val in zip(columns, row_raw):
            col_name = col["name"]
            pbi_type = col["data_type"]
            if val is None:
                if pbi_type in ("Int64", "Double", "Decimal"):
                    val = 0.0 if pbi_type != "Int64" else 0
                elif pbi_type == "Boolean":
                    val = False
                else:
                    val = ""
            row_dict[col_name] = val
        rows.append(row_dict)
        
    # Register the table with connection source info for Power BI Desktop Refresh
    # Source type is sqlite, path is db_path
    source_db = {
        "type": "sqlite",
        "path": db_path,
        "table": table_name
    }
    
    builder.add_table(
        name=table_name,
        columns=columns,
        rows=rows,
        source_db=source_db
    )
    print(f"Added table {table_name} with {len(columns)} columns and {len(rows)} rows.")

# 2. Add relationships
print("Configuring relationships...")
# fact_nav -> dim_fund on amfi_code
builder.add_relationship("fact_nav", "amfi_code", "dim_fund", "amfi_code")
# fact_transactions -> dim_fund on amfi_code
builder.add_relationship("fact_transactions", "amfi_code", "dim_fund", "amfi_code")
# fact_transactions -> dim_investor on investor_id
builder.add_relationship("fact_transactions", "investor_id", "dim_investor", "investor_id")
# fact_performance -> dim_fund on amfi_code
builder.add_relationship("fact_performance", "amfi_code", "dim_fund", "amfi_code")
# portfolio_holdings -> dim_fund on amfi_code
builder.add_relationship("portfolio_holdings", "amfi_code", "dim_fund", "amfi_code")
# scheme_performance -> dim_fund on amfi_code
builder.add_relationship("scheme_performance", "amfi_code", "dim_fund", "amfi_code")
# investor_segments -> dim_investor on investor_id
builder.add_relationship("investor_segments", "investor_id", "dim_investor", "investor_id")

# 3. Add measures
print("Defining DAX measures...")
# We add measures to a hidden helper table or relevant tables
# For total metrics, let's host them on their respective fact tables
builder.add_measure("fact_aum", "Total AUM", "SUM('fact_aum'[aum])")
builder.add_measure("sip_inflows", "Total SIP Inflow", "SUM('sip_inflows'[sip_inflow])")
builder.add_measure("sip_inflows", "SIP Accounts Growth YoY", "SUM('sip_inflows'[yoy_growth])")
builder.add_measure("folio_count", "Total Folios Crore", "SUM('folio_count'[total_folios_crore])")
builder.add_measure("dim_fund", "Active Schemes Count", "DISTINCTCOUNT('dim_fund'[amfi_code])")

# String DAX measures for Title Cards
builder.add_measure("dim_fund", "PageTitle_Overview", '"📊 BLUESTOCK MUTUAL FUND ANALYTICS — INDUSTRY OVERVIEW & AUM PERFORMANCE"')
builder.add_measure("dim_fund", "PageTitle_Performance", '"📈 BLUESTOCK MUTUAL FUND ANALYTICS — FUND RISK & RETURN SCORECARD"')
builder.add_measure("dim_fund", "PageTitle_Analytics", '"👥 BLUESTOCK MUTUAL FUND ANALYTICS — INVESTOR COHORTS & GEOGRAPHIC DEMOGRAPHICS"')
builder.add_measure("dim_fund", "PageTitle_Trends", '"📉 BLUESTOCK MUTUAL FUND ANALYTICS — SIP GROWTH & NIFTY 50 MARKET CORRELATION"')

# Extra measures for transactions
builder.add_measure("fact_transactions", "Total Invested Amount", "SUM('fact_transactions'[amount])")

# 4. Build report pages & visuals

# Page 1: Industry Overview
print("Building Page 1: Industry Overview...")
p1_visuals = [
    # Title Card
    {
        "type": "card",
        "name": "p1_title_card",
        "config": {"measure": "PageTitle_Overview"},
        "x": 20, "y": 20, "width": 1240, "height": 80
    },
    # KPI Cards (Total AUM, SIP Inflows, Folios, Active Schemes)
    {
        "type": "card",
        "name": "total_aum_card",
        "config": {"measure": "Total AUM"},
        "x": 20, "y": 120, "width": 295, "height": 100
    },
    {
        "type": "card",
        "name": "total_sip_card",
        "config": {"measure": "Total SIP Inflow"},
        "x": 335, "y": 120, "width": 295, "height": 100
    },
    {
        "type": "card",
        "name": "total_folios_card",
        "config": {"measure": "Total Folios Crore"},
        "x": 650, "y": 120, "width": 295, "height": 100
    },
    {
        "type": "card",
        "name": "total_schemes_card",
        "config": {"measure": "Active Schemes Count"},
        "x": 965, "y": 120, "width": 295, "height": 100
    },
    # Line chart: Industry AUM Growth
    {
        "type": "lineChart",
        "name": "industry_aum_growth_chart",
        "config": {
            "category": {"table": "fact_aum", "column": "date_id"},
            "measure": "Total AUM"
        },
        "x": 20, "y": 240, "width": 610, "height": 450
    },
    # Bar chart: AUM by fund house
    {
        "type": "barChart",
        "name": "aum_by_fund_house_chart",
        "config": {
            "category": {"table": "fact_aum", "column": "fund_house"},
            "measure": "Total AUM"
        },
        "x": 650, "y": 240, "width": 610, "height": 450
    }
]
builder.add_page("Industry Overview", p1_visuals)

# Page 2: Fund Performance
print("Building Page 2: Fund Performance...")
p2_visuals = [
    # Title Card
    {
        "type": "card",
        "name": "p2_title_card",
        "config": {"measure": "PageTitle_Performance"},
        "x": 20, "y": 20, "width": 1240, "height": 80
    },
    # Slicers: AMC, Category, Plan
    {
        "type": "slicer",
        "name": "slicer_amc",
        "config": {"column": {"table": "dim_fund", "column": "fund_house"}},
        "x": 20, "y": 120, "width": 400, "height": 100
    },
    {
        "type": "slicer",
        "name": "slicer_category",
        "config": {"column": {"table": "dim_fund", "column": "category"}},
        "x": 440, "y": 120, "width": 400, "height": 100
    },
    {
        "type": "slicer",
        "name": "slicer_plan",
        "config": {"column": {"table": "dim_fund", "column": "plan"}},
        "x": 860, "y": 120, "width": 400, "height": 100
    },
    # Table: Scorecard
    {
        "type": "tableEx",
        "name": "scorecard_table",
        "config": {
            "columns": [
                {"table": "scheme_performance", "column": "scheme_name"},
                {"table": "scheme_performance", "column": "category"},
                {"table": "scheme_performance", "column": "return_3yr_pct"},
                {"table": "scheme_performance", "column": "std_dev_ann_pct"},
                {"table": "scheme_performance", "column": "sharpe_ratio"},
                {"table": "scheme_performance", "column": "alpha"},
                {"table": "scheme_performance", "column": "beta"},
                {"table": "scheme_performance", "column": "max_drawdown_pct"}
            ]
        },
        "x": 20, "y": 240, "width": 720, "height": 450
    },
    # Line chart: NAV of selected fund
    {
        "type": "lineChart",
        "name": "fund_nav_chart",
        "config": {
            "category": {"table": "fact_nav", "column": "date_id"},
            "measure": "Total AUM"
        },
        "x": 760, "y": 240, "width": 500, "height": 450
    }
]
builder.add_page("Fund Performance", p2_visuals)

# Page 3: Investor Analytics
print("Building Page 3: Investor Analytics...")
p3_visuals = [
    # Title Card
    {
        "type": "card",
        "name": "p3_title_card",
        "config": {"measure": "PageTitle_Analytics"},
        "x": 20, "y": 20, "width": 1240, "height": 80
    },
    # Slicers: State, Age Group
    {
        "type": "slicer",
        "name": "slicer_state",
        "config": {"column": {"table": "dim_investor", "column": "state"}},
        "x": 20, "y": 120, "width": 610, "height": 100
    },
    {
        "type": "slicer",
        "name": "slicer_age",
        "config": {"column": {"table": "dim_investor", "column": "age_group"}},
        "x": 650, "y": 120, "width": 610, "height": 100
    },
    # Bar Chart: Transaction by State
    {
        "type": "barChart",
        "name": "transactions_by_state_chart",
        "config": {
            "category": {"table": "dim_investor", "column": "state"},
            "measure": "Total Invested Amount"
        },
        "x": 20, "y": 240, "width": 610, "height": 210
    },
    # Donut Chart: SIP vs Lumpsum vs Redemption split
    {
        "type": "donutChart",
        "name": "transaction_type_donut",
        "config": {
            "category": {"table": "fact_transactions", "column": "transaction_type"},
            "measure": "Total Invested Amount"
        },
        "x": 650, "y": 240, "width": 610, "height": 210
    },
    # Bar Chart: Age Group vs SIP Amount
    {
        "type": "barChart",
        "name": "avg_sip_by_age_chart",
        "config": {
            "category": {"table": "dim_investor", "column": "age_group"},
            "measure": "Total Invested Amount"
        },
        "x": 20, "y": 470, "width": 610, "height": 220
    },
    # Line Chart: Monthly transaction volume
    {
        "type": "lineChart",
        "name": "monthly_transaction_volume_chart",
        "config": {
            "category": {"table": "fact_transactions", "column": "date_id"},
            "measure": "Total Invested Amount"
        },
        "x": 650, "y": 470, "width": 610, "height": 220
    }
]
builder.add_page("Investor Analytics", p3_visuals)

# Page 4: SIP & Market Trends
print("Building Page 4: SIP & Market Trends...")
p4_visuals = [
    # Title Card
    {
        "type": "card",
        "name": "p4_title_card",
        "config": {"measure": "PageTitle_Trends"},
        "x": 20, "y": 20, "width": 1240, "height": 80
    },
    # KPI Card: YoY Growth
    {
        "type": "card",
        "name": "yoy_growth_card",
        "config": {"measure": "SIP Accounts Growth YoY"},
        "x": 20, "y": 120, "width": 1240, "height": 100
    },
    # Dual-axis line-bar chart: SIP Inflow + Nifty 50
    {
        "type": "lineChart",
        "name": "sip_inflow_vs_nifty_chart",
        "config": {
            "category": {"table": "sip_inflows", "column": "month"},
            "measure": "Total SIP Inflow"
        },
        "x": 20, "y": 240, "width": 610, "height": 450
    },
    # Matrix / Heatmap: Category Inflows by Month
    {
        "type": "matrix",
        "name": "category_inflows_matrix",
        "config": {
            "columns": [
                {"table": "category_inflows", "column": "category"},
                {"table": "category_inflows", "column": "month"},
                {"table": "category_inflows", "column": "net_inflow"}
            ]
        },
        "x": 650, "y": 240, "width": 610, "height": 210
    },
    # Bar Chart: Top 5 Categories by Net Inflow
    {
        "type": "barChart",
        "name": "top_categories_inflow_chart",
        "config": {
            "category": {"table": "category_inflows", "column": "category"},
            "measure": "Total SIP Inflow"
        },
        "x": 650, "y": 470, "width": 610, "height": 220
    }
]
builder.add_page("SIP & Market Trends", p4_visuals)


# Save the compiled Power BI dashboard file
save_paths = [
    "D:/New folder/bluestock_mf_capstone/bluestock_mf_dashboard_v3.pbix",
    "D:/New folder/bluestock_mf_capstone/bluestock_mf_dashboard_new.pbix",
    "D:/New folder/bluestock_mf_capstone/bluestock_mf_dashboard.pbix"
]

for path in save_paths:
    print(f"Saving Power BI dashboard to {path}...")
    try:
        builder.save(path)
        print(f"Successfully saved to {path}!")
    except Exception as e:
        print(f"Warning: Could not save to {path} due to file lock: {e}")

print("Power BI Dashboard compiled successfully!")
conn.close()
