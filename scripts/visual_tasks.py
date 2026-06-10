import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Database path
db_path = "data/db/bluestock_mf.db"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Establish connection
conn = sqlite3.connect(db_path)

# Custom plot styling for Seaborn charts to blend with dark mode
def apply_dark_theme():
    sns.set_theme(style="dark")
    plt.rcParams['figure.facecolor'] = '#090d16'
    plt.rcParams['axes.facecolor'] = '#090d16'
    plt.rcParams['text.color'] = '#f8fafc'
    plt.rcParams['axes.labelcolor'] = '#94a3b8'
    plt.rcParams['xtick.color'] = '#94a3b8'
    plt.rcParams['ytick.color'] = '#94a3b8'
    plt.rcParams['axes.edgecolor'] = '#1e293b'
    plt.rcParams['grid.color'] = '#1e293b'

# ----------------------------------------------------
# TASK 1: Category Inflow Heatmap
# ----------------------------------------------------
def build_task_1():
    print("Generating Task 1: Category Inflow Heatmap...")
    apply_dark_theme()
    
    # Query category inflows
    df = pd.read_sql("SELECT month, category, net_inflow FROM category_inflows", conn)
    
    # Pivot month x category
    pivot_df = df.pivot(index='category', columns='month', values='net_inflow')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f", 
        cmap="coolwarm", 
        ax=ax, 
        cbar_kws={'label': 'Net Inflow (Crores)'},
        linewidths=0.5,
        linecolor="#1e293b"
    )
    
    ax.set_title("Monthly Category-wise Net Inflow Heatmap", fontsize=14, pad=15, color="#f8fafc", weight='bold')
    ax.set_xlabel("Months", fontsize=12, labelpad=10)
    ax.set_ylabel("Fund Categories", fontsize=12, labelpad=10)
    
    # Add interpretation comment as a footnote
    interpretation_text = (
        "Interpretation Note:\n"
        "1. Warm/red colors indicate peak inflow periods; cold/blue colors represent lower allocations.\n"
        "2. Multi-cap and Flexi-cap categories show steady institutional retail interest throughout the months.\n"
        "3. Sectoral and Liquid categories demonstrate high inflow fluctuations due to quarterly tax/redemption cycles."
    )
    fig.text(0.1, -0.15, interpretation_text, fontsize=10, color="#94a3b8", wrap=True)
    
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "category_inflow_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Task 1 completed. Saved to {plot_path}")

# ----------------------------------------------------
# TASK 2: Folio Count Growth Analysis
# ----------------------------------------------------
def build_task_2():
    print("Generating Task 2: Folio Count Growth Analysis...")
    df_folios = pd.read_sql("SELECT month, total_folios_crore FROM folio_count ORDER BY month", conn)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_folios['month'],
        y=df_folios['total_folios_crore'],
        mode='lines+markers',
        name='Total Folios',
        line=dict(color='#0ea5e9', width=3.5),
        marker=dict(size=8, color='#38bdf8', symbol='circle')
    ))
    
    # Calculate percentage growth
    start_val = df_folios['total_folios_crore'].iloc[0]
    end_val = df_folios['total_folios_crore'].iloc[-1]
    growth_pct = ((end_val - start_val) / start_val) * 100
    
    # Annotate peak
    highest_idx = df_folios['total_folios_crore'].idxmax()
    highest_month = df_folios.loc[highest_idx, 'month']
    highest_val = df_folios.loc[highest_idx, 'total_folios_crore']
    
    fig.add_annotation(
        x=highest_month,
        y=highest_val,
        text=f"Peak: {highest_val:.2f} Cr (Dec 2025)",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-45,
        bgcolor="#10b981",
        bordercolor="#059669",
        font=dict(color="white", size=11, family="sans-serif")
    )
    
    # Annotate baseline
    fig.add_annotation(
        x=df_folios['month'].iloc[0],
        y=df_folios['total_folios_crore'].iloc[0],
        text=f"Baseline: {start_val:.2f} Cr (Jan 2022)",
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=45,
        bgcolor="#3b82f6",
        bordercolor="#2563eb",
        font=dict(color="white", size=11, family="sans-serif")
    )
    
    # Mark a middle milestone (crossing 20Cr)
    mid_milestone = df_folios[df_folios['total_folios_crore'] >= 20.0].iloc[0]
    fig.add_annotation(
        x=mid_milestone['month'],
        y=mid_milestone['total_folios_crore'],
        text=f"Crossed 20 Cr ({mid_milestone['month']})",
        showarrow=True,
        arrowhead=2,
        ax=-30,
        ay=40,
        bgcolor="#f59e0b",
        bordercolor="#d97706",
        font=dict(color="white", size=11, family="sans-serif")
    )
    
    fig.update_layout(
        title=dict(
            text=f"Mutual Fund Folio Growth Trend (2022–2025)<br><sup>Platform Total Growth: +{growth_pct:.2f}% ({start_val} Cr to {end_val} Cr)</sup>",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis_title="Timeline",
        yaxis_title="Folio Count (Crores)",
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=60, r=40, t=90, b=60),
        height=600
    )
    
    plot_path = os.path.join(reports_dir, "folio_growth_analysis.html")
    fig.write_html(plot_path)
    print(f"Task 2 completed. Saved interactive HTML to {plot_path}")

# ----------------------------------------------------
# TASK 3: NAV Return Correlation Matrix
# ----------------------------------------------------
def build_task_3():
    print("Generating Task 3: NAV Return Correlation Matrix...")
    apply_dark_theme()
    
    # Select 10 representative funds
    funds_df = pd.read_sql("SELECT amfi_code, fund_name FROM dim_fund LIMIT 10", conn)
    codes = tuple(funds_df['amfi_code'].tolist())
    
    # Load daily NAV history
    nav_df = pd.read_sql(f"SELECT amfi_code, date_id, nav FROM fact_nav WHERE amfi_code IN {codes} ORDER BY date_id", conn)
    
    # Pivot to date x fund matrix
    nav_pivot = nav_df.pivot(index='date_id', columns='amfi_code', values='nav').sort_index()
    
    # Calculate daily returns
    returns_df = nav_pivot.pct_change().dropna()
    
    # Rename columns to short names
    code_to_name = dict(zip(funds_df['amfi_code'], funds_df['fund_name'].str.replace("Regular Plan - Growth", "").str.strip().str.slice(0, 22) + "..."))
    returns_df = returns_df.rename(columns=code_to_name)
    
    # Compute correlation matrix
    corr_matrix = returns_df.corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1.0, 
        vmax=1.0, 
        ax=ax,
        linewidths=0.5,
        linecolor="#1e293b",
        annot_kws={"size": 10, "weight": "bold"}
    )
    
    ax.set_title("NAV Return Correlation Matrix (Top 10 Funds)", fontsize=14, pad=20, color="#f8fafc", weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "nav_correlation_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Task 3 completed. Saved to {plot_path}")

# ----------------------------------------------------
# TASK 4: NAV Trend Analysis (40 Schemes, Daily, Plotly highlights)
# ----------------------------------------------------
def build_task_4():
    print("Generating Task 4: NAV Trend Analysis (40 schemes)...")
    df_nav_40 = pd.read_sql("""
        SELECT n.date_id, f.fund_name, n.nav 
        FROM fact_nav n 
        JOIN dim_fund f ON n.amfi_code = f.amfi_code 
        ORDER BY n.date_id
    """, conn)
    
    df_nav_40_pivot = df_nav_40.pivot(index='date_id', columns='fund_name', values='nav').reset_index()
    
    fig = go.Figure()
    cols_to_plot = [c for c in df_nav_40_pivot.columns if c != 'date_id']
    
    for col_name in cols_to_plot:
        fig.add_trace(go.Scatter(
            x=df_nav_40_pivot['date_id'],
            y=df_nav_40_pivot[col_name],
            name=col_name[:20] + "...",
            mode='lines',
            line=dict(width=1.2),
            hovertemplate="<b>" + col_name[:25] + "</b><br>Date: %{x}<br>NAV: %{y:.2f}<extra></extra>"
        ))
        
    fig.update_layout(
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
            text="Daily NAV Trendlines for all 40 Schemes (2022–2026)",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis_title="Date",
        yaxis_title="Net Asset Value (NAV in INR)",
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
        height=600
    )
    
    plot_path = os.path.join(reports_dir, "nav_trend_analysis_40_schemes.html")
    fig.write_html(plot_path)
    print(f"Task 4 completed. Saved interactive HTML to {plot_path}")

# ----------------------------------------------------
# TASK 5: AUM Grouped Bar Chart by Fund House (Seaborn, Highlight SBI)
# ----------------------------------------------------
def build_task_5():
    print("Generating Task 5: AUM Growth Grouped Bar Chart (Highlight SBI)...")
    apply_dark_theme()
    
    df_aum = pd.read_sql("SELECT date_id, fund_house, aum FROM fact_aum", conn)
    df_aum['year'] = pd.to_datetime(df_aum['date_id']).dt.year
    df_aum = df_aum[df_aum['year'].isin([2022, 2023, 2024, 2025])]
    df_aum_yearly = df_aum.groupby(['year', 'fund_house'])['aum'].last().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_aum_yearly, 
        x="year", 
        y="aum", 
        hue="fund_house", 
        ax=ax,
        palette="viridis"
    )
    
    # Increase Y-limit to make space for the annotation text
    ax.set_ylim(0, 1600000.0)
    
    # Highlight SBI at ₹12.5L Cr dominance in 2025 (x=3 in seaborn)
    ax.annotate(
        "SBI Mutual Fund\nDominance: ₹12.5L Cr (2025)",
        xy=(3.22, 1250000.0),
        xytext=(0.5, 1400000.0), # Move text up and left to avoid bar overlap
        arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.0, headwidth=5, headlength=5),
        color="#10b981",
        weight="bold",
        fontsize=9
    )
    
    # Position legend horizontally below the plot in 3 columns
    ax.legend(
        title="Fund House", 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=3, 
        fontsize=9, 
        title_fontsize=9
    )
    
    ax.set_title("Grouped AUM Standings by Fund House (2022–2025)", fontsize=14, pad=20, color="#f8fafc", weight='bold')
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("AUM (Crores)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "aum_growth_seaborn_grouped_bar.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Task 5 completed. Saved to {plot_path}")

# ----------------------------------------------------
# TASK 6: SIP Inflow Time-Series (Plotly, Annotate Dec 2025 Peak)
# ----------------------------------------------------
def build_task_6():
    print("Generating Task 6: SIP Inflow Time-Series (Annotate Peak)...")
    df_sip = pd.read_sql("SELECT month, sip_inflow FROM sip_inflows ORDER BY month", conn)
    df_sip = df_sip[df_sip['month'] <= '2025-12']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sip['month'],
        y=df_sip['sip_inflow'],
        mode='lines+markers',
        name='SIP Inflow',
        line=dict(color='#10b981', width=3.5),
        marker=dict(size=8, color='#38bdf8', symbol='circle')
    ))
    
    # Annotate Dec 2025 ATH (₹31,002 Cr)
    fig.add_annotation(
        x="2025-12",
        y=31002.0,
        text="🏆 All-Time High: ₹31,002 Cr (Dec 2025)",
        showarrow=True,
        arrowhead=2,
        ax=-60,
        ay=-45,
        bgcolor="#10b981",
        bordercolor="#059669",
        font=dict(color="white", size=11, family="sans-serif")
    )
    
    fig.update_layout(
        title=dict(
            text="Mutual Fund Monthly SIP Inflows Trend (2022–2025)",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis_title="Timeline",
        yaxis_title="SIP Inflows (Crores)",
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=60, r=40, t=80, b=60),
        height=600
    )
    
    plot_path = os.path.join(reports_dir, "sip_inflow_time_series_annotated.html")
    fig.write_html(plot_path)
    print(f"Task 6 completed. Saved interactive HTML to {plot_path}")

if __name__ == "__main__":
    build_task_1()
    build_task_2()
    build_task_3()
    build_task_4()
    build_task_5()
    build_task_6()
    conn.close()
    print("All tasks executed successfully.")
