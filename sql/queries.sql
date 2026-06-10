-- SQL Requirements for Bluestock Mutual Fund Analytics Capstone

-- ==========================================
-- 1. BASIC QUERIES
-- ==========================================
-- Retrieve all fund records
SELECT * FROM dim_fund;

-- Retrieve all investor records
SELECT * FROM dim_investor;


-- ==========================================
-- 2. FILTERING
-- ==========================================
-- Filter funds that are equity-based
SELECT * FROM dim_fund
WHERE category LIKE '%Equity%';

-- Filter transactions of type BUY greater than 1,00,000 INR
SELECT * FROM fact_transactions
WHERE transaction_type = 'BUY' AND amount > 100000;


-- ==========================================
-- 3. AGGREGATION
-- ==========================================
-- Calculate total AUM by Fund House (AMC) for the latest available date
SELECT 
    fund_house, 
    ROUND(SUM(aum), 2) as total_aum_crores,
    COUNT(amfi_code) as funds_count
FROM fact_aum
WHERE date_id = (SELECT MAX(date_id) FROM fact_aum)
GROUP BY fund_house
ORDER BY total_aum_crores DESC;

-- Total SIP inflow per category for the latest month
SELECT 
    f.category,
    ROUND(SUM(s.sip_amount), 2) as total_sip_crores
FROM sip_inflows s
JOIN dim_fund f ON s.amfi_code = f.amfi_code
WHERE s.date_id = (SELECT MAX(date_id) FROM sip_inflows)
GROUP BY f.category
ORDER BY total_sip_crores DESC;


-- ==========================================
-- 4. JOINS
-- ==========================================
-- Combine NAV records with fund dimension details for a specific date
SELECT 
    n.date_id,
    n.amfi_code,
    f.fund_name,
    f.category,
    f.fund_house,
    n.nav
FROM fact_nav n
JOIN dim_fund f ON n.amfi_code = f.amfi_code
WHERE n.date_id = '2026-06-03'
ORDER BY f.category, n.nav DESC;

-- Join Transactions with Investor details and Fund details to see transaction history
SELECT 
    t.transaction_id,
    i.name as investor_name,
    i.state,
    f.fund_name,
    f.category,
    t.transaction_date_id, -- Note: in db schema, date_id was imported as date_id
    t.transaction_type,
    t.amount,
    t.units,
    t.nav
FROM fact_transactions t
JOIN dim_investor i ON t.investor_id = i.investor_id
JOIN dim_fund f ON t.amfi_code = f.amfi_code
LIMIT 10;


-- ==========================================
-- 5. WINDOW FUNCTIONS (IMPORTANT)
-- ==========================================

-- A. Rolling 30-Day NAV Average for a specific fund (SBI Bluechip - AMFI: 119812)
-- Shows how to use AVG() OVER with preceding partition
SELECT 
    date_id,
    nav,
    AVG(nav) OVER (
        ORDER BY date_id 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_30d_avg_nav
FROM fact_nav
WHERE amfi_code = 119812
ORDER BY date_id
LIMIT 60;


-- B. Rank funds within their category by their latest NAV (as a proxy for size/unit value rank)
SELECT 
    f.category,
    f.fund_name,
    n.nav,
    RANK() OVER (
        PARTITION BY f.category 
        ORDER BY n.nav DESC
    ) as nav_rank_in_category
FROM fact_nav n
JOIN dim_fund f ON n.amfi_code = f.amfi_code
WHERE n.date_id = (SELECT MAX(date_id) FROM fact_nav)
ORDER BY f.category, nav_rank_in_category;


-- C. Rolling 7-Day NAV Average for all funds using PARTITION BY and ORDER BY
SELECT 
    amfi_code,
    date_id,
    nav,
    AVG(nav) OVER (
        PARTITION BY amfi_code 
        ORDER BY date_id 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7d_avg_nav
FROM fact_nav
ORDER BY amfi_code, date_id
LIMIT 100;
