

import pyodbc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── CONNECTION CONFIG — only change SERVER if needed ─────────────────────────
SERVER   = r'DESKTOP-TRG6BMQ'
DATABASE = 'dwh_parapharmacie'
DRIVER   = 'ODBC Driver 17 for SQL Server'    # confirmed from your error log
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    """Open and return a live pyodbc connection (Windows Auth)."""
    conn_str = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"Trusted_Connection=yes;"
        f"TrustServerCertificate=yes;"
    )
    conn = pyodbc.connect(conn_str, timeout=30)
    print(f"✓ Connected  [{DATABASE}] on [{SERVER}]")
    return conn


def run_query(sql: str, conn) -> pd.DataFrame:
    """
    Execute T-SQL and return a pandas DataFrame.
    Uses cursor directly (avoids pandas SQLite detection bug).
    Always pass an already-open connection.
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    df   = pd.DataFrame.from_records(rows, columns=cols)
    print(f"✓ {len(df):,} rows × {len(df.columns)} columns")
    return df


def test_connection():
    """Quick smoke-test — run this first."""
    conn = get_connection()
    df   = run_query("SELECT DB_NAME() AS db, GETDATE() AS server_time", conn)
    print(df.to_string(index=False))
    conn.close()
    print("✓ Connection test passed.")


# ════════════════════════════════════════════════════════════════════════════
# PRE-BUILT QUERIES  (all column names verified against real schema)
# ════════════════════════════════════════════════════════════════════════════

# ── GOAL 1 — Time Series ─────────────────────────────────────────────────────
# Week / Month number / Quarter do NOT exist in Dim_DateTime
# → pull Full_DateTime as DATE, derive all calendar features in Python
QUERY_GOAL1_TIMESERIES = """
SELECT
    CAST(dt.Full_DateTime AS DATE)              AS sale_date,
    dt.Year,
    dt.Month_Name,
    dt.Season,
    dt.Is_Weekend,
    SUM(f.SoldItemsQty)                         AS total_qty_sold,
    SUM(f.TotalSales)                           AS total_revenue,
    SUM(f.InvoicesValue)                        AS total_invoiced,
    COUNT(DISTINCT f.ClientID)                  AS nb_customers,
    COUNT(DISTINCT f.DocumentNumber)            AS nb_transactions,
    AVG(f.TotalSales)                           AS avg_basket_value,
    SUM(f.ReturnedItemsQty)                     AS total_returns_qty,
    SUM(f.ReturnsValue)                         AS total_returns_value,
    SUM(f.DiscountAmount)                       AS total_discounts
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus]   f
JOIN [dwh_parapharmacie].[dbo].[Dim_DateTime]   dt
    ON f.DateID = dt.Date_Time_ID
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
GROUP BY
    CAST(dt.Full_DateTime AS DATE),
    dt.Year, dt.Month_Name, dt.Season, dt.Is_Weekend
ORDER BY sale_date
"""

# ── GOAL 2 — Customer Segmentation ───────────────────────────────────────────
QUERY_GOAL2_SEGMENTATION = """
SELECT
    f.ClientID,
    c.CustomerName,
    c.Type                                      AS customer_type,
    c.RiskLevel,
    COUNT(DISTINCT f.DocumentNumber)            AS nb_transactions,
    SUM(f.SoldItemsQty)                         AS total_qty,
    SUM(f.TotalSales)                           AS total_revenue,
    AVG(f.TotalSales)                           AS avg_basket,
    MAX(f.TotalSales)                           AS max_basket,
    MIN(f.TotalSales)                           AS min_basket,
    SUM(f.DiscountAmount)                       AS total_discounts,
    AVG(f.DiscountPct)                          AS avg_discount_pct,
    SUM(f.ReturnedItemsQty)                     AS total_returns,
    COUNT(DISTINCT f.ProductID)                 AS nb_distinct_products,
    MIN(CAST(dt.Full_DateTime AS DATE))         AS first_purchase,
    MAX(CAST(dt.Full_DateTime AS DATE))         AS last_purchase,
    DATEDIFF(DAY,
        MIN(CAST(dt.Full_DateTime AS DATE)),
        MAX(CAST(dt.Full_DateTime AS DATE)))    AS lifespan_days,
    DATEDIFF(DAY,
        MAX(CAST(dt.Full_DateTime AS DATE)),
        CAST('2025-04-30' AS DATE))             AS recency_days,
    CAST(SUM(CASE WHEN f.IsPayed=1 THEN 1 ELSE 0 END) AS FLOAT)
        / NULLIF(COUNT(f.DocumentNumber),0)     AS payment_rate
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus]   f
JOIN [dwh_parapharmacie].[dbo].[Dim_Customer]   c
    ON f.ClientID = c.CustomerID
JOIN [dwh_parapharmacie].[dbo].[Dim_DateTime]   dt
    ON f.DateID = dt.Date_Time_ID
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
GROUP BY f.ClientID, c.CustomerName, c.Type, c.RiskLevel
"""

# ── GOAL 3 — Creditworthiness Classification ─────────────────────────────────
QUERY_GOAL3_CREDIT = """
SELECT
    f.ClientID,
    c.Type                                      AS customer_type,
    c.RiskLevel,
    COUNT(DISTINCT f.DocumentNumber)            AS nb_invoices,
    SUM(f.InvoicesValue)                        AS total_invoiced,
    SUM(f.AccountsReceivable)                   AS total_receivable,
    SUM(f.TotalSales)                           AS total_sales,
    AVG(CAST(f.PaymentDelaysDays AS FLOAT))     AS avg_payment_delay,
    MAX(f.PaymentDelaysDays)                    AS max_payment_delay,
    SUM(CASE WHEN f.IsPayed=1 THEN 1 ELSE 0 END)  AS nb_paid,
    SUM(CASE WHEN f.IsPayed=0 THEN 1 ELSE 0 END)  AS nb_unpaid,
    CAST(SUM(CASE WHEN f.IsPayed=1 THEN 1 ELSE 0 END) AS FLOAT)
        / NULLIF(COUNT(f.DocumentNumber),0)     AS payment_rate,
    AVG(f.DiscountPct)                          AS avg_discount_pct,
    SUM(f.DiscountAmount)                       AS total_discount,
    AVG(f.TotalSales)                           AS avg_invoice_value,
    MAX(f.TotalSales)                           AS max_invoice_value,
    SUM(f.ReturnedItemsQty)                     AS total_returns,
    COUNT(DISTINCT f.ProductID)                 AS nb_products_bought,
    CASE
        WHEN AVG(CAST(f.PaymentDelaysDays AS FLOAT)) > 90
          OR CAST(SUM(CASE WHEN f.IsPayed=0 THEN 1 ELSE 0 END) AS FLOAT)
             / NULLIF(COUNT(f.DocumentNumber),0) > 0.5
        THEN 'High'
        WHEN AVG(CAST(f.PaymentDelaysDays AS FLOAT)) BETWEEN 30 AND 90
          OR CAST(SUM(CASE WHEN f.IsPayed=0 THEN 1 ELSE 0 END) AS FLOAT)
             / NULLIF(COUNT(f.DocumentNumber),0) BETWEEN 0.2 AND 0.5
        THEN 'Medium'
        ELSE 'Low'
    END                                         AS computed_risk
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus]   f
JOIN [dwh_parapharmacie].[dbo].[Dim_Customer]   c
    ON f.ClientID = c.CustomerID
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
GROUP BY f.ClientID, c.Type, c.RiskLevel
"""

# ── GOAL 4 — Margin / Price Regression ───────────────────────────────────────
QUERY_GOAL4_MARGIN = """
SELECT
    f.ProductID,
    p.ProductName,
    p.ProductCategory,
    p.ProductBrand,
    p.PurchasePrice,
    p.SellingPrice,
    f.SupplierID,
    s.SupplierName,
    s.PaymentCondition,
    s.LeadTimeDays,
    dt.Year,
    dt.Month_Name,
    dt.Season,
    SUM(f.SoldItemsQty)                         AS qty_sold,
    SUM(f.TotalSales)                           AS total_revenue,
    SUM(f.CostVariance)                         AS total_cost,
    SUM(f.TotalSales) - SUM(f.CostVariance)     AS gross_margin,
    CASE
        WHEN SUM(f.CostVariance) > 0
        THEN (SUM(f.TotalSales) - SUM(f.CostVariance))
              / SUM(f.CostVariance) * 100
        ELSE NULL
    END                                         AS margin_pct,
    AVG(f.DiscountPct)                          AS avg_discount_pct,
    AVG(f.TaxAmount)                            AS avg_tax,
    AVG(CAST(f.OnHand      AS FLOAT))           AS avg_on_hand,
    AVG(CAST(f.SafetyStock AS FLOAT))           AS avg_safety_stock,
    AVG(f.StockValue)                           AS avg_stock_value,
    f.StockStatus,
    SUM(f.ReturnedItemsQty)                     AS returns_qty,
    COUNT(DISTINCT f.ClientID)                  AS nb_customers
FROM [dwh_parapharmacie].[dbo].[Fact_Revenus]   f
JOIN [dwh_parapharmacie].[dbo].[DIM_Product]    p
    ON f.ProductID  = p.ProductID
JOIN [dwh_parapharmacie].[dbo].[DIM_Supplier]   s
    ON f.SupplierID = s.SupplierID
JOIN [dwh_parapharmacie].[dbo].[Dim_DateTime]   dt
    ON f.DateID     = dt.Date_Time_ID
WHERE f.DocumentType IN ('VenteComptoir','FactureClient','BonLivraison')
  AND f.SupplierID IS NOT NULL
GROUP BY
    f.ProductID, p.ProductName, p.ProductCategory, p.ProductBrand,
    p.PurchasePrice, p.SellingPrice,
    f.SupplierID, s.SupplierName, s.PaymentCondition, s.LeadTimeDays,
    dt.Year, dt.Month_Name, dt.Season, f.StockStatus
"""
