"""
sql_eda.py
----------
Loads the churn CSV into an in-memory SQLite database and runs a few
exploratory SQL queries. This demonstrates SQL fluency on the same
data used for modeling (a common interview ask: "show me the churn
rate by contract type using SQL").
"""

import sqlite3
import pandas as pd

df = pd.read_csv("data/churn.csv")

conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")


def run(query: str, label: str):
    print(f"\n--- {label} ---")
    result = pd.read_sql_query(query, conn)
    print(result.to_string(index=False))


run(
    """
    SELECT Contract,
           COUNT(*) AS total_customers,
           SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
           ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY Contract
    ORDER BY churn_rate_pct DESC;
    """,
    "Churn rate by contract type",
)

run(
    """
    SELECT InternetService,
           ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
           ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY InternetService
    ORDER BY churn_rate_pct DESC;
    """,
    "Churn rate & avg charges by internet service",
)

run(
    """
    SELECT
        CASE
            WHEN tenure < 12 THEN '0-12 months'
            WHEN tenure < 36 THEN '12-36 months'
            ELSE '36+ months'
        END AS tenure_bucket,
        COUNT(*) AS total_customers,
        ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY tenure_bucket
    ORDER BY churn_rate_pct DESC;
    """,
    "Churn rate by tenure bucket",
)

conn.close()
