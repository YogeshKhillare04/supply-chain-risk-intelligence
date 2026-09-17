# SQL Analytics

This folder contains SQL queries that analyze the DataCo Supply Chain dataset from a business and operational perspective.

## Purpose

While the main ML pipeline is implemented in Python (notebooks), these SQL queries demonstrate analytical skills that are essential for a Data Scientist role — especially at data-driven companies where SQL is used daily for ad-hoc analysis, reporting, and data exploration.

## Files

| File | Focus | Key SQL Concepts |
|------|-------|-----------------|
| `business_analysis.sql` | Revenue, segments, markets | GROUP BY, CASE WHEN, subqueries, CTEs, LAG |
| `delivery_analysis.sql` | Late delivery patterns, shipping performance | Aggregations, CASE WHEN, date functions |
| `risk_analysis.sql` | Advanced risk patterns, prioritization | Window functions (RANK, LAG, running totals), CTEs, HAVING |

## How to Run

These queries are designed for **SQLite** but can be adapted for PostgreSQL or MySQL.

### Option 1: SQLite (simplest)

```bash
# Load the CSV into SQLite
sqlite3 supply_chain.db
.mode csv
.headers on
.import data/raw/DataCoSupplyChainDataset.csv orders

# Run a query file
.read sql/business_analysis.sql
```

### Option 2: Python + SQLite

```python
import sqlite3
import pandas as pd

# Load data
df = pd.read_csv("data/raw/DataCoSupplyChainDataset.csv", encoding="latin1")

# Create SQLite database
conn = sqlite3.connect(":memory:")
df.to_sql("orders", conn, index=False)

# Run a query
result = pd.read_sql("SELECT * FROM orders LIMIT 5", conn)
print(result)
```

## Note

These queries are analytical tools, not part of the ML pipeline. The model training and evaluation are done entirely in Python. The SQL queries answer the same kinds of business questions that a data scientist would investigate using SQL in a production environment.
