-- ============================================================
-- Business Analysis Queries
-- DataCo Supply Chain Dataset
-- ============================================================
-- These queries analyze business performance metrics across
-- different dimensions. They complement the Python-based
-- analysis by demonstrating SQL proficiency.
--
-- To run: Load DataCoSupplyChainDataset.csv into SQLite
--   sqlite3 supply_chain.db
--   .mode csv
--   .import DataCoSupplyChainDataset.csv orders
-- ============================================================


-- 1. Overall business summary
SELECT 
    COUNT(*) AS total_orders,
    COUNT(DISTINCT "Customer Id") AS unique_customers,
    ROUND(AVG(Sales), 2) AS avg_order_value,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG("Benefit per order"), 2) AS avg_profit_per_order,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_delivery_pct
FROM orders;


-- 2. Revenue and order volume by market
SELECT 
    Market,
    COUNT(*) AS order_count,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG(Sales), 2) AS avg_order_value,
    ROUND(AVG("Benefit per order"), 2) AS avg_profit,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct
FROM orders
GROUP BY Market
ORDER BY total_revenue DESC;


-- 3. Top 10 product categories by revenue
SELECT 
    "Category Name",
    COUNT(*) AS order_count,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG(Sales), 2) AS avg_order_value,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct
FROM orders
GROUP BY "Category Name"
ORDER BY total_revenue DESC
LIMIT 10;


-- 4. Customer segment analysis
SELECT 
    "Customer Segment",
    COUNT(*) AS order_count,
    ROUND(AVG(Sales), 2) AS avg_order_value,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct,
    ROUND(AVG("Order Item Quantity"), 1) AS avg_quantity
FROM orders
GROUP BY "Customer Segment"
ORDER BY total_revenue DESC;


-- 5. Revenue by shipping mode with profitability
SELECT 
    "Shipping Mode",
    COUNT(*) AS order_count,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG("Benefit per order"), 2) AS avg_profit,
    ROUND(
        SUM(CASE WHEN "Benefit per order" > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        1
    ) AS profitable_order_pct
FROM orders
GROUP BY "Shipping Mode"
ORDER BY total_revenue DESC;


-- 6. Top 15 countries by order volume
SELECT 
    "Order Country",
    COUNT(*) AS order_count,
    ROUND(SUM(Sales), 2) AS total_revenue,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct
FROM orders
GROUP BY "Order Country"
ORDER BY order_count DESC
LIMIT 15;


-- 7. High-value orders analysis (above median sales)
-- Using a subquery to calculate the median
SELECT 
    CASE 
        WHEN Sales > (SELECT AVG(Sales) FROM orders) THEN 'Above Average'
        ELSE 'Below Average'
    END AS order_value_tier,
    COUNT(*) AS order_count,
    ROUND(AVG(Sales), 2) AS avg_sales,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct,
    ROUND(AVG("Benefit per order"), 2) AS avg_profit
FROM orders
GROUP BY order_value_tier;


-- 8. Monthly revenue trend (using CTE)
WITH monthly_revenue AS (
    SELECT 
        SUBSTR("order date (DateOrders)", 1, 7) AS order_month,
        COUNT(*) AS order_count,
        ROUND(SUM(Sales), 2) AS revenue,
        ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct
    FROM orders
    GROUP BY SUBSTR("order date (DateOrders)", 1, 7)
)
SELECT 
    order_month,
    order_count,
    revenue,
    late_pct,
    ROUND(revenue - LAG(revenue) OVER (ORDER BY order_month), 2) AS revenue_change
FROM monthly_revenue
ORDER BY order_month;
