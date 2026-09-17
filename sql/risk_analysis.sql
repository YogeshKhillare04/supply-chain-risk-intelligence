-- ============================================================
-- Risk Analysis Queries (Advanced SQL)
-- DataCo Supply Chain Dataset
-- ============================================================
-- These queries demonstrate more advanced SQL concepts:
-- window functions, CTEs, ranking, and complex aggregations
-- to identify risk patterns and prioritize interventions.
-- ============================================================


-- 1. Rank shipping modes by late delivery rate within each market
-- Demonstrates: Window function (RANK), partitioning
SELECT 
    Market,
    "Shipping Mode",
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate,
    RANK() OVER (
        PARTITION BY Market 
        ORDER BY AVG(Late_delivery_risk) DESC
    ) AS risk_rank
FROM orders
GROUP BY Market, "Shipping Mode"
ORDER BY Market, risk_rank;


-- 2. Running cumulative late delivery rate over time
-- Demonstrates: Window function (running aggregate)
WITH monthly_data AS (
    SELECT 
        SUBSTR("order date (DateOrders)", 1, 7) AS month,
        COUNT(*) AS total_orders,
        SUM(Late_delivery_risk) AS late_orders
    FROM orders
    GROUP BY SUBSTR("order date (DateOrders)", 1, 7)
)
SELECT 
    month,
    total_orders,
    late_orders,
    ROUND(late_orders * 100.0 / total_orders, 2) AS monthly_late_rate,
    SUM(late_orders) OVER (ORDER BY month) AS cumulative_late,
    SUM(total_orders) OVER (ORDER BY month) AS cumulative_total,
    ROUND(
        SUM(late_orders) OVER (ORDER BY month) * 100.0 / 
        SUM(total_orders) OVER (ORDER BY month),
        2
    ) AS cumulative_late_rate
FROM monthly_data
ORDER BY month;


-- 3. Identify high-risk combinations (market + shipping mode + category)
-- Demonstrates: CTE, HAVING, multi-level grouping
WITH risk_combinations AS (
    SELECT 
        Market,
        "Shipping Mode",
        "Category Name",
        COUNT(*) AS order_count,
        ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate,
        ROUND(SUM(Sales), 2) AS total_revenue
    FROM orders
    GROUP BY Market, "Shipping Mode", "Category Name"
    HAVING COUNT(*) >= 100  -- Only meaningful sample sizes
)
SELECT 
    Market,
    "Shipping Mode",
    "Category Name",
    order_count,
    late_rate,
    total_revenue,
    CASE 
        WHEN late_rate >= 70 THEN 'HIGH RISK'
        WHEN late_rate >= 50 THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END AS risk_level
FROM risk_combinations
WHERE late_rate >= 50
ORDER BY late_rate DESC
LIMIT 20;


-- 4. Customer segments with above-average late delivery rates
-- Demonstrates: Subquery in WHERE clause
SELECT 
    "Customer Segment",
    "Order Region",
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate
FROM orders
GROUP BY "Customer Segment", "Order Region"
HAVING AVG(Late_delivery_risk) > (
    SELECT AVG(Late_delivery_risk) FROM orders
)
ORDER BY late_rate DESC
LIMIT 15;


-- 5. Top categories contributing to delayed orders
-- Demonstrates: CTE with percentage contribution
WITH category_delays AS (
    SELECT 
        "Category Name",
        SUM(Late_delivery_risk) AS delayed_orders,
        COUNT(*) AS total_orders
    FROM orders
    GROUP BY "Category Name"
),
total_delays AS (
    SELECT SUM(Late_delivery_risk) AS all_delayed FROM orders
)
SELECT 
    cd."Category Name",
    cd.delayed_orders,
    cd.total_orders,
    ROUND(cd.delayed_orders * 100.0 / cd.total_orders, 1) AS category_late_rate,
    ROUND(cd.delayed_orders * 100.0 / td.all_delayed, 2) AS pct_of_all_delays,
    -- Running total of delay contribution
    ROUND(
        SUM(cd.delayed_orders) OVER (ORDER BY cd.delayed_orders DESC) * 100.0 / td.all_delayed,
        1
    ) AS cumulative_pct_of_delays
FROM category_delays cd, total_delays td
ORDER BY cd.delayed_orders DESC;


-- 6. Shipping mode performance comparison
-- Demonstrates: CASE WHEN for conditional aggregation, self-comparison
SELECT 
    "Shipping Mode",
    COUNT(*) AS total_orders,
    
    -- On-time stats
    SUM(CASE WHEN Late_delivery_risk = 0 THEN 1 ELSE 0 END) AS on_time_count,
    ROUND(AVG(CASE WHEN Late_delivery_risk = 0 THEN Sales END), 2) AS avg_sales_on_time,
    
    -- Late stats
    SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) AS late_count,
    ROUND(AVG(CASE WHEN Late_delivery_risk = 1 THEN Sales END), 2) AS avg_sales_late,
    
    -- Overall
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_pct,
    ROUND(AVG("Days for shipment (scheduled)"), 1) AS avg_scheduled
FROM orders
GROUP BY "Shipping Mode"
ORDER BY late_pct DESC;


-- 7. Risk score bucketing based on multiple factors
-- Demonstrates: Complex CASE WHEN logic
SELECT 
    CASE 
        WHEN "Days for shipment (scheduled)" <= 2 
             AND "Shipping Mode" = 'Standard Class' THEN 'Very High Risk'
        WHEN "Days for shipment (scheduled)" <= 2 THEN 'High Risk'
        WHEN "Shipping Mode" = 'Same Day' THEN 'Medium Risk'
        WHEN "Shipping Mode" = 'First Class' THEN 'Medium Risk'
        ELSE 'Standard Risk'
    END AS rule_based_risk,
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS actual_late_rate,
    ROUND(AVG(Sales), 2) AS avg_sales
FROM orders
GROUP BY rule_based_risk
ORDER BY actual_late_rate DESC;


-- 8. Month-over-month change in late delivery rate
-- Demonstrates: LAG window function
WITH monthly AS (
    SELECT 
        SUBSTR("order date (DateOrders)", 1, 7) AS month,
        ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY SUBSTR("order date (DateOrders)", 1, 7)
)
SELECT 
    month,
    late_rate,
    order_count,
    LAG(late_rate) OVER (ORDER BY month) AS prev_month_rate,
    ROUND(
        late_rate - LAG(late_rate) OVER (ORDER BY month),
        2
    ) AS rate_change,
    CASE 
        WHEN late_rate > LAG(late_rate) OVER (ORDER BY month) THEN '↑ Worsening'
        WHEN late_rate < LAG(late_rate) OVER (ORDER BY month) THEN '↓ Improving'
        ELSE '→ Stable'
    END AS trend
FROM monthly
ORDER BY month;
