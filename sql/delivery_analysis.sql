-- ============================================================
-- Delivery Performance Analysis Queries
-- DataCo Supply Chain Dataset
-- ============================================================
-- These queries focus on understanding delivery performance,
-- late delivery patterns, and operational metrics.
-- ============================================================


-- 1. Late delivery rate by shipping mode
SELECT 
    "Shipping Mode",
    COUNT(*) AS total_orders,
    SUM(Late_delivery_risk) AS late_orders,
    COUNT(*) - SUM(Late_delivery_risk) AS on_time_orders,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_delivery_rate,
    ROUND(AVG("Days for shipment (scheduled)"), 1) AS avg_scheduled_days,
    ROUND(AVG("Days for shipping (real)"), 1) AS avg_actual_days
FROM orders
GROUP BY "Shipping Mode"
ORDER BY late_delivery_rate DESC;


-- 2. Late delivery rate by region
SELECT 
    "Order Region",
    COUNT(*) AS total_orders,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate,
    ROUND(AVG("Days for shipping (real)"), 1) AS avg_actual_days,
    ROUND(AVG(Sales), 2) AS avg_order_value
FROM orders
GROUP BY "Order Region"
ORDER BY late_rate DESC;


-- 3. Delivery performance: scheduled vs actual days
-- Identifies where deliveries consistently take longer than planned
SELECT 
    "Days for shipment (scheduled)" AS scheduled_days,
    ROUND(AVG("Days for shipping (real)"), 2) AS avg_actual_days,
    ROUND(AVG("Days for shipping (real)") - "Days for shipment (scheduled)", 2) AS avg_delay,
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 1) AS late_rate
FROM orders
GROUP BY "Days for shipment (scheduled)"
ORDER BY scheduled_days;


-- 4. Monthly late delivery trend
SELECT 
    SUBSTR("order date (DateOrders)", 1, 7) AS month,
    COUNT(*) AS total_orders,
    SUM(Late_delivery_risk) AS late_orders,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate
FROM orders
GROUP BY SUBSTR("order date (DateOrders)", 1, 7)
ORDER BY month;


-- 5. Delivery status breakdown
SELECT 
    "Delivery Status",
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 1) AS percentage,
    ROUND(AVG(Sales), 2) AS avg_order_value
FROM orders
GROUP BY "Delivery Status"
ORDER BY order_count DESC;


-- 6. Late delivery rate by order quantity (bucketed)
SELECT 
    CASE 
        WHEN "Order Item Quantity" = 1 THEN '1 item'
        WHEN "Order Item Quantity" BETWEEN 2 AND 3 THEN '2-3 items'
        WHEN "Order Item Quantity" BETWEEN 4 AND 5 THEN '4-5 items'
        ELSE '5+ items'
    END AS quantity_bucket,
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate,
    ROUND(AVG(Sales), 2) AS avg_sales
FROM orders
GROUP BY quantity_bucket
ORDER BY late_rate DESC;


-- 7. Orders with both high value and late delivery
-- Using a CTE to identify the threshold
WITH order_stats AS (
    SELECT 
        AVG(Sales) AS avg_sales,
        AVG(Sales) + 1.5 * AVG(Sales) AS high_value_threshold
    FROM orders
)
SELECT 
    "Shipping Mode",
    COUNT(*) AS high_value_late_orders,
    ROUND(AVG(Sales), 2) AS avg_sales,
    ROUND(AVG("Benefit per order"), 2) AS avg_profit_impact
FROM orders, order_stats
WHERE Sales > order_stats.high_value_threshold
  AND Late_delivery_risk = 1
GROUP BY "Shipping Mode"
ORDER BY high_value_late_orders DESC;


-- 8. Day-of-week delivery performance
-- Note: This requires date parsing capability
SELECT 
    CASE CAST(strftime('%w', "order date (DateOrders)") AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_of_week,
    COUNT(*) AS order_count,
    ROUND(AVG(Late_delivery_risk) * 100, 2) AS late_rate
FROM orders
GROUP BY day_of_week
ORDER BY 
    CAST(strftime('%w', "order date (DateOrders)") AS INTEGER);
