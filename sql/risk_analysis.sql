-- ============================================================
-- risk_analysis.sql
-- Customer Risk Analysis
-- ============================================================

-- 1. Risk distribution
SELECT Risk_Label,
       COUNT(*) AS Count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM risk_analysis), 2) AS Pct
FROM risk_analysis
GROUP BY Risk_Label;

-- 2. High-risk customers
SELECT r.Customer_ID, c.Age, c.Monthly_Income, c.Credit_Score,
       r.Missed_Payments, r.Debt_to_Income_Ratio, r.Default_Status, r.Risk_Label
FROM risk_analysis r
JOIN customers c ON r.Customer_ID = c.Customer_ID
WHERE r.Risk_Label = 'High'
ORDER BY r.Missed_Payments DESC
LIMIT 20;

-- 3. Risk by credit score category
SELECT
    CASE
        WHEN r.Credit_Score < 500 THEN 'Very Poor (<500)'
        WHEN r.Credit_Score < 600 THEN 'Poor (500-599)'
        WHEN r.Credit_Score < 700 THEN 'Fair (600-699)'
        WHEN r.Credit_Score < 750 THEN 'Good (700-749)'
        ELSE 'Excellent (750+)'
    END AS Credit_Category,
    COUNT(*) AS Total,
    SUM(CASE WHEN r.Risk_Label = 'High' THEN 1 ELSE 0 END) AS High_Risk,
    ROUND(100.0 * SUM(CASE WHEN r.Risk_Label = 'High' THEN 1 ELSE 0 END) / COUNT(*), 2) AS High_Risk_Pct
FROM risk_analysis r
GROUP BY Credit_Category
ORDER BY High_Risk_Pct DESC;

-- 4. Risk by missed payments
SELECT
    CASE
        WHEN Missed_Payments = 0 THEN '0 Missed'
        WHEN Missed_Payments <= 2 THEN '1-2 Missed'
        WHEN Missed_Payments <= 5 THEN '3-5 Missed'
        ELSE '6+ Missed'
    END AS Missed_Bucket,
    COUNT(*) AS Total,
    SUM(CASE WHEN Risk_Label = 'High' THEN 1 ELSE 0 END) AS High_Risk,
    ROUND(100.0 * SUM(CASE WHEN Risk_Label = 'High' THEN 1 ELSE 0 END) / COUNT(*), 2) AS High_Risk_Pct
FROM risk_analysis
GROUP BY Missed_Bucket
ORDER BY High_Risk_Pct DESC;

-- 5. Customers with high DTI
SELECT r.Customer_ID, c.Monthly_Income, r.Debt_to_Income_Ratio,
       r.Missed_Payments, r.Risk_Label
FROM risk_analysis r
JOIN customers c ON r.Customer_ID = c.Customer_ID
WHERE r.Debt_to_Income_Ratio > 0.5
ORDER BY r.Debt_to_Income_Ratio DESC
LIMIT 20;

-- 6. Risk by state
SELECT c.State,
       COUNT(*) AS Total,
       SUM(CASE WHEN r.Risk_Label = 'High' THEN 1 ELSE 0 END) AS High_Risk,
       ROUND(100.0 * SUM(CASE WHEN r.Risk_Label = 'High' THEN 1 ELSE 0 END) / COUNT(*), 2) AS High_Risk_Pct
FROM risk_analysis r
JOIN customers c ON r.Customer_ID = c.Customer_ID
GROUP BY c.State
ORDER BY High_Risk_Pct DESC;

-- 7. Risk + Default correlation
SELECT r.Risk_Label,
       COUNT(*) AS Total,
       SUM(r.Default_Status) AS Defaults,
       ROUND(100.0 * SUM(r.Default_Status) / COUNT(*), 2) AS Default_Rate_Pct
FROM risk_analysis r
GROUP BY r.Risk_Label;
