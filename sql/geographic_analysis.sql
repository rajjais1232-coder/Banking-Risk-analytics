-- ============================================================
-- geographic_analysis.sql
-- Geographic Distribution Analysis
-- ============================================================

-- 1. Customers by state
SELECT c.State,
       COUNT(*) AS Total_Customers,
       ROUND(AVG(c.Monthly_Income), 0) AS Avg_Income,
       ROUND(AVG(c.Account_Balance), 0) AS Avg_Balance,
       ROUND(AVG(c.Credit_Score), 1) AS Avg_Credit_Score,
       ROUND(100.0 * SUM(c.Churn_Status) / COUNT(*), 2) AS Churn_Rate_Pct,
       ROUND(100.0 * SUM(c.Default_Status) / COUNT(*), 2) AS Default_Rate_Pct
FROM customers c
GROUP BY c.State
ORDER BY Total_Customers DESC;

-- 2. Risk by state (ranked)
SELECT c.State,
       COUNT(*) AS Total,
       SUM(CASE WHEN c.Risk_Label = 'High' THEN 1 ELSE 0 END) AS High_Risk,
       SUM(CASE WHEN c.Risk_Label = 'Medium' THEN 1 ELSE 0 END) AS Medium_Risk,
       SUM(CASE WHEN c.Risk_Label = 'Low' THEN 1 ELSE 0 END) AS Low_Risk,
       ROUND(100.0 * SUM(CASE WHEN c.Risk_Label = 'High' THEN 1 ELSE 0 END) / COUNT(*), 2) AS High_Risk_Pct
FROM customers c
GROUP BY c.State
ORDER BY High_Risk_Pct DESC;

-- 3. Top cities by average balance
SELECT c.City, c.State,
       COUNT(*) AS Customers,
       ROUND(AVG(c.Account_Balance), 0) AS Avg_Balance,
       ROUND(AVG(c.Monthly_Income), 0) AS Avg_Income
FROM customers c
GROUP BY c.City, c.State
HAVING Customers >= 50
ORDER BY Avg_Balance DESC
LIMIT 15;

-- 4. Loan performance by state
SELECT c.State,
       COUNT(l.Customer_ID) AS Loan_Customers,
       ROUND(SUM(l.Loan_Amount) / 1e7, 2) AS Total_Loans_Crore,
       ROUND(100.0 * SUM(l.Default_Status) / COUNT(l.Customer_ID), 2) AS Default_Rate_Pct
FROM loans l
JOIN customers c ON l.Customer_ID = c.Customer_ID
WHERE l.Loan_Amount > 0
GROUP BY c.State
ORDER BY Default_Rate_Pct DESC;

-- 5. Window function: Rank states by average income
SELECT State, Avg_Income,
       RANK() OVER (ORDER BY Avg_Income DESC) AS Income_Rank
FROM (
    SELECT State, ROUND(AVG(Monthly_Income), 0) AS Avg_Income
    FROM customers
    GROUP BY State
) sub;
