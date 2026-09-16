-- ============================================================
-- transaction_analysis.sql
-- Transaction Analytics
-- ============================================================

-- 1. Total transactions and amount
SELECT
    SUM(Transaction_Count) AS Total_Transactions,
    ROUND(SUM(Transaction_Amount) / 1e7, 2) AS Total_Transaction_Crore,
    ROUND(AVG(Average_Transaction_Value), 2) AS Avg_Txn_Value
FROM transactions;

-- 2. Top 10 customers by transaction amount
SELECT t.Customer_ID,
       t.Transaction_Count,
       t.Transaction_Amount,
       t.Average_Transaction_Value,
       c.Occupation, c.State
FROM transactions t
JOIN customers c ON t.Customer_ID = c.Customer_ID
ORDER BY t.Transaction_Amount DESC
LIMIT 10;

-- 3. Digital banking adoption
SELECT
    SUM(Online_Banking_Usage)                                AS Online_Users,
    SUM(Mobile_Banking_Usage)                               AS Mobile_Users,
    SUM(CASE WHEN Online_Banking_Usage = 1 AND Mobile_Banking_Usage = 1 THEN 1 ELSE 0 END) AS Both_Digital,
    COUNT(*)                                                 AS Total,
    ROUND(100.0 * SUM(Online_Banking_Usage) / COUNT(*), 2)  AS Online_Pct,
    ROUND(100.0 * SUM(Mobile_Banking_Usage) / COUNT(*), 2)  AS Mobile_Pct
FROM customers;

-- 4. ATM usage distribution
SELECT
    CASE
        WHEN ATM_Usage = 0 THEN '0'
        WHEN ATM_Usage <= 3 THEN '1-3'
        WHEN ATM_Usage <= 6 THEN '4-6'
        WHEN ATM_Usage <= 10 THEN '7-10'
        ELSE '10+'
    END AS ATM_Bucket,
    COUNT(*) AS Count
FROM customers
GROUP BY ATM_Bucket
ORDER BY ATM_Bucket;

-- 5. Transactions by occupation
SELECT c.Occupation,
       COUNT(*) AS Customers,
       ROUND(AVG(t.Transaction_Amount), 0) AS Avg_Txn_Amount,
       ROUND(AVG(t.Transaction_Count), 1)  AS Avg_Txn_Count
FROM transactions t
JOIN customers c ON t.Customer_ID = c.Customer_ID
GROUP BY c.Occupation
ORDER BY Avg_Txn_Amount DESC;

-- 6. Transaction amount vs risk
SELECT c.Risk_Label,
       ROUND(AVG(t.Transaction_Amount), 0) AS Avg_Txn_Amount,
       ROUND(AVG(t.Transaction_Count), 1)  AS Avg_Txn_Count
FROM transactions t
JOIN customers c ON t.Customer_ID = c.Customer_ID
GROUP BY c.Risk_Label;
