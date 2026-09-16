-- ============================================================
-- customer_kpi_analysis.sql
-- Banking Customer KPI Analysis
-- ============================================================

-- 1. Total customers
SELECT COUNT(*) AS Total_Customers FROM customers;

-- 2. Average customer balance
SELECT ROUND(AVG(Account_Balance), 2) AS Avg_Account_Balance FROM customers;

-- 3. Average income
SELECT ROUND(AVG(Monthly_Income), 2) AS Avg_Monthly_Income FROM customers;

-- 4. Total deposits
SELECT ROUND(SUM(Account_Balance) / 1e7, 2) AS Total_Deposits_Crore FROM customers;

-- 5. Loan default rate
SELECT
    ROUND(AVG(Default_Status) * 100, 2) AS Default_Rate_Pct
FROM customers;

-- 6. Customer churn rate
SELECT
    ROUND(AVG(Churn_Status) * 100, 2) AS Churn_Rate_Pct
FROM customers;

-- 7. Average products per customer
SELECT ROUND(AVG(Number_of_Products), 2) AS Avg_Products FROM customers;

-- 8. Average credit score
SELECT ROUND(AVG(Credit_Score), 1) AS Avg_Credit_Score FROM customers;

-- 9. Complaint rate
SELECT ROUND(100.0 * SUM(CASE WHEN Complaints > 0 THEN 1 ELSE 0 END) / COUNT(*), 2)
    AS Complaint_Rate_Pct
FROM customers;

-- 10. Risk distribution
SELECT Risk_Label,
       COUNT(*) AS Count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2) AS Pct
FROM customers
GROUP BY Risk_Label
ORDER BY CASE Risk_Label WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;

-- 11. Active vs Inactive customers
SELECT
    SUM(CASE WHEN Churn_Status = 0 THEN 1 ELSE 0 END) AS Active_Customers,
    SUM(CASE WHEN Churn_Status = 1 THEN 1 ELSE 0 END) AS Inactive_Customers
FROM customers;

-- 12. Customers by account type
SELECT Account_Type,
       COUNT(*) AS Total,
       ROUND(AVG(Account_Balance), 2) AS Avg_Balance
FROM customers
GROUP BY Account_Type
ORDER BY Total DESC;

-- 13. Customers by age group
SELECT
    CASE
        WHEN Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN Age BETWEEN 36 AND 45 THEN '36-45'
        WHEN Age BETWEEN 46 AND 55 THEN '46-55'
        ELSE '56+'
    END AS Age_Group,
    COUNT(*) AS Count,
    ROUND(AVG(Monthly_Income), 2) AS Avg_Income
FROM customers
GROUP BY Age_Group
ORDER BY Age_Group;

-- 14. Top 10 customers by account balance
SELECT Customer_ID, Age, Occupation, Account_Balance, Credit_Score, Risk_Label
FROM customers
ORDER BY Account_Balance DESC
LIMIT 10;

-- 15. Average tenure by account type
SELECT Account_Type, ROUND(AVG(Tenure_Years), 1) AS Avg_Tenure_Years
FROM customers
GROUP BY Account_Type;
