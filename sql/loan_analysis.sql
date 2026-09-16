-- ============================================================
-- loan_analysis.sql
-- Loan Portfolio Analysis
-- ============================================================

-- 1. Total loans
SELECT
    COUNT(*) AS Total_Loan_Customers,
    ROUND(SUM(Loan_Amount) / 1e7, 2) AS Total_Loan_Crore,
    ROUND(AVG(Loan_Amount), 0) AS Avg_Loan_Amount,
    ROUND(AVG(Debt_to_Income_Ratio), 3) AS Avg_DTI
FROM loans
WHERE Loan_Amount > 0;

-- 2. Loan status breakdown
SELECT Loan_Status,
       COUNT(*) AS Count,
       ROUND(SUM(Loan_Amount) / 1e7, 2) AS Total_Crore,
       ROUND(AVG(Loan_Amount), 0) AS Avg_Amount
FROM loans
GROUP BY Loan_Status
ORDER BY Count DESC;

-- 3. Loan default analysis
SELECT
    Default_Status,
    COUNT(*) AS Count,
    ROUND(AVG(Loan_Amount), 0) AS Avg_Loan,
    ROUND(AVG(Missed_Payments), 2) AS Avg_Missed_Payments,
    ROUND(AVG(Debt_to_Income_Ratio), 3) AS Avg_DTI
FROM loans
GROUP BY Default_Status;

-- 4. Loan amount by state
SELECT c.State,
       COUNT(*) AS Loan_Customers,
       ROUND(SUM(l.Loan_Amount) / 1e7, 2) AS Total_Loans_Crore,
       ROUND(AVG(l.Loan_Amount), 0) AS Avg_Loan,
       ROUND(100.0 * SUM(l.Default_Status) / COUNT(*), 2) AS Default_Rate_Pct
FROM loans l
JOIN customers c ON l.Customer_ID = c.Customer_ID
WHERE l.Loan_Amount > 0
GROUP BY c.State
ORDER BY Total_Loans_Crore DESC;

-- 5. Missed payments distribution
SELECT
    CASE
        WHEN Missed_Payments = 0 THEN '0'
        WHEN Missed_Payments BETWEEN 1 AND 2 THEN '1-2'
        WHEN Missed_Payments BETWEEN 3 AND 5 THEN '3-5'
        ELSE '6+'
    END AS Missed_Bucket,
    COUNT(*) AS Count,
    ROUND(100.0 * SUM(Default_Status) / COUNT(*), 2) AS Default_Rate
FROM loans
GROUP BY Missed_Bucket
ORDER BY Default_Rate DESC;

-- 6. High DTI customers at risk
SELECT l.Customer_ID, c.Monthly_Income, l.Loan_Amount, l.EMI,
       l.Debt_to_Income_Ratio, l.Missed_Payments, c.Risk_Label
FROM loans l
JOIN customers c ON l.Customer_ID = c.Customer_ID
WHERE l.Debt_to_Income_Ratio > 0.5 AND l.Loan_Amount > 0
ORDER BY l.Debt_to_Income_Ratio DESC
LIMIT 20;

-- 7. Loan tenure analysis
SELECT Loan_Tenure,
       COUNT(*) AS Count,
       ROUND(AVG(Loan_Amount), 0) AS Avg_Amount,
       ROUND(AVG(EMI), 0) AS Avg_EMI
FROM loans
WHERE Loan_Tenure > 0
GROUP BY Loan_Tenure
ORDER BY Loan_Tenure;
