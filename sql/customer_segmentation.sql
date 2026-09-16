-- ============================================================
-- customer_segmentation.sql
-- Customer Segmentation Analysis
-- ============================================================

-- Segment customers using business rules
SELECT
    Customer_ID,
    Account_Balance,
    Monthly_Income,
    Credit_Score,
    Number_of_Products,
    Transaction_Amount,
    Online_Banking_Usage,
    Mobile_Banking_Usage,
    Churn_Status,
    Risk_Label,
    CASE
        WHEN Account_Balance >= 500000 AND Credit_Score >= 750 AND Number_of_Products >= 3
             THEN 'Premium'
        WHEN Account_Balance >= 200000 AND Credit_Score >= 650
             THEN 'High Value'
        WHEN Transaction_Amount >= 100000 AND Online_Banking_Usage = 1
             THEN 'Digital First'
        WHEN Loan_Amount > 0 AND Debt_to_Income_Ratio > 0.35
             THEN 'Loan Dependent'
        WHEN Churn_Status = 1 OR Days_Since_Transaction > 90
             THEN 'At Risk'
        WHEN Transaction_Amount < 20000 AND Branch_Visits <= 1
             THEN 'Low Engagement'
        ELSE 'Regular'
    END AS Customer_Segment
FROM customers
LEFT JOIN loans USING (Customer_ID);

-- Segment summary
SELECT
    Segment,
    COUNT(*) AS Count,
    ROUND(AVG(Monthly_Income), 0) AS Avg_Income,
    ROUND(AVG(Account_Balance), 0) AS Avg_Balance,
    ROUND(AVG(Credit_Score), 1) AS Avg_Credit_Score,
    ROUND(AVG(CAST(Churn_Status AS FLOAT)) * 100, 2) AS Churn_Rate_Pct
FROM (
    SELECT
        c.Customer_ID, c.Monthly_Income, c.Account_Balance,
        c.Credit_Score, c.Churn_Status, c.Days_Since_Transaction,
        l.Loan_Amount, l.Debt_to_Income_Ratio,
        c.Transaction_Count,
        CASE
            WHEN c.Account_Balance >= 500000 AND c.Credit_Score >= 750 AND c.Number_of_Products >= 3
                 THEN 'Premium'
            WHEN c.Account_Balance >= 200000 AND c.Credit_Score >= 650
                 THEN 'High Value'
            WHEN c.Online_Banking_Usage = 1 AND c.Mobile_Banking_Usage = 1
                 THEN 'Digital First'
            WHEN l.Loan_Amount > 0 AND l.Debt_to_Income_Ratio > 0.35
                 THEN 'Loan Dependent'
            WHEN c.Churn_Status = 1 OR c.Days_Since_Transaction > 90
                 THEN 'At Risk'
            WHEN c.Transaction_Count < 5
                 THEN 'Low Engagement'
            ELSE 'Regular'
        END AS Segment
    FROM customers c
    LEFT JOIN loans l ON c.Customer_ID = l.Customer_ID
) sub
GROUP BY Segment
ORDER BY Count DESC;
