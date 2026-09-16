"""
src/data_processing.py
======================
Complete data-cleaning pipeline for the banking customer dataset.
"""

import os
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a clean DataFrame."""
    original_shape = df.shape
    print(f"  [clean] Raw shape: {original_shape}")

    # 1. Drop exact duplicates
    df = df.drop_duplicates(subset="Customer_ID", keep="first")
    print(f"  [clean] After dedup: {df.shape}")

    # 2. Fix data types
    numeric_cols = [
        "Age", "Account_Balance", "Monthly_Income", "Credit_Score",
        "Tenure_Years", "Number_of_Products", "Loan_Amount", "Loan_Tenure",
        "EMI", "Monthly_Expenses", "Transaction_Count", "Transaction_Amount",
        "Average_Transaction_Value", "ATM_Usage", "Branch_Visits",
        "Missed_Payments", "Late_Payments", "Existing_Loans",
        "Customer_Satisfaction", "Complaints", "Support_Calls",
        "Churn_Status", "Default_Status",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_cols = ["Online_Banking_Usage", "Mobile_Banking_Usage",
                   "Credit_Card_Usage", "Debit_Card_Usage"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 3. Date conversion
    if "Last_Transaction_Date" in df.columns:
        df["Last_Transaction_Date"] = pd.to_datetime(
            df["Last_Transaction_Date"], errors="coerce"
        )

    # 4. Impute numeric missing values
    median_cols = ["Age", "Monthly_Income", "Credit_Score",
                   "Account_Balance", "Loan_Amount", "Customer_Satisfaction"]
    for col in median_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 5. Validate ranges
    df = df[df["Age"].between(18, 100)]
    df = df[df["Credit_Score"].between(300, 900)]
    df = df[df["Monthly_Income"] > 0]
    df = df[df["Account_Balance"] >= 0]
    df = df[df["Transaction_Amount"] >= 0]

    # 6. Fill remaining numeric NAs with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 7. Standardise categorical values
    for col in ["Gender", "State", "City", "Occupation", "Account_Type", "Loan_Status", "Risk_Label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 8. Fix Last_Transaction_Date NaT
    if "Last_Transaction_Date" in df.columns:
        fallback = pd.Timestamp("2024-01-01")
        df["Last_Transaction_Date"] = df["Last_Transaction_Date"].fillna(fallback)
        df["Days_Since_Transaction"] = (
            pd.Timestamp("2024-12-31") - df["Last_Transaction_Date"]
        ).dt.days.clip(lower=0)

    # 9. Recalculate DTI for cleanliness
    df["Debt_to_Income_Ratio"] = np.where(
        df["Monthly_Income"] > 0,
        (df["EMI"] / df["Monthly_Income"]).round(3),
        0.0,
    )
    df["Debt_to_Income_Ratio"] = df["Debt_to_Income_Ratio"].clip(0, 2)

    # 10. Ensure Risk_Label exists and is clean
    valid_risks = {"Low", "Medium", "High"}
    df["Risk_Label"] = df["Risk_Label"].apply(
        lambda x: x if x in valid_risks else "Medium"
    )

    print(f"  [clean] Final shape: {df.shape}")
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, base_dir: str) -> str:
    path = os.path.join(base_dir, "data", "processed", "banking_customers_clean.csv")
    df.to_csv(path, index=False)
    return path
