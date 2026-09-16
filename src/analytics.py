"""
src/analytics.py
================
Builds the SQLite database and provides analytics/KPI helper functions.
"""

import os
import sqlite3
import pandas as pd
import numpy as np


# ── Database helpers ──────────────────────────────────────────────────────────

def build_database(df: pd.DataFrame, base_dir: str) -> str:
    """Create banking_risk.db and populate all tables."""
    db_path = os.path.join(base_dir, "database", "banking_risk.db")
    conn = sqlite3.connect(db_path)

    # customers table
    customer_cols = [
        "Customer_ID", "Age", "Gender", "City", "State", "Occupation",
        "Account_Type", "Account_Balance", "Monthly_Income", "Credit_Score",
        "Tenure_Years", "Number_of_Products", "Monthly_Expenses",
        "Online_Banking_Usage", "Mobile_Banking_Usage", "ATM_Usage",
        "Branch_Visits", "Credit_Card_Usage", "Debit_Card_Usage",
        "Customer_Satisfaction", "Complaints", "Support_Calls",
        "Last_Transaction_Date", "Days_Since_Transaction",
        "Churn_Status", "Default_Status", "Risk_Label",
    ]
    customers = df[[c for c in customer_cols if c in df.columns]].copy()
    if "Last_Transaction_Date" in customers.columns:
        customers["Last_Transaction_Date"] = customers["Last_Transaction_Date"].astype(str)
    customers.to_sql("customers", conn, if_exists="replace", index=False)

    # transactions table
    txn_cols = ["Customer_ID", "Transaction_Count", "Transaction_Amount",
                "Average_Transaction_Value", "ATM_Usage"]
    txn = df[[c for c in txn_cols if c in df.columns]].copy()
    txn.to_sql("transactions", conn, if_exists="replace", index=False)

    # loans table
    loan_cols = ["Customer_ID", "Loan_Status", "Loan_Amount", "Loan_Tenure",
                 "EMI", "Missed_Payments", "Late_Payments", "Existing_Loans",
                 "Debt_to_Income_Ratio", "Default_Status"]
    loans = df[[c for c in loan_cols if c in df.columns]].copy()
    loans.to_sql("loans", conn, if_exists="replace", index=False)

    # risk_analysis table
    risk_cols = ["Customer_ID", "Credit_Score", "Debt_to_Income_Ratio",
                 "Missed_Payments", "Late_Payments", "Default_Status",
                 "Churn_Status", "Risk_Label", "Complaints"]
    risk = df[[c for c in risk_cols if c in df.columns]].copy()
    risk.to_sql("risk_analysis", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    return db_path


def get_connection(base_dir: str):
    db_path = os.path.join(base_dir, "database", "banking_risk.db")
    return sqlite3.connect(db_path)


# ── KPI calculations ──────────────────────────────────────────────────────────

def calculate_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    active = int((df["Churn_Status"] == 0).sum())
    inactive = total - active

    loan_customers = df[df["Loan_Amount"] > 0]
    default_rate = (
        df["Default_Status"].mean() * 100 if "Default_Status" in df.columns else 0
    )
    churn_rate = (
        df["Churn_Status"].mean() * 100 if "Churn_Status" in df.columns else 0
    )

    risk_counts = df["Risk_Label"].value_counts()

    return {
        "Total Customers": total,
        "Active Customers": active,
        "Inactive Customers": inactive,
        "Average Age": round(df["Age"].mean(), 1),
        "Average Income": round(df["Monthly_Income"].mean(), 0),
        "Average Account Balance": round(df["Account_Balance"].mean(), 0),
        "Total Deposits (₹Cr)": round(df["Account_Balance"].sum() / 1e7, 2),
        "Total Loan Amount (₹Cr)": round(df["Loan_Amount"].sum() / 1e7, 2),
        "Average Credit Score": round(df["Credit_Score"].mean(), 1),
        "Average Transaction Amount": round(df["Transaction_Amount"].mean(), 0),
        "Total Transactions": int(df["Transaction_Count"].sum()),
        "Avg Products per Customer": round(df["Number_of_Products"].mean(), 2),
        "Loan Customers": len(loan_customers),
        "Loan Default Rate (%)": round(default_rate, 2),
        "Customer Churn Rate (%)": round(churn_rate, 2),
        "High Risk Customers": int(risk_counts.get("High", 0)),
        "Medium Risk Customers": int(risk_counts.get("Medium", 0)),
        "Low Risk Customers": int(risk_counts.get("Low", 0)),
        "Complaint Rate (%)": round((df["Complaints"] > 0).mean() * 100, 2),
        "Avg Debt-to-Income Ratio": round(df["Debt_to_Income_Ratio"].mean(), 3),
    }
