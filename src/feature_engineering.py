"""
src/feature_engineering.py
===========================
Prepares features for ML model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "Age", "Monthly_Income", "Credit_Score", "Account_Balance",
    "Tenure_Years", "Number_of_Products", "Loan_Amount", "EMI",
    "Monthly_Expenses", "Transaction_Count", "Transaction_Amount",
    "Average_Transaction_Value", "Missed_Payments", "Late_Payments",
    "Existing_Loans", "Debt_to_Income_Ratio", "Complaints",
    "Support_Calls", "ATM_Usage", "Branch_Visits",
    "Online_Banking_Usage", "Mobile_Banking_Usage",
    "Credit_Card_Usage", "Debit_Card_Usage",
    "Customer_Satisfaction", "Days_Since_Transaction",
    # Encoded categoricals
    "Gender_enc", "Occupation_enc", "Account_Type_enc",
]


def engineer_features(df: pd.DataFrame):
    """Return (X, y, feature_names)."""
    df = df.copy()

    # Encode categoricals
    for col in ["Gender", "Occupation", "Account_Type"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    # Days since last transaction
    if "Days_Since_Transaction" not in df.columns:
        df["Days_Since_Transaction"] = 0

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y_raw = df["Risk_Label"].values

    # Encode target
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    y = np.array([mapping.get(v, 1) for v in y_raw])

    return X, y, available
