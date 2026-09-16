"""
generate_data.py
================
Generates a realistic synthetic banking customer dataset (10,000 records),
saves to data/raw/banking_customers.csv, runs cleaning, builds the SQLite
database, and trains the ML models.

NOTE: This project uses entirely SYNTHETIC data for demonstration purposes.
      No real customer data is used.

Usage:
    python generate_data.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
N = 10_000

STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh",
    "Gujarat", "Rajasthan", "West Bengal", "Andhra Pradesh", "Telangana",
    "Kerala", "Punjab", "Haryana", "Bihar", "Madhya Pradesh",
]
STATE_WEIGHTS = [0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06,
                 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]

CITIES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Delhi": ["New Delhi", "Dwarka", "Rohini", "Noida"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore"],
    "Telangana": ["Hyderabad", "Warangal", "Karimnagar", "Khammam"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
    "Haryana": ["Gurugram", "Faridabad", "Hisar", "Rohtak"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior"],
}

OCCUPATIONS = ["Salaried", "Self-Employed", "Business Owner", "Professional",
               "Retired", "Freelancer", "Homemaker", "Student"]
OCC_WEIGHTS = [0.40, 0.20, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02]

ACCOUNT_TYPES = ["Savings", "Current", "Salary", "NRI"]
ACC_WEIGHTS = [0.55, 0.20, 0.20, 0.05]

OCC_INCOME = {
    "Salaried": 55_000, "Self-Employed": 65_000,
    "Business Owner": 120_000, "Professional": 90_000,
    "Retired": 30_000, "Freelancer": 45_000,
    "Homemaker": 20_000, "Student": 12_000,
}


def generate_dataset(n=N):
    print(f"  Generating {n} synthetic banking records ...")
    rng = np.random.default_rng(SEED)

    ages = np.clip(rng.normal(38, 12, n).astype(int), 18, 75)
    genders = rng.choice(["Male", "Female", "Other"], n, p=[0.54, 0.44, 0.02])
    states = rng.choice(STATES, n, p=STATE_WEIGHTS)
    cities = [random.choice(CITIES[s]) for s in states]
    occupations = rng.choice(OCCUPATIONS, n, p=OCC_WEIGHTS)
    account_types = rng.choice(ACCOUNT_TYPES, n, p=ACC_WEIGHTS)

    base_inc = np.array([OCC_INCOME[o] for o in occupations])
    monthly_income = np.clip(
        (base_inc * rng.lognormal(0, 0.4, n)).astype(int), 8_000, 1_000_000
    )

    income_norm = (monthly_income - monthly_income.min()) / ((monthly_income.max() - monthly_income.min()) + 1)
    age_norm = (ages - 18) / 57.0
    credit_scores = np.clip(
        (500 + 250 * income_norm + 50 * age_norm + rng.normal(0, 60, n)).astype(int),
        300, 900
    )
    account_balance = np.clip(
        (monthly_income * rng.uniform(0.5, 8, n)).astype(int), 1_000, 5_000_000
    )
    tenure_years = np.clip(rng.exponential(4, n).astype(int), 0, 30)
    num_products = rng.choice([1, 2, 3, 4, 5], n, p=[0.25, 0.30, 0.25, 0.15, 0.05])

    has_loan = rng.choice([0, 1], n, p=[0.45, 0.55])
    loan_amount = np.where(
        has_loan == 1,
        np.clip((monthly_income * rng.uniform(3, 24, n)).astype(int), 10_000, 5_000_000),
        0,
    )
    loan_tenure_vals = [12, 24, 36, 48, 60, 84, 120, 180, 240]
    loan_tenure = np.where(has_loan == 1, rng.choice(loan_tenure_vals, n), 0)
    emi = np.where(has_loan == 1, (loan_amount / np.maximum(loan_tenure, 1)).astype(int), 0)
    loan_status = np.where(
        has_loan == 1,
        rng.choice(["Active", "Closed", "NPA"], n, p=[0.65, 0.25, 0.10]),
        "None",
    )
    existing_loans = np.where(
        has_loan == 1, rng.choice([1, 2, 3, 4], n, p=[0.55, 0.28, 0.12, 0.05]), 0
    )
    monthly_expenses = np.clip(
        (monthly_income * rng.uniform(0.3, 0.85, n)).astype(int), 5_000, 500_000
    )
    transaction_count = np.clip(rng.poisson(18, n), 1, 120)
    transaction_amount = np.clip(
        (monthly_income * rng.uniform(0.2, 1.5, n)).astype(int), 500, 2_000_000
    )
    avg_txn = (transaction_amount / transaction_count).astype(int)

    online = rng.choice([0, 1], n, p=[0.30, 0.70])
    mobile = rng.choice([0, 1], n, p=[0.25, 0.75])
    atm = rng.integers(0, 20, n)
    branch = rng.integers(0, 10, n)
    cc = rng.choice([0, 1], n, p=[0.45, 0.55])
    dc = rng.choice([0, 1], n, p=[0.10, 0.90])

    mp_high = rng.choice(range(12), n, p=[0.10,0.15,0.15,0.15,0.12,0.10,0.08,0.06,0.04,0.02,0.02,0.01])
    mp_low  = rng.choice(range(12), n, p=[0.551,0.200,0.100,0.060,0.040,0.020,0.010,0.010,0.005,0.002,0.001,0.001])
    missed_payments = np.where(credit_scores < 600, mp_high, mp_low).astype(int)
    late_payments = np.clip((missed_payments * rng.uniform(0.5, 1.5, n)).astype(int), 0, 15)

    dti = np.where(
        monthly_income > 0,
        np.clip(emi / np.maximum(monthly_income, 1), 0.0, 1.5),
        0.0,
    ).round(3)

    satisfaction = np.clip(rng.normal(3.5, 1.0, n), 1, 5).round(1)
    complaints = np.where(
        satisfaction < 3, rng.integers(0, 8, n), rng.choice([0,1,2], n, p=[0.70,0.20,0.10])
    ).astype(int)
    support_calls = np.clip(complaints + rng.integers(0, 4, n), 0, 15)

    base_date = datetime(2024, 12, 31)
    days_since = rng.exponential(30, n).astype(int)
    last_txn = [(base_date - timedelta(days=int(d))).strftime("%Y-%m-%d") for d in days_since]

    churn_prob = (
        0.05
        + 0.15 * (satisfaction < 2.5)
        + 0.10 * (days_since > 60)
        + 0.10 * (complaints > 2)
        + 0.10 * (num_products == 1)
        + 0.05 * (tenure_years < 2)
    )
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    default_prob = (
        0.02
        + 0.25 * (credit_scores < 550)
        + 0.15 * (dti > 0.5)
        + 0.12 * (missed_payments > 3)
        + 0.08 * (loan_status == "NPA")
        + 0.05 * (monthly_income < 20_000)
    )
    default = (rng.uniform(0, 1, n) < np.clip(default_prob, 0, 1)).astype(int)

    risk_score = (
        (credit_scores < 600).astype(float) * 2.5
        + (dti > 0.4).astype(float) * 2.0
        + (missed_payments > 2).astype(float) * 1.8
        + (default == 1).astype(float) * 3.0
        + (loan_amount > monthly_income * 18).astype(float) * 1.5
        + (monthly_income < 25_000).astype(float) * 1.0
        + (late_payments > 2).astype(float) * 1.2
        + (existing_loans > 2).astype(float) * 0.8
        + (complaints > 2).astype(float) * 0.5
        + rng.normal(0, 0.5, n)
    )
    risk_label = np.where(risk_score >= 5, "High", np.where(risk_score >= 2, "Medium", "Low"))

    df = pd.DataFrame({
        "Customer_ID": [f"CUST{str(i+1).zfill(6)}" for i in range(n)],
        "Age": ages, "Gender": genders, "City": cities, "State": states,
        "Occupation": occupations, "Account_Type": account_types,
        "Account_Balance": account_balance, "Monthly_Income": monthly_income,
        "Credit_Score": credit_scores, "Tenure_Years": tenure_years,
        "Number_of_Products": num_products, "Loan_Status": loan_status,
        "Loan_Amount": loan_amount, "Loan_Tenure": loan_tenure, "EMI": emi,
        "Monthly_Expenses": monthly_expenses,
        "Transaction_Count": transaction_count,
        "Transaction_Amount": transaction_amount,
        "Average_Transaction_Value": avg_txn,
        "Online_Banking_Usage": online, "Mobile_Banking_Usage": mobile,
        "ATM_Usage": atm, "Branch_Visits": branch,
        "Credit_Card_Usage": cc, "Debit_Card_Usage": dc,
        "Missed_Payments": missed_payments, "Late_Payments": late_payments,
        "Existing_Loans": existing_loans, "Debt_to_Income_Ratio": dti,
        "Customer_Satisfaction": satisfaction, "Complaints": complaints,
        "Support_Calls": support_calls, "Last_Transaction_Date": last_txn,
        "Churn_Status": churn, "Default_Status": default,
        "Risk_Label": risk_label,
    })

    # Inject realistic missing values
    for col, frac in [("Age", 0.005), ("Monthly_Income", 0.008),
                      ("Credit_Score", 0.006), ("Customer_Satisfaction", 0.010)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Inject ~30 duplicate rows to test dedup
    dup_idx = rng.choice(n, 30, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print("\n=== Step 1: Generating raw dataset ===")
    df = generate_dataset()
    raw_path = os.path.join(base, "data", "raw", "banking_customers.csv")
    df.to_csv(raw_path, index=False)
    print(f"  Saved -> {raw_path}  ({len(df):,} rows)")

    print("\n=== Step 2: Data cleaning ===")
    from src.data_processing import clean_data, save_processed
    df_clean = clean_data(df)
    proc_path = save_processed(df_clean, base)
    print(f"  Saved -> {proc_path}  ({len(df_clean):,} rows)")

    print("\n=== Step 3: Building SQLite database ===")
    from src.analytics import build_database
    db_path = build_database(df_clean, base)
    print(f"  Database -> {db_path}")

    print("\n=== Step 4: Feature engineering ===")
    from src.feature_engineering import engineer_features
    X, y, feature_names = engineer_features(df_clean)
    print(f"  Features: {len(feature_names)}  |  Samples: {X.shape[0]}")
    from collections import Counter
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    for k, v in Counter(y).items():
        print(f"    {label_map[k]}: {v}")

    print("\n=== Step 5: Training ML models ===")
    from src.model_training import train_all_models
    results, best_model, best_name = train_all_models(X, y, feature_names, base)

    print("\n=== Step 6: Evaluating models ===")
    from src.model_evaluation import evaluate_and_save
    evaluate_and_save(results, best_model, best_name, feature_names, base)

    print(f"\n  Best model: {best_name}")
    print(f"  ROC-AUC: {results[best_name]['roc_auc']}")
    print(f"  F1 (weighted): {results[best_name]['f1_weighted']}")
    print("\n✅  All setup complete!")
    print("    Run:  python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()

