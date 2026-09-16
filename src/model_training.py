"""
src/model_training.py
=====================
Trains Logistic Regression, Decision Tree, and Random Forest classifiers.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
import joblib


def train_all_models(X, y, feature_names, base_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                multi_class="multinomial", solver="lbfgs", random_state=42
            )),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=42
            )),
        ]),
    }

    # Try XGBoost optionally
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                n_jobs=-1, random_state=42
            )),
        ])
        print("  XGBoost available — included.")
    except ImportError:
        print("  XGBoost not installed — skipping.")

    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        results[name] = {
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "roc_auc": round(auc, 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        print(f"  {name}: ROC-AUC={results[name]['roc_auc']}  F1={results[name]['f1_weighted']}")

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["pipeline"]

    # Save best model
    model_path = os.path.join(base_dir, "models", "risk_model.pkl")
    joblib.dump({"model": best_model, "feature_names": feature_names, "best_name": best_name}, model_path)

    return results, best_model, best_name
