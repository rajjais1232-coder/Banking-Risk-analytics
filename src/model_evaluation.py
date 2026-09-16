"""
src/model_evaluation.py
=======================
Evaluates all trained models and saves metrics/charts.
"""

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve


LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def evaluate_and_save(results: dict, best_model, best_name: str, feature_names: list, base_dir: str):
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    models_dir = os.path.join(base_dir, "models")

    # ── Model comparison CSV ──────────────────────────────────────────────────
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "ROC_AUC": res["roc_auc"],
            "F1_Weighted": res["f1_weighted"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "Best": "✓" if name == best_name else "",
        })
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(models_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Model metrics saved → {metrics_path}")

    # ── Confusion matrix for best model ──────────────────────────────────────
    cm = np.array(results[best_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Med", "High"],
                yticklabels=["Low", "Med", "High"], ax=ax)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    cm_path = os.path.join(charts_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # ── ROC curve (best model, one-vs-rest per class) ────────────────────────
    y_test = results[best_name]["y_test"]
    y_proba = results[best_name]["y_proba"]
    n_classes = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{LABEL_MAP[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC Curves — {best_name}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    roc_path = os.path.join(charts_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # ── Feature importance (RF / DT) ─────────────────────────────────────────
    try:
        clf = best_model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            fi = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
            fi.to_csv(os.path.join(reports_dir, "feature_importance.csv"))
            fig, ax = plt.subplots(figsize=(8, 6))
            fi.head(20).sort_values().plot(kind="barh", ax=ax, color="#3b82d4")
            ax.set_title(f"Feature Importance — {best_name}")
            fig.tight_layout()
            fig.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=150)
            plt.close(fig)
    except Exception:
        pass

    print(f"  Evaluation charts saved → {charts_dir}")
