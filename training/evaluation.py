"""Shared Stratified K-Fold evaluation + MLflow logging for training experiments.

Every experiment stage (model-selection, feature-engineering,
hyperparameter-tuning, ...) evaluates candidate pipelines the same way:
out-of-fold predictions over the pool, full per-class precision/recall,
and the urgency-tier-aware metrics (see docs/technical-decisions.md).
Centralized here so each stage only varies what it's actually searching.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from training.urgency import tier_rank


def category_slug(category: str) -> str:
    """Short, consistent metric-name slug: first word of the category name."""
    return category.split()[0]


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evaluate_and_log(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    cv: StratifiedKFold,
    categories: list[str],
    title: str,
) -> dict:
    """Run CV, log the full metric set + artifacts to the active MLflow run.

    Returns a flat dict (f1_macro_mean/std, undertriage_rate, per-category
    recall) suitable for a results table — not everything logged to MLflow.
    """
    fold_f1_scores = []
    y_pred_oof = pd.Series(index=y.index, dtype=object)

    for train_idx, val_idx in cv.split(X, y):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipeline.predict(X.iloc[val_idx])
        y_pred_oof.iloc[val_idx] = preds
        fold_f1_scores.append(f1_score(y.iloc[val_idx], preds, average="macro"))

    f1_macro_mean = float(np.mean(fold_f1_scores))
    f1_macro_std = float(np.std(fold_f1_scores))
    accuracy = float((y_pred_oof == y).mean())

    report = classification_report(y, y_pred_oof, output_dict=True)

    per_class_metrics = {}
    for category in categories:
        slug = category_slug(category)
        per_class_metrics[f"recall_{slug}"] = report[category]["recall"]
        per_class_metrics[f"precision_{slug}"] = report[category]["precision"]

    true_tier = y.map(tier_rank)
    pred_tier = y_pred_oof.map(tier_rank)
    tier_accuracy = float((true_tier == pred_tier).mean())
    undertriage_rate = float((pred_tier < true_tier).mean())
    overtriage_rate = float((pred_tier > true_tier).mean())

    mlflow.log_metrics(
        {
            "f1_macro_mean": f1_macro_mean,
            "f1_macro_std": f1_macro_std,
            "accuracy": accuracy,
            "tier_accuracy": tier_accuracy,
            "undertriage_rate": undertriage_rate,
            "overtriage_rate": overtriage_rate,
            **per_class_metrics,
        }
    )
    mlflow.log_text(classification_report(y, y_pred_oof), "classification_report.txt")

    cm = confusion_matrix(y, y_pred_oof, labels=categories)
    fig = plot_confusion_matrix(cm, categories, title=title)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    recalls = {category_slug(c): report[c]["recall"] for c in categories}
    return {
        "f1_macro_mean": f1_macro_mean,
        "f1_macro_std": f1_macro_std,
        "undertriage_rate": undertriage_rate,
        "overtriage_rate": overtriage_rate,
        **{f"recall_{slug}": r for slug, r in recalls.items()},
    }
