"""MLflow experiment 1: model selection.

5 candidate models x 2 TF-IDF configs (conservative/rich), evaluated via
Stratified K-Fold on the training pool. Decision metric: F1-macro,
cross-checked against cardiovascular-disease recall (a false negative there
is this system's most dangerous failure mode — see
docs/technical-decisions.md). See docs/architecture.md for the full plan.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training import mlflow_config
from training.data import load_pool_and_test

RANDOM_STATE = 42
N_FOLDS = 5
CATEGORIES = [
    "cardiovascular diseases",
    "digestive system diseases",
    "general pathological conditions",
    "neoplasms",
    "nervous system diseases",
]

TFIDF_CONFIGS = {
    "conservative": {
        "ngram_range": (1, 1),
        "max_features": 5000,
        "min_df": 2,
        "sublinear_tf": False,
        "stop_words": "english",
    },
    "rich": {
        "ngram_range": (1, 2),
        "max_features": 10000,
        "min_df": 2,
        "sublinear_tf": True,
        "stop_words": "english",
    },
}


def build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "linear_svc": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "multinomial_nb": MultinomialNB(),
        "complement_nb": ComplementNB(),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


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


def run() -> list[dict]:
    pool, _ = load_pool_and_test()
    X = pool["medical_abstract"].reset_index(drop=True)
    y = pool["condition_name"].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlflow_config.configure()
    mlflow.set_experiment("model-selection")

    results = []
    for tfidf_name, tfidf_params in TFIDF_CONFIGS.items():
        for model_name, model in build_models().items():
            with mlflow.start_run(run_name=f"{model_name}__{tfidf_name}"):
                pipeline = Pipeline(
                    [("tfidf", TfidfVectorizer(**tfidf_params)), ("clf", model)]
                )

                mlflow.log_params({f"tfidf_{k}": v for k, v in tfidf_params.items()})
                mlflow.log_params({"model": model_name, "tfidf_config": tfidf_name})

                fold_f1_scores = []
                y_pred_oof = pd.Series(index=y.index, dtype=object)

                for train_idx, val_idx in cv.split(X, y):
                    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                    preds = pipeline.predict(X.iloc[val_idx])
                    y_pred_oof.iloc[val_idx] = preds
                    fold_f1_scores.append(
                        f1_score(y.iloc[val_idx], preds, average="macro")
                    )

                f1_macro_mean = float(np.mean(fold_f1_scores))
                f1_macro_std = float(np.std(fold_f1_scores))
                accuracy = float((y_pred_oof == y).mean())

                report = classification_report(y, y_pred_oof, output_dict=True)

                per_class_metrics = {}
                for category in CATEGORIES:
                    slug = category.replace(" ", "_")
                    per_class_metrics[f"recall_{slug}"] = report[category]["recall"]
                    per_class_metrics[f"precision_{slug}"] = report[category]["precision"]

                mlflow.log_metrics(
                    {
                        "f1_macro_mean": f1_macro_mean,
                        "f1_macro_std": f1_macro_std,
                        "accuracy": accuracy,
                        **per_class_metrics,
                    }
                )
                mlflow.log_text(
                    classification_report(y, y_pred_oof), "classification_report.txt"
                )

                cm = confusion_matrix(y, y_pred_oof, labels=CATEGORIES)
                fig = plot_confusion_matrix(
                    cm, CATEGORIES, title=f"{model_name} / {tfidf_name}"
                )
                mlflow.log_figure(fig, "confusion_matrix.png")
                plt.close(fig)

                recalls = {c: report[c]["recall"] for c in CATEGORIES}
                results.append(
                    {
                        "model": model_name,
                        "tfidf": tfidf_name,
                        "f1_macro_mean": f1_macro_mean,
                        "f1_macro_std": f1_macro_std,
                        **{f"recall_{c}": r for c, r in recalls.items()},
                    }
                )
                recall_str = "  ".join(f"{c.split()[0]}={r:.3f}" for c, r in recalls.items())
                print(
                    f"{model_name:20s} {tfidf_name:12s} "
                    f"f1_macro={f1_macro_mean:.4f}±{f1_macro_std:.4f}  {recall_str}"
                )

    return results


if __name__ == "__main__":
    run()
