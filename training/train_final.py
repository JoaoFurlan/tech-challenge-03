"""Final model: retrain the fully-chosen pipeline on the full pool,
evaluate exactly once on the held-out test set, and save the artifact.

Pipeline: ComplementNB(alpha=0.5, norm=True) + TfidfVectorizer(unigram,
max_features=10000, min_df=2, sublinear_tf=False, stop_words="english") —
see docs/architecture.md / docs/technical-decisions.md for the full
reasoning behind each choice. This is the ONE evaluation against the test
set held out since training/data.py's initial split — never touched
during model-selection, feature-engineering, or hyperparameter-tuning.
The saved artifact (models/pipeline.joblib) is what the FastAPI app loads
and what optimization/export_and_benchmark.py exports to ONNX.
"""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from training import mlflow_config
from training.data import load_pool_and_test
from training.evaluation import category_slug, plot_confusion_matrix
from training.urgency import CATEGORIES, tier_rank

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "pipeline.joblib"
VOCABULARY_PATH = MODELS_DIR / "vocabulary.json"

TFIDF_PARAMS = {
    "ngram_range": (1, 1),
    "max_features": 10000,
    "min_df": 2,
    "sublinear_tf": False,
    "stop_words": "english",
}
CLF_PARAMS = {"alpha": 0.5, "norm": True}


def run() -> dict:
    pool, test = load_pool_and_test()
    X_pool, y_pool = pool["medical_abstract"], pool["condition_name"]
    X_test, y_test = test["medical_abstract"], test["condition_name"]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", ComplementNB(**CLF_PARAMS)),
        ]
    )
    pipeline.fit(X_pool, y_pool)

    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)

    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    accuracy = float((y_pred == y_test).mean())
    report = classification_report(y_test, y_pred, output_dict=True)

    true_tier = y_test.map(tier_rank)
    pred_tier = y_pred.map(tier_rank)
    tier_accuracy = float((true_tier == pred_tier).mean())
    undertriage_rate = float((pred_tier < true_tier).mean())
    overtriage_rate = float((pred_tier > true_tier).mean())

    per_class_metrics = {}
    for category in CATEGORIES:
        slug = category_slug(category)
        per_class_metrics[f"recall_{slug}"] = report[category]["recall"]
        per_class_metrics[f"precision_{slug}"] = report[category]["precision"]

    mlflow_config.configure()
    mlflow.set_experiment("final-model")
    with mlflow.start_run(run_name="complement_nb_final"):
        mlflow.log_params({f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()})
        mlflow.log_params({f"clf_{k}": v for k, v in CLF_PARAMS.items()})
        mlflow.log_params({"pool_size": len(X_pool), "test_size": len(X_test)})

        mlflow.log_metrics(
            {
                "f1_macro": f1_macro,
                "accuracy": accuracy,
                "tier_accuracy": tier_accuracy,
                "undertriage_rate": undertriage_rate,
                "overtriage_rate": overtriage_rate,
                **per_class_metrics,
            }
        )
        mlflow.log_text(
            classification_report(y_test, y_pred), "classification_report.txt"
        )

        cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
        fig = plot_confusion_matrix(
            cm, CATEGORIES, title="Final model — held-out test set"
        )
        mlflow.log_figure(fig, "confusion_matrix.png")

        mlflow.sklearn.log_model(pipeline, name="model")

    MODELS_DIR.mkdir(exist_ok=True)
    dump(pipeline, MODEL_PATH)

    # app/model.py's low-confidence guard needs the vectorizer's vocabulary
    # to detect near-empty TF-IDF feature vectors (e.g. "stomachache" --
    # not a substring match of "stomach", genuinely absent from training
    # vocabulary) -- see docs/technical-decisions.md. Exported as plain
    # JSON so the served app doesn't need scikit-learn/joblib at runtime.
    vocabulary = sorted(pipeline.named_steps["tfidf"].vocabulary_.keys())
    VOCABULARY_PATH.write_text(json.dumps(vocabulary))

    recall_str = "  ".join(
        f"{k.removeprefix('recall_')}={v:.3f}"
        for k, v in per_class_metrics.items()
        if k.startswith("recall_")
    )
    print(f"Test set: {len(X_test)} documents (never touched before this run)")
    print(f"f1_macro={f1_macro:.4f}  accuracy={accuracy:.4f}")
    print(
        f"tier_accuracy={tier_accuracy:.4f}  undertriage={undertriage_rate:.4f}  "
        f"overtriage={overtriage_rate:.4f}"
    )
    print(recall_str)
    print(f"Saved pipeline to {MODEL_PATH}")

    return {
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "undertriage_rate": undertriage_rate,
        "overtriage_rate": overtriage_rate,
        **per_class_metrics,
    }


if __name__ == "__main__":
    run()
