"""MLflow experiment 1: model selection.

5 candidate models x 2 TF-IDF configs (conservative/rich), evaluated via
Stratified K-Fold on the training pool. Decision metric: F1-macro,
co-decisive with undertriage_rate (a false negative on cardiovascular, or
any cross-tier miss, is this system's most dangerous failure mode — see
docs/technical-decisions.md). See docs/architecture.md for the full plan.

Result: ComplementNB (conservative TF-IDF) chosen over the raw F1-macro
leader for a substantially lower undertriage_rate — see
docs/technical-decisions.md for the full reasoning. Downstream stages fix
ComplementNB as the model.
"""

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training import mlflow_config
from training.data import load_pool_and_test
from training.evaluation import evaluate_and_log
from training.urgency import CATEGORIES

RANDOM_STATE = 42
N_FOLDS = 5

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

                metrics = evaluate_and_log(
                    pipeline, X, y, cv, CATEGORIES, title=f"{model_name} / {tfidf_name}"
                )
                metrics["model"] = model_name
                metrics["tfidf"] = tfidf_name
                results.append(metrics)

                recall_str = "  ".join(
                    f"{k.removeprefix('recall_')}={v:.3f}"
                    for k, v in metrics.items()
                    if k.startswith("recall_")
                )
                print(
                    f"{model_name:20s} {tfidf_name:12s} "
                    f"f1_macro={metrics['f1_macro_mean']:.4f}±{metrics['f1_macro_std']:.4f}  "
                    f"undertriage={metrics['undertriage_rate']:.4f}  {recall_str}"
                )

    return results


if __name__ == "__main__":
    run()
