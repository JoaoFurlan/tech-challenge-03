"""MLflow experiment 3: hyperparameter tuning.

ComplementNB hyperparameters (alpha, fit_prior, norm), with the TF-IDF
config fixed to the feature-engineering result (see
docs/technical-decisions.md). Originally planned via GridSearchCV +
mlflow.sklearn.autolog() per docs/architecture.md, but autolog's child-run
creation for GridSearchCV throws internally on the installed MLflow
version (only the parent run gets logged, no per-candidate visibility) —
switched to the same manual per-combination logging used in
model_selection.py / feature_engineering.py, which also gives full
per-class precision/recall and confusion matrices per candidate (bare
GridSearchCV scoring wouldn't have). Decision protocol: F1-macro
co-decisive with undertriage_rate, same as prior stages.
"""

import itertools

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from training import mlflow_config
from training.data import load_pool_and_test
from training.evaluation import evaluate_and_log
from training.urgency import CATEGORIES

RANDOM_STATE = 42
N_FOLDS = 5

TFIDF_PARAMS = {
    "ngram_range": (1, 1),
    "max_features": 10000,
    "min_df": 2,
    "sublinear_tf": False,
    "stop_words": "english",
}

GRID = {
    "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    "norm": [True, False],
    "fit_prior": [True, False],
}


def run() -> list[dict]:
    pool, _ = load_pool_and_test()
    X = pool["medical_abstract"].reset_index(drop=True)
    y = pool["condition_name"].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlflow_config.configure()
    mlflow.set_experiment("hyperparameter-tuning")

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo, strict=True))
        run_name = (
            f"alpha{params['alpha']}_norm{params['norm']}_fitprior{params['fit_prior']}"
        )

        with mlflow.start_run(run_name=run_name):
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                    ("clf", ComplementNB(**params)),
                ]
            )
            mlflow.log_params({f"clf_{k}": v for k, v in params.items()})
            mlflow.log_params({f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()})

            metrics = evaluate_and_log(pipeline, X, y, cv, CATEGORIES, title=run_name)
            metrics.update(params)
            results.append(metrics)

            recall_str = "  ".join(
                f"{k.removeprefix('recall_')}={v:.3f}"
                for k, v in metrics.items()
                if k.startswith("recall_")
            )
            print(
                f"{run_name:35s} f1_macro={metrics['f1_macro_mean']:.4f}  "
                f"undertriage={metrics['undertriage_rate']:.4f}  {recall_str}"
            )

    return results


if __name__ == "__main__":
    run()
