"""MLflow experiment 2: feature engineering.

Full TF-IDF grid (36 combinations) with the model-selection winner
(ComplementNB, default hyperparameters — see docs/technical-decisions.md)
fixed. Same evaluation protocol as model-selection: F1-macro co-decisive
with undertriage_rate. Hyperparameter tuning of ComplementNB itself
(alpha, norm) is the next, separate stage.
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

GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [5000, 10000, 20000],
    "min_df": [1, 2, 5],
    "sublinear_tf": [True, False],
}


def run() -> list[dict]:
    pool, _ = load_pool_and_test()
    X = pool["medical_abstract"].reset_index(drop=True)
    y = pool["condition_name"].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlflow_config.configure()
    mlflow.set_experiment("feature-engineering")

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo, strict=True))
        params["stop_words"] = "english"

        ngram_str = f"{params['ngram_range'][0]}-{params['ngram_range'][1]}"
        run_name = (
            f"ngram{ngram_str}"
            f"_maxfeat{params['max_features']}"
            f"_mindf{params['min_df']}"
            f"_sublinear{params['sublinear_tf']}"
        )

        with mlflow.start_run(run_name=run_name):
            pipeline = Pipeline(
                [("tfidf", TfidfVectorizer(**params)), ("clf", ComplementNB())]
            )
            mlflow.log_params({f"tfidf_{k}": v for k, v in params.items()})
            mlflow.log_param("model", "complement_nb")

            metrics = evaluate_and_log(pipeline, X, y, cv, CATEGORIES, title=run_name)
            metrics.update(params)
            results.append(metrics)

            recall_str = "  ".join(
                f"{k.removeprefix('recall_')}={v:.3f}"
                for k, v in metrics.items()
                if k.startswith("recall_")
            )
            print(
                f"{run_name:45s} f1_macro={metrics['f1_macro_mean']:.4f}  "
                f"undertriage={metrics['undertriage_rate']:.4f}  {recall_str}"
            )

    return results


if __name__ == "__main__":
    run()
