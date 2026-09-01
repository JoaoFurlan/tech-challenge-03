"""MLflow experiment 2 (follow-up): max_df sweep + stop_words comparison.

Anchored on the near-optimal region found by the main 36-combo grid
(training/feature_engineering.py): ngram_range=(1,1), max_features=10000,
min_df=2, sublinear_tf=False — unigram-only clearly beat unigram+bigram on
undertriage_rate there. Tests two dimensions that grid didn't cover:
max_df (drop overly common boilerplate terms) and stop_words (English list
vs. none), see docs/technical-decisions.md.
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

ANCHOR = {
    "ngram_range": (1, 1),
    "max_features": 10000,
    "min_df": 2,
    "sublinear_tf": False,
}

GRID = {
    "max_df": [1.0, 0.7, 0.5],
    "stop_words": ["english", None],
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
        variant = dict(zip(keys, combo, strict=True))
        params = {**ANCHOR, **variant}

        stop_words_str = params["stop_words"] or "none"
        run_name = f"followup_maxdf{params['max_df']}_stopwords{stop_words_str}"

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
                f"{run_name:35s} f1_macro={metrics['f1_macro_mean']:.4f}  "
                f"undertriage={metrics['undertriage_rate']:.4f}  {recall_str}"
            )

    return results


if __name__ == "__main__":
    run()
