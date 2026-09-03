"""CLI entry points for the Airflow retrain-demo DAG (dags/train_pipeline_dag.py).

Airflow runs in its own isolated environment, deliberately without the
training dependencies (scikit-learn/pandas/mlflow) -- see
docs/technical-decisions.md. Each DAG task instead shells out to
`uv run --group training python -m training.airflow_tasks <command>`, so
this stays a thin CLI wrapper around training/train_final.py, not a
duplicate of its logic. The single JSON object each command prints as the
LAST line of stdout is what the calling task parses into its XCom return
value -- everything printed before that (train_final's own progress/metric
prints) is left alone and simply ignored by the parser.
"""

import json
import sys

from training.train_final import save_vocabulary, train_and_evaluate


def train() -> None:
    result = train_and_evaluate()
    print(json.dumps({"model_path": result["model_path"], "f1_macro": result["f1_macro"]}))


def save_model() -> None:
    model_path = sys.argv[2]
    vocabulary_path = save_vocabulary(model_path)
    print(json.dumps({"vocabulary_path": vocabulary_path}))


COMMANDS = {"train": train, "save-model": save_model}

if __name__ == "__main__":
    COMMANDS[sys.argv[1]]()
