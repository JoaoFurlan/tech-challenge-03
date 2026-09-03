"""Simulated train/retrain pipeline: dvc pull -> train -> save model artifact.

Manually triggered, not scheduled -- there's no live data stream feeding this
project, so there's no real recurring retrain need. This DAG demonstrates the
orchestration capability itself; see README.md § Data & retrain pipeline flow
and docs/architecture.md / docs/technical-decisions.md § Airflow for the full
reasoning, including why Airflow runs in its own isolated environment here
rather than sharing the project's main uv-managed one.

Each task shells out to `uv run --group training ...` rather than importing
training code directly, since Airflow's own environment deliberately doesn't
have scikit-learn/pandas/mlflow installed. `train` and `save_model` hand off
only a file path through XCom, not the fitted pipeline object -- see
training/airflow_tasks.py and training/train_final.py.
"""

import json
import subprocess
from pathlib import Path

import pendulum
from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> str:
    """Run a subprocess in the project root, streaming its output into the
    Airflow task log, and return its stdout for the caller to parse.

    Prints output before checking the return code -- `check=True` would
    raise before a failing command's own stdout/stderr ever reached the
    task log, hiding the actual error.
    """
    result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result.stdout


@dag(
    dag_id="train_pipeline_dag",
    description="Simulated train/retrain: dvc pull -> train -> save model artifact.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["medsys"],
)
def train_pipeline_dag():
    @task
    def ingest() -> None:
        """dvc pull the tracked dataset -- same S3 remote used everywhere else."""
        _run(["uv", "run", "--group", "training", "dvc", "pull", "data.dvc"])

    @task
    def train() -> str:
        """Fit + evaluate + log to MLflow + save the pipeline; returns its path."""
        stdout = _run(
            [
                "uv",
                "run",
                "--group",
                "training",
                "python",
                "-m",
                "training.airflow_tasks",
                "train",
            ]
        )
        payload = json.loads(stdout.strip().splitlines()[-1])
        print(f"f1_macro={payload['f1_macro']:.4f}")
        return payload["model_path"]

    @task
    def save_model(model_path: str) -> str:
        """Derive vocabulary.json from the saved pipeline (see training/train_final.py)."""
        stdout = _run(
            [
                "uv",
                "run",
                "--group",
                "training",
                "python",
                "-m",
                "training.airflow_tasks",
                "save-model",
                model_path,
            ]
        )
        payload = json.loads(stdout.strip().splitlines()[-1])
        return payload["vocabulary_path"]

    ingest_task = ingest()
    model_path = train()
    save_model(model_path)
    ingest_task >> model_path


train_pipeline_dag()
