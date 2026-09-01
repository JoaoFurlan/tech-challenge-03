"""Shared MLflow tracking config for all training/optimization scripts.

The plain filesystem backend ('./mlruns') is in maintenance mode as of the
installed MLflow version and refuses to serve `mlflow ui` — see
docs/technical-decisions.md. Using the recommended local SQLite backend
instead; still fully local, no tracking-server infra.
"""

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"


def configure() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
