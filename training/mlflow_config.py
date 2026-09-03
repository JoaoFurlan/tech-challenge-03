"""Shared MLflow tracking config for all training/optimization scripts.

The plain filesystem backend ('./mlruns') is in maintenance mode as of the
installed MLflow version and refuses to serve `mlflow ui` — see
docs/technical-decisions.md. Using the recommended local SQLite backend
instead; still fully local, no tracking-server infra.

Overridable via $MLFLOW_TRACKING_URI: the Airflow container (see
airflow/docker-compose.yml) points this at its own separate store rather
than the host's `mlflow.db` -- reusing that db would try to reuse the
`final-model` experiment's already-recorded artifact_location, which is an
absolute Windows path from native runs and isn't writable inside the
Linux container. See docs/technical-decisions.md.
"""

import os

import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def configure() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
