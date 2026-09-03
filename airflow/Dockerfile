# Airflow for the train/retrain demo DAG. Airflow doesn't support native
# Windows (see docs/technical-decisions.md) -- this container is how it
# actually runs. Standalone mode only (`airflow standalone`, one process),
# same "manually triggered, not scheduled" role documented in README.md /
# docs/architecture.md -- not the official multi-service production compose.
FROM apache/airflow:3.3.1-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Each DAG task shells out to `uv run --group training ...` rather than
# importing training code into Airflow's own interpreter -- keeps Airflow's
# dependencies isolated from scikit-learn/pandas/mlflow, avoiding any
# resolver conflict between the two. See training/airflow_tasks.py.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/airflow/.local/bin:${PATH}"

# uv's project venv would otherwise land inside the bind-mounted repo
# (/opt/airflow/project/.venv), colliding with the host's own Windows
# .venv for the main API -- redirect it to a container-local path instead.
ENV UV_PROJECT_ENVIRONMENT=/home/airflow/training-venv

# Own MLflow store, isolated from the host's mlflow.db -- see
# training/mlflow_config.py for why sharing it breaks on Linux.
RUN mkdir -p /opt/airflow/mlflow-training
ENV MLFLOW_TRACKING_URI=sqlite:////opt/airflow/mlflow-training/mlflow.db
