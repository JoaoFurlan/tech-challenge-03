# Pipeline de Treino e Deploy Automático (Etapa 2, discipline 2 of 2)

Course folder: `Pipeline de Treino e Deploy Automático/` (Aulas 01–08). Pairs with
`course-notes/integracao-cicd.md` for Etapa 2 (GitHub Actions workflow + Airflow DAG).

## MLOps pipeline anatomy — the TFX reference model (Aula 1)

The model's training code is a small fraction of a real ML system — the rest
(validating data, engineering features, evaluating quality, checking infra
readiness, serving) is "hidden technical debt." Google's TFX names each stage:

| Component | Function |
|---|---|
| ExampleGen | Ingest + split data |
| StatisticsGen | Descriptive stats on the dataset |
| SchemaGen | Infer types/domains — living data contract |
| ExampleValidator | Detect anomalies/drift, blocks bad data before training |
| Transform | Feature engineering |
| Trainer | Train the model |
| Evaluator | Deep validation, slice analysis (fairness) |
| InfraValidator | Load-test the model in a sandboxed server |
| Pusher | Publish to registry |

**Training-serving skew** is the single most important concept here: if the
feature-transformation code at training time isn't *exactly* the same code that
runs at inference, predictions silently diverge from what was validated offline.
scikit-learn's `Pipeline` (preprocessing + model as one artifact) makes this
structurally impossible rather than something to remember.

## Data ingestion & feature engineering (Aula 2)

- **GIGO** (Garbage In, Garbage Out) — data-quality gates should reject bad data
  *before* it reaches training.
- **Logical vs. physical separation** (KeystoneML): separate *what* a
  transformation does (a pure, deterministic definition — e.g. "compute TF-IDF
  vectors from this text column") from *how* it executes (single-threaded,
  distributed, GPU, etc.). Without this separation, you can't change execution
  strategy without rewriting the logic, and can't test logic without standing up
  infrastructure.
  - Maps directly onto scikit-learn's `fit()`/`transform()`: an **Estimator** is a
    factory that, given training data, *produces* a Transformer
    (`TfidfVectorizer().fit(texts)` → a fitted vectorizer holding the learned
    vocabulary/IDF weights). The **Transformer** is then a pure, reusable function
    (`fitted_vectorizer.transform(new_text)`) callable identically anywhere,
    without knowing how it was originally fitted.
  - This is why bundling everything into `sklearn.pipeline.Pipeline` is good
    practice: it packages "text in → prediction out" as one artifact, usable
    identically in training and in the FastAPI endpoint — preventing
    training-serving skew.
- **Feature scaling** is a hard requirement for gradient-based models — unscaled
  features with different magnitudes destabilize gradient descent.

## Model training & validation (Aula 3) — important for our specific model

- **Bias-variance tradeoff**: high bias = underfitting, high variance = overfitting.
- **MLflow**: experiment tracking + model registry — logs params/metrics/artifacts,
  manages lifecycle stages (Staging → Production → Archived). Chosen for this
  project.
- **Accuracy is "dangerous and misleading" on imbalanced datasets.** Our
  normal/atenção/urgente classes are almost certainly imbalanced (most triage
  cases probably aren't "urgente"). **Stratified K-Fold cross-validation** (keeps
  class proportions consistent across folds) is the right validation strategy, and
  **recall on the "urgente" class specifically** should be weighted over overall
  accuracy — missing an urgent case (false negative) is worse than a false alarm.
  Worth stating explicitly as a modeling decision in the README/video.

## Deploy de Modelos — Implantação Inicial (Aula 4)

Confirms Etapa 1 patterns rather than introducing new ones:
- Load the model once, at startup, in global scope — never inside the request handler.
- Use a production ASGI server (Uvicorn, behind Gunicorn as process manager),
  sized to CPU cores — never the framework's dev server in production.
- Dockerfile layer ordering: dependency-install step before code-copy step, so
  code changes don't invalidate the (slow) dependency layer cache.
- Multi-stage builds, slim base images.

## Orquestração de Pipelines com Airflow (Aula 5) — the actual Etapa 2 deliverable

**What Airflow is:** a workflow **orchestrator** — its only job is running a set of
tasks in the right order, at the right time, handling failures and dependencies
automatically.

**Core object: the DAG** (Directed Acyclic Graph) — tasks are nodes, dependencies
are edges (e.g. "train" can't start until "ingest" *actually succeeded*, not just
"time has passed"). Acyclic = no task depends on itself, guaranteeing termination.

**Why not cron:** cron only knows about *time*. It has no concept of "did the
previous step succeed," no per-step retry logic, no UI into what ran/failed, and a
mid-pipeline failure requires manually figuring out what to re-run. Airflow tracks
per-task state, retries with backoff, and lets you re-run just the broken step.

**Where it fits in the ML lifecycle for this project** — three separate tools,
three separate concerns:
- **GitHub Actions** — runs on every git push, validates *code* (lint/test),
  builds/pushes the Docker image.
- **Airflow** — runs the *training/retraining process* itself, on demand or on a
  schedule: ingest CSV → train pipeline → save model artifact (+ log to MLflow).
- **FastAPI + Docker (EC2)** — the always-on service that *uses* whatever model
  artifact currently exists to answer real-time requests.

**Airflow's pieces**: Scheduler (watches DAGs, schedules ready tasks), Metadata DB
(execution state/audit trail), Webserver (UI), Executor + Workers.

**TaskFlow API** — current idiomatic style: plain Python functions decorated with
`@task` instead of the older `PythonOperator` boilerplate; Airflow handles passing
data between tasks automatically.

**Anti-pattern flagged:** don't push large data volumes through Airflow's metadata
DB — stage data in storage, orchestrate pointers to it. Not a concern at our CSV
scale.

**Scope for this challenge:** a genuinely small DAG — "load CSV → train → save
model," 2–3 tasks, `@task`-decorator style, `LocalExecutor`/`SequentialExecutor`
complexity. No distributed workers, no event-driven triggers needed.

**Why Airflow matters here despite no real retraining need:** the challenge asks
to "simular o processo de treinamento" — demonstrating the *capability* to
orchestrate retraining correctly, not solving an actual recurring-retraining
problem (there's no live stream of new labeled laudos feeding this project). Two
things still matter in practice:
1. The DAG should actually be **triggered/run** (manually, at least once — ideally
   twice, to prove idempotency: a second run produces a fresh model artifact
   without erroring or duplicating state), not just exist as unexecuted code —
   that's what "DAG funcional" actually grades.
2. Fair to state plainly in the README/video: "in production, this DAG would be
   triggered on a schedule or by a drift signal from the monitoring stack; here
   it's demonstrated via manual trigger since there's no live data stream." In a
   real production system with an actual stream of new labeled data, this is
   exactly where Airflow earns its keep — running the same DAG repeatedly over
   the model's lifetime, handling failures, giving an audit trail of what ran when.

## Reprodutibilidade e Qualidade do Código (Aula 6)

- **Set and log random seeds** (`random_state` in scikit-learn) — stochastic
  training can converge differently across runs otherwise. Cheap, worth doing.
- **SOLID applied to ML pipelines**: don't tangle data-loading, feature-engineering,
  and validation code together.
- **Cyclomatic complexity** (PyLint) as a quantifiable code-quality metric —
  optional polish, not required by the challenge's CI/CD spec (lint + test only).

## Treinamento Automático e Re-Treino / CI/CD de ML (Aulas 7–8) — mostly out of scope

- Continuous Training, drift detection (**PSI** >0.25 threshold, **KS test**),
  Shadow Deployment for validating a retrained "challenger" against the current
  "champion" before promoting it.
- Kubernetes HPA/KEDA, Blue-Green/Canary deploys — enterprise-scale serving.

Not buildable within this challenge's scope (no live production traffic, no
Kubernetes) — but "how would you know if this model needs retraining?" is worth
one forward-looking sentence in the README.

## What's load-bearing vs. flavor

**Directly usable:** Stratified K-Fold + recall-weighted evaluation for imbalanced
urgency classes, `random_state` seeding, multi-stage Docker build,
TaskFlow-API-style Airflow DAG, MLflow experiment tracking, OIDC for GitHub
Actions → AWS.

**Good context, not required:** TFX vocabulary, drift detection concepts.

**Out of scope entirely:** Kubernetes/HPA/KEDA, Blue-Green/Canary, AutoML, PyTorch
Lightning (scikit-learn, not deep learning, is our path).
