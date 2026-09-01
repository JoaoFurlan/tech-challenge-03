# Architecture

Living record of the plan for the Tech Challenge Fase 3 project — hospital laudo
triage classifier. Written before implementation starts; updated as we build.
Rationale/trade-offs for each decision live in `technical-decisions.md`.

## Problem framing

Classify a medical text report (laudo) into urgency tier **normal / attention /
urgent**, served as a real-time REST API. Rather than inventing noisy urgency
labels to train against directly, the model is trained on the dataset's real
ground truth — 5 disease categories — and urgency is derived from that
prediction via a deterministic, documented mapping layer.

## Dataset

**Medical Abstracts TC Corpus** (Kaggle / `sebischair/Medical-Abstracts-TC-Corpus`).
14,438 labeled abstracts across 5 categories:

| Category | Count |
|---|---|
| General pathological conditions | 4,805 |
| Neoplasms | 3,163 |
| Cardiovascular diseases | 3,051 |
| Nervous system diseases | 1,925 |
| Digestive system diseases | 1,494 |

Moderate imbalance (~3.2x between largest and smallest class), well above the
challenge's 2,000-sample minimum. Hosted on **S3**, versioned via **DVC** — the
training pipeline pulls it (`dvc pull`), it isn't committed as a raw CSV.

**Processed dataset: 8,298 documents**, not 14,438. Kaggle ships this corpus
pre-split into train (11,550 rows) and test (2,888 rows); combining them
(`training/data.py`) revealed 988 abstracts leaking across that split, and
2,929 abstracts carrying *conflicting* category labels (the same document
tagged under 2+ categories — see `technical-decisions.md`). Both the original
split and the ambiguous documents are discarded; the clean 8,298 documents are
re-split ourselves (~15% held-out test, fixed `random_state=42`, stratified).
Class balance is preserved post-cleaning (~3.4x, general pathological
conditions 2,394 / digestive 699).

## Urgency mapping (deterministic, post-prediction)

The trained classifier predicts one of the 5 categories. Urgency is then derived
in two steps:

**1. Category → baseline urgency tier**

| Category | Baseline |
|---|---|
| Cardiovascular diseases | urgent |
| Nervous system diseases | attention |
| Neoplasms | attention |
| Digestive system diseases | attention |
| General pathological conditions | normal |

**2. Per-abstract keyword adjustment** (case-insensitive regex over the abstract
text; net score = escalate hits − de-escalate hits; positive nudges the tier up
one step, negative nudges it down one step, capped at urgent / floored at
normal):

- Escalate (+1): `acute`, `emergency`, `severe`, `critical`, `sudden`,
  `life-threatening`
- De-escalate (−1): `chronic`, `stable`, `routine`, `mild`, `follow-up`,
  `long-term`

Word lists are a starting proposal — to be sanity-checked against real abstract
text once the pipeline is running, not assumed correct in advance.

## Repo structure

```
tech-challenge-03-v1/
├── .github/workflows/ci.yml       # lint -> test -> build -> push to ECR (OIDC)
├── app/                            # FastAPI service
│   ├── main.py                     # /predict, /metrics endpoints
│   ├── model.py                    # load pipeline artifact, predict category
│   └── urgency.py                  # category baseline + keyword adjustment
├── training/
│   ├── model_selection.py          # MLflow experiment 1
│   ├── feature_engineering.py      # MLflow experiment 2
│   ├── hyperparameter_tuning.py    # MLflow experiment 3
│   └── train_final.py              # retrain winner on full train+val, save artifact
├── optimization/
│   └── export_and_benchmark.py     # ONNX export + quantization/pruning + latency benchmark (MLflow experiment 4)
├── dags/
│   └── train_pipeline_dag.py       # Airflow: dvc pull -> train -> save (+ MLflow log)
├── frontend/
│   └── streamlit_app.py            # demo UI calling the FastAPI /predict endpoint
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/       # datasource + dashboard JSON, auto-provisioned
├── tests/
├── data/                            # gitignored, DVC-tracked
├── models/                          # gitignored, model artifacts
├── mlflow.db                        # gitignored, MLflow run/experiment metadata (SQLite)
├── mlruns/                          # gitignored, MLflow run artifacts (figures, reports)
├── Dockerfile                       # multi-stage build
├── docker-compose.yml               # api + prometheus + grafana (+ streamlit)
├── pyproject.toml                   # uv-managed
├── README.md                        # public-facing, incl. AWS architecture section
└── docs/
    ├── architecture.md               # this file
    ├── technical-decisions.md
    ├── model-card.md
    └── course-notes/                 # lecture material summaries (already written)
        ├── cloud-deployment.md
        ├── cicd-integration.md
        ├── training-pipeline.md
        ├── performance-monitoring.md
        ├── monitoring-services.md
        ├── latency-performance.md
        └── content/                  # raw lecture PDFs, gitignored
```

## Modeling pipeline

### Candidate models (model-selection stage)

Logistic Regression, LinearSVC, Multinomial Naive Bayes, Complement Naive Bayes,
Random Forest. `class_weight="balanced"` wherever supported (no resampling
library — unnecessary at this imbalance ratio).

### Feature engineering — TF-IDF

Two representative configs tested across all 5 candidates in model-selection
(catches a model whose ranking depends on feature richness, without paying for
the full grid):

| | Conservative | Rich |
|---|---|---|
| `ngram_range` | (1,1) | (1,2) |
| `max_features` | 5,000 | 10,000 |
| `min_df` | 2 | 2 |
| `sublinear_tf` | False | True |
| `stop_words` | english | english |

Full grid (`ngram_range` × `max_features` × `min_df` × `sublinear_tf`, 36
combinations) searched only on the winning model, in a separate stage.

### Metrics & evaluation

- **Selection metric: F1-macro** (unweighted mean across the 5 categories — a
  model can't win purely by being good at the largest class).
- Supporting evidence logged for every run: per-class precision/recall, accuracy
  (context only, not decisive), confusion matrix (artifact — critical for
  catching a cardiovascular abstract misclassified as general pathological,
  which would silently under-triage downstream).
- Winner confirmed on F1-macro, then explicitly cross-checked against
  cardiovascular-class recall before finalizing.
- ROC-AUC/PR-AUC and MCC considered, not adopted — multi-class complexity
  (OvR averaging, `predict_proba` requirement, LinearSVC needing
  `CalibratedClassifierCV`) not worth it for a 2-week solo project; F1-macro +
  per-class recall + confusion matrix already cover the same ground.

### Train/validation/test split

1. Held-out **test set** (~15–20%, stratified), carved out once at the start
   with a fixed `random_state`. Never touched during model-selection,
   feature-engineering, or hyperparameter-tuning.
2. Remaining pool (~80–85%) used for all three experimentation stages via
   **Stratified K-Fold** (same folds reused across stages, same seed, for
   comparability).
3. Final step: retrain the fully-chosen pipeline (model + TF-IDF config +
   hyperparameters) on the entire pool, evaluate once on the held-out test set
   — that number goes in the README. Same artifact gets exported to ONNX.

**Leakage prevention**: `sklearn.pipeline.Pipeline` bundles the TF-IDF vectorizer
and classifier into one object — when passed to `cross_validate`/`GridSearchCV`,
the vectorizer is automatically refit on only the training fold at each split,
structurally preventing vocabulary/IDF leakage from validation data. Duplicate
check on raw abstracts before splitting. Same fitted `Pipeline` object is saved,
loaded by FastAPI, and exported to ONNX — one artifact, no train/serve mismatch.

### MLflow — 4 experiments, ~60–85 runs total, local tracking (`mlflow ui`)

| Experiment | Runs | What varies | Decision metric |
|---|---|---|---|
| `model-selection` | 10 | 5 models × 2 TF-IDF configs | F1-macro + `undertriage_rate` (tier-aware, co-decisive — see `technical-decisions.md`) |
| `feature-engineering` | 36 + 6 follow-up | full TF-IDF grid + max_df/stop_words follow-up, ComplementNB fixed | F1-macro + `undertriage_rate` |
| `hyperparameter-tuning` | ~10–30 | ComplementNB hyperparams (`alpha`, `norm`), via `GridSearchCV` | F1-macro + `undertriage_rate` |
| `latency-optimization` | ~3 | baseline vs. ONNX FP32 vs. ONNX INT8-quantized | P50/P95/P99 latency |

**Model-selection result: ComplementNB (conservative TF-IDF)** —
F1-macro 0.765, `undertriage_rate` 0.053, chosen over the raw F1-macro
leader (LinearSVC/rich, 0.798) for a substantially lower dangerous-miss
rate at an accepted, documented accuracy cost. Full reasoning in
`technical-decisions.md`. Downstream stages (feature-engineering,
hyperparameter-tuning) fix ComplementNB as the model and search its
surrounding config space instead of re-opening model choice.

**Feature-engineering result: `ngram_range=(1,1)`, `max_features=10000`,
`min_df=2`, `sublinear_tf=False`, `stop_words="english"`** — F1-macro
0.773, `undertriage_rate` 0.0525. Confirms the original "conservative"
TF-IDF config was already close to the safety frontier; bigrams win on
raw F1-macro but consistently worsen undertriage_rate, so unigram-only was
kept. `max_df` tested with no measurable effect. Full reasoning in
`technical-decisions.md`.

### Latency optimization (Etapa 4) — branches by winning model type

- **Linear winner** (LogReg/LinearSVC): ONNX export (`skl2onnx`) + dynamic INT8
  quantization (`onnxruntime.quantization`).
- **Random Forest winner**: ONNX export (`skl2onnx`, `TreeEnsembleClassifier` op)
  + cost-complexity pruning (`ccp_alpha`) and/or reduced `n_estimators` —
  quantization doesn't meaningfully apply to tree ensembles (no dense weight
  matrices to compress).
- Report **P50/P95/P99** over the *whole* `/predict` pipeline (preprocessing +
  inference + response), not just `model.predict()` — preprocessing can be
  40–60% of total latency in an unoptimized system.

## Tooling

- **uv** — dependency management, replaces `requirements.txt`.
- **DVC + S3** — dataset versioning, real remote (not local-only), pulled by the
  Airflow ingest task.
- **MLflow** — local SQLite-backed tracking (`mlflow.db`), no server infra
  beyond the `mlflow ui` process itself, viewed via `mlflow ui` in browser.
  Not the plain filesystem backend (`mlruns/`) originally planned — the
  installed MLflow version puts that backend in maintenance mode and refuses
  to serve it; see `technical-decisions.md`.

## API & Docker

FastAPI, `/predict` (plain `def`, not `async def` — CPU-bound inference) and
`/metrics` (prometheus_client). Model loaded once at startup, global scope.
Multi-stage Dockerfile (builder installs deps, slim runtime copies only final
artifacts). Uvicorn as the ASGI server.

## CI/CD

GitHub Actions: **lint (ruff) → test (pytest) → build → push to ECR**.
Authentication via **OIDC** (IAM role trusting `token.actions.githubusercontent.com`,
scoped to this repo) — no static AWS keys stored as GitHub secrets. Image tagged
by commit SHA (never `latest`).

## AWS architecture (real-time deploy) — actually deployed, not just written up

The challenge only requires this decision to be documented in the README, not
deployed. We're deploying it for real anyway — a live endpoint is a stronger
STAR-video "Result" than a localhost recording, and the chosen service has low
enough operational overhead to make that extra realism cheap.

- **App Runner** — always-on inference, model stays loaded in memory (no
  Lambda-style per-invocation cold start), fully managed: points at a tagged
  image in ECR and runs it, no EC2 instance, no SSH, no security groups, no OS
  patching. Chosen over raw EC2 for the same real-time rationale with far less
  operational overhead.
- **ECR** — image registry, fed by CI; App Runner deploys directly from a
  tagged image here.
- **S3** — DVC remote for the dataset.
- Full written justification (batch vs. real-time, App Runner vs. Lambda vs.
  Batch vs. SageMaker vs. raw EC2) still goes in `README.md` — that section is
  graded regardless of whether deployment is real.
- Service torn down after the grading/demo window to avoid ongoing billing.

## Airflow

**Standalone mode** (`airflow standalone`, SQLite backend) — not the official
multi-container production docker-compose (Postgres + Redis + webserver +
scheduler + worker), which is disproportionate for a 3-task demo DAG. 3
`@task`-decorated tasks: `dvc pull` (ingest) → train → save model artifact
(+ log to MLflow). Demonstrated via manual trigger — no live data stream feeding
this project, so there's no real recurring retrain need; the DAG proves the
orchestration capability works, run at least twice to confirm idempotency.

## Monitoring

`docker-compose.yml`: api + prometheus + grafana. `prometheus_client` Counter
(request count, error count) and Histogram (latency — gives P50/P95/P99 for
free). Grafana dashboard **auto-provisioned** (datasource + dashboard JSON as
files, not manual clicking) — 3 required panels: request count, latency, error
rate.

## Demo frontend (non-graded extra)

Lightweight **Streamlit** app, separate service, calls the FastAPI `/predict`
endpoint over HTTP (doesn't duplicate model logic). Purely for a better visual
in the STAR video than Swagger docs/curl — explicitly called out as a nicety,
not a required deliverable.

## Explicitly out of scope (see `technical-decisions.md` for full rationale)

Kubernetes/HPA/KEDA, Canary/Shadow deployment, drift detection implementation
(PSI/KS — concept mentioned in README as forward-looking, not built), word/
transformer embeddings (BERT/ClinicalBERT), AutoML/joint Bayesian search across
model+features+hyperparameters, ROC-AUC/PR-AUC/MCC, live production traffic of
any kind.
