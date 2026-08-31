# Tech Challenge Fase 3 (POS Tech / MLET)

## What this project is

Postgraduate tech challenge (worth 90% of the grade across all disciplines this term).
Theme: **Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência**.

Scenario: a hospital needs automatic triage of medical text reports (laudos médicos) —
classify urgency as **normal / attention / urgent** — via a lightweight NLP text classifier
served as a REST API in a Docker container.

Full requirements are in `MLET - Tech Challenge Fase 3 (1).pdf` (already read/summarized in-session).

This directory is the **experimentation/build ground**. The final "clean" submission repo
will be a separate repo assembled later from what's built here.

## Required deliverables

1. GitHub repo with CI/CD via GitHub Actions (lint → test → build, ≥2 automations).
2. Airflow DAG (or script) simulating train/retrain: load CSV → train → save model.
3. Dockerfile for the FastAPI inference service.
4. Monitoring stack: API + Prometheus + Grafana via Docker Compose, dashboard with ≥3 panels
   (request count, latency, error rate).
5. Latency optimization: apply ONNX export, quantization, or pruning; compare original vs
   optimized latency with real numbers.
6. README: written (not deployed) AWS architecture decision — batch vs real-time — kept concise.
7. 5-minute STAR-method video (Situation/Task/Action/Result) demonstrating the project.

Required stack: scikit-learn (or similar) for the model, FastAPI for the API,
`prometheus-client` for instrumentation, Airflow for orchestration.

Grading weights: Modeling/optimization 20%, Monitoring 20%, CI/CD 15%, Airflow 15%,
README 15%, Video 15%.

## Decisions made so far

- **Dataset:** Medical Abstracts TC Corpus (Kaggle). Note: its native labels are disease
  categories (neoplasms, digestive, cardiovascular, etc.), not urgency levels — mapping
  those categories (or another signal) to normal/attention/urgent is an **open decision**,
  not yet resolved.
- **Solo project.** This folder is not yet a git repo; that's intentional until we decide
  how to handle git/GitHub for CI/CD (open decision — see below).
- **Cloud architecture doc:** concise (1-2 pages in README), targeting **AWS** specifically,
  justifying real-time vs batch.
- **DVC** will be used for dataset versioning (ties dataset snapshots to git commits).
- **uv** will be used as the Python dependency/lockfile manager (replaces loose
  `requirements.txt`, chosen over Poetry/pip-compile).
- **MLflow** will be used for experiment tracking / model registry (params, metrics,
  artifacts, lifecycle stages).

## Course lecture materials in this repo

- `Deploy em Nuvem/` — lecture PDFs for the "Deploy em Nuvem" discipline (Etapa 1).
  Aula 03 covers AWS specifically and teaches: **ECR** (container image registry),
  **EC2** (real-time inference, full control, good for always-on low-latency APIs),
  **AWS Lambda** (serverless/on-demand, has cold-start), **AWS Batch** (batch jobs),
  **Amazon SageMaker** (managed training/deploy), **S3** (artifact/dataset storage).
  The README's AWS architecture decision should use this vocabulary rather than
  services not taught in the course (e.g. prefer EC2 over ECS/Fargate for the
  real-time API argument, since EC2 is what's explicitly taught for that use case).
- More lecture PDFs for other disciplines (CI/CD, Airflow, Monitoring, Latência) are
  being added by the user before we finalize decisions — **check for new folders/PDFs
  here before resuming planning.**

## Status

Paused mid-planning at the user's request. The user is adding more course lecture PDFs
(for the other 3 disciplines/etapas) to this directory so we can ground remaining
decisions in the actual course content before choosing:

- The urgency label-mapping strategy for the dataset
- The base classifier (TF-IDF + Linear SVM/LogReg vs Random Forest, per PDF's example)
- The latency optimization technique (ONNX vs quantization vs pruning)
- Whether to `git init` / push to a real GitHub repo now (needed for CI/CD to actually run)
  or stay git-free until the "clean" repo is set up

**Do not resume planning or start implementation until the user says so.**
When resumed, first check this directory for newly added lecture PDFs and read
anything relevant before re-engaging on the open decisions above.
