# Tech Challenge Fase 3 (POS Tech / MLET)

## What this project is

Postgraduate tech challenge (worth 90% of the grade across all disciplines this term).
Theme: **Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência**.

Scenario: a hospital needs automatic triage of medical text reports (laudos médicos) —
classify urgency as **normal / attention / urgent** — via a lightweight NLP text classifier
served as a REST API in a Docker container.

Full requirements are in `MLET - Tech Challenge Fase 3.pdf` (already read/summarized in-session).

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
6. README: written AWS architecture decision — batch vs real-time — kept concise.
   (The challenge only requires this in writing; we're additionally deploying
   for real via AWS App Runner — see `docs/architecture.md` § AWS architecture.)
7. 5-minute STAR-method video (Situation/Task/Action/Result) demonstrating the project.

Required stack: scikit-learn (or similar) for the model, FastAPI for the API,
`prometheus-client` for instrumentation, Airflow for orchestration.

Grading weights: Modeling/optimization 20%, Monitoring 20%, CI/CD 15%, Airflow 15%,
README 15%, Video 15%.

## Where the actual plan lives

Planning is done — the full build plan and its reasoning live in:

- **`docs/architecture.md`** — the *what*: dataset, repo structure, modeling pipeline
  (candidate models, feature engineering, metrics, train/val/test split), the 4 MLflow
  experiments, latency-optimization branching, tooling, CI/CD, AWS layout, Airflow,
  monitoring, and what's explicitly out of scope.
- **`docs/technical-decisions.md`** — the *why*: rationale behind every non-obvious call,
  especially deliberate simplifications (e.g. why category-classification-then-mapping
  instead of direct urgency labels, why Airflow standalone instead of the production
  docker-compose, why F1-macro over ROC-AUC).
- **`docs/model-card.md`** — model documentation (Mitchell et al. format), not required by
  the challenge but added as standard practice; quantitative sections are placeholders
  until the training pipeline runs.
- **`docs/course-notes/`** — summaries of all 6 course lecture folders (now under
  `docs/course-notes/content/`, gitignored — raw PDFs, not part of the submission),
  written during the planning phase, each tying course concepts back to this project.

Project language is **English throughout** (docs, code, comments, commit messages) —
the only exception is the original challenge PDF itself, which is source material.

## Repo / Git status

Git repo initialized, pushed to `https://github.com/JoaoFurlan/tech-challenge-03-v1`
(private). Commits so far: project context + course notes, architecture/decisions docs,
English-language pass. A pre-push hook on this machine requires interactive confirmation
(commit identity vs. authenticated `gh` account) — pushes need to be run by the user in
their own terminal (`! git push origin main`), not through a tool call.

## Collaboration style for implementation work

When writing code for this project — training pipeline, modeling experiments,
API, everything — **do not build large chunks autonomously and present a
finished result.** Work in small, checkpointed steps and pause for explicit
go-ahead before moving to the next one. This matters most around modeling:
model choices, metric results, feature-engineering decisions, and anything
with a judgment call should be shown and discussed before proceeding — e.g.
run one MLflow experiment, show the results, and ask before moving to the
next one, rather than running all 4 experiments end-to-end and reporting back
only at the finish line. The user wants to see and participate in these
decisions as they happen, not review them after the fact.

## Status

**Planning and documentation phase complete.** All course material reviewed, all major
design decisions made and recorded (see docs above). Dataset confirmed (Medical
Abstracts TC Corpus, 14,438 records, 5 categories). **No implementation code has been
written yet** — `app/`, `training/`, `optimization/`, `dags/`, `frontend/`, `monitoring/`,
`tests/`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml` all still need to be
created per `docs/architecture.md`'s repo structure.

**Blocked on:** AWS CLI configuration (needed to provision the S3 bucket for DVC, the
ECR repository, the IAM OIDC role for GitHub Actions, and the App Runner service for
the real deployment — see `docs/architecture.md` § CI/CD and § AWS architecture).
Everything else can proceed in parallel.

When resumed, read `docs/architecture.md` and `docs/technical-decisions.md` first —
they supersede any older assumptions in this file.
