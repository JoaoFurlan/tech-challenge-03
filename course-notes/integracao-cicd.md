# Integração com CI/CD — GitHub Actions (Etapa 2, discipline 1 of 2)

Course folder: `Integração com CICD (GitHub Actions)/` (Aulas 01–08). This course
frames CI/CD entirely through **MLOps**, not generic DevOps — the specific twist ML
adds to CI/CD that exists in any software project.

## Why MLOps ≠ DevOps

Traditional CI/CD only validates **deterministic code** — same code, same behavior
every time. ML artifacts are **probabilistic** — the same code can produce a
different model depending on data, hyperparameters, and environment. CI must
control four sources of variability:

| Source | How CI controls it |
|---|---|
| Code | Standard linting/tests |
| Data | Version datasets with **DVC** — CI does `dvc pull` to fetch the exact snapshot tied to a commit |
| Hyperparameters | `strategy: matrix` fans out parallel runs across combos |
| Environment | Build the Docker image *before* training — training runs inside the container, not the bare runner |

## Quality gates & shift-left

Cheap checks (lint, unit tests) run on **every commit**; expensive checks (data
contracts, security scans, full training) only run at merge or on schedule.

- **Data contracts** (Great Expectations / Pandera): validate schema/statistics of
  incoming data *before* training — automated GIGO prevention.
- **Security scanning**: Trivy (container CVEs), Bandit (hardcoded secrets in
  Python), Dependabot/CodeQL (dependency vulnerabilities).
- **"Dummy training" on PRs**: fast, tiny-sample run just proves the pipeline works
  end-to-end without paying for a full run; full training only on merge to `main`.

## Notebook hygiene (if prototyping in Jupyter)

Notebooks are bad for git — JSON diffs, embedded execution counters/base64 plot
output create unresolvable merge conflicts. Fixes: **`nbstripout`** (strips outputs
before commit), **`nbqa`** (runs normal linters inside notebook cells). Also:
lockfiles instead of loose `requirements.txt` — a floating `pandas>=1.0` constraint
lets CI silently install a different version than what was tested locally.

## The ML-specific testing triad

1. **Feature-engineering tests** — property-based testing (Hypothesis) throws
   edge-case inputs (nulls, empty strings) at preprocessing to catch silent bugs.
2. **Data contracts** — schema/distribution checks before `model.fit()`.
3. **Behavioral tests** — does an irrelevant input change (typo) leave the
   prediction unchanged? Does a known-direction relationship hold?
4. **API/integration tests via GitHub Actions Service Containers** — spin up the
   built Docker image + FastAPI inside the CI runner, hit it with real JSON, assert
   200 on valid input / 422 (not 500) on malformed input, optionally assert a
   latency budget (fail build if P95 exceeds a threshold).

## Docker specifics mapping directly onto our Dockerfile

- **Load the model once, at startup, in global scope** — never inside the request
  handler.
- **Multi-stage builds**: heavy "builder" stage compiles/installs; only final
  runtime artifacts get copied into a slim final image (e.g. `python:3.10-slim`) —
  can shrink 4GB → ~600MB and removes compilers from the attack surface.
- **Never tag `latest`** — tag by commit SHA (+ optionally semver) so rollback
  means something; `latest` gets silently overwritten.
- **OIDC federation over static AWS keys** — the mechanism behind pushing to ECR
  from GitHub Actions without long-lived secrets. Repeated so often across this
  course (and the Pipeline de Treino folder) that it's clearly the single most
  emphasized security practice taught.

## Scope: what's in vs. out for this project

**Directly usable (Aulas 1–5):** lint→test→build workflow, ≥2 automations (the
challenge's literal requirement), Dockerfile best practices, multi-stage builds,
semantic commits, basic data-contract-style validation on the CSV, OIDC for the
ECR push.

**Good to know, not to build (Aulas 6–8):** Canary/Shadow deployments,
drift-triggered automatic retraining via webhooks, Champion-vs-Challenger model
promotion, Human-in-the-Loop PR bots approving retrained models, org-wide reusable
workflow templates. Enterprise MLOps maturity, well beyond a solo Tech Challenge —
flagged as scope creep if attempted.

## Decisions this topic informed

- **DVC** will be used for dataset versioning (ties dataset snapshots to commits).
- **uv** will be used as the Python dependency/lockfile manager (replaces the
  "use a lockfile instead of loose `requirements.txt`" recommendation — uv chosen
  specifically over Poetry/pip-compile).
- One minimal data-quality check (even just a pandas null/shape assertion) before
  training in the Airflow DAG — small effort, matches the challenge PDF's
  "boas práticas" language.
