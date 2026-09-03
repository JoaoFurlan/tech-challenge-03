# MedSys — Hospital Laudo Triage Classifier

Postgraduate tech challenge (POS Tech / MLET, Fase 3): automatic triage of
medical text reports (laudos médicos) into urgency tiers — **normal /
attention / urgent** — served as a REST API, with a CI/CD pipeline,
monitoring stack, and a latency-optimized model.

This README is filled in progressively as the project is built. See
`docs/architecture.md` (the *what*) and `docs/technical-decisions.md` (the
*why*) for the full build plan and rationale.

**Live deployment**: http://medsys.us-east-1.elasticbeanstalk.com/ — real-time
inference API (`/predict`, `/health`, `/metrics`) running on AWS Elastic
Beanstalk. Not the documented target (App Runner — see `docs/architecture.md`
§ AWS architecture for why); the written justification below reflects the
documented decision regardless. Torn down after the grading/demo window.
Deployed automatically by CI on every push to `main` — GitHub Actions runs
lint → test → build → smoke test → push to ECR → deploy to Beanstalk →
verify health (auto-rollback on failure), no manual steps. See
[§ CI/CD](#cicd) for the full pipeline.

**Demo frontend** (non-graded extra, `frontend/streamlit_app.py`): a
Streamlit UI over `/predict`, hosted separately on **Streamlit Community
Cloud** (free) rather than a second AWS environment — deliberately kept off
the same AWS account to avoid doubling Free Tier EC2 instance-hours for a
component that's just an HTTP client. Deploy URL to be added here once set
up.

**Check the live API works right now**:

```bash
curl http://medsys.us-east-1.elasticbeanstalk.com/health
curl -X POST http://medsys.us-east-1.elasticbeanstalk.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with acute chest pain and shortness of breath, elevated troponin levels observed"}'
```

## Contents

- [Getting started](#getting-started) — setup, running the API, tests, linting
- [Dataset](#dataset)
- [Model](#model)
- [Training & experimentation](#training--experimentation) — reproducing every MLflow run
- [Latency optimization](#latency-optimization)
- [AWS architecture: real-time vs. batch](#aws-architecture-real-time-vs-batch)
- [Data & retrain pipeline flow](#data--retrain-pipeline-flow) — including [running the Airflow DAG](#running-the-dag-locally)
- [Monitoring](#monitoring) — including running it [locally](#running-it-locally) and [on AWS](#running-it-on-aws)
- [CI/CD](#cicd) — what runs on every push, and how to check it

## Getting started

**Prerequisites**: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Docker
Desktop (with `buildx`) for anything container-based, AWS CLI configured
for anything touching S3/DVC or the AWS-hosted pieces.

```bash
git clone https://github.com/JoaoFurlan/tech-challenge-03-v1
cd tech-challenge-03-v1
uv sync --group dev          # base + dev deps (fastapi, pytest, ruff, ...)
```

Dependencies are split into optional `uv` groups so the served Docker image
never installs what it doesn't need (see `Dockerfile`) — pull in more as you
need them:

| Need | Command |
|---|---|
| Run/modify the API, run tests, lint | `uv sync --group dev` |
| Train, retrain, export to ONNX, run MLflow experiments | `uv sync --group training` |
| Run the Streamlit demo frontend | `uv sync --extra frontend` |

**Model artifacts aren't in git** (DVC-tracked, in S3) — pull them before
running the API or training scripts locally:

```bash
uv run --group training dvc pull models/pipeline_fp32.onnx models/vocabulary.json
```

**Run the tests**:

```bash
uv run pytest -q
```

**Lint** (same check CI runs):

```bash
uv run ruff check .
```

**Run the API without Docker** (fastest loop for iterating on `app/`):

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then `curl http://localhost:8000/health`, or open
http://localhost:8000/docs for interactive Swagger docs.

**Run the API + full monitoring stack via Docker** — see
[Monitoring § Running it locally](#running-it-locally) below.

## Dataset

[Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/chaitanyakck/medical-text)
(`sebischair/Medical-Abstracts-TC-Corpus`) — medical abstracts labeled across
5 disease categories, used as ground truth for the classifier. Urgency is
derived from the predicted category via a deterministic mapping layer, not
learned directly (see `docs/technical-decisions.md`).

Kaggle ships this corpus pre-split into train/test files totaling 14,438
rows. We don't use that split as-is — combining the two files surfaced:

- **988 abstracts leaking across the official train/test split** (the same
  document in both files).
- **2,929 abstracts carrying conflicting category labels** — the same
  document text assigned to more than one category. Inspection showed this
  is a real artifact of the source corpus (some documents were originally
  multi-labeled, e.g. both `cardiovascular diseases` and the generic
  `general pathological conditions` bucket), exploded into separate
  single-label rows by this Kaggle release.

Both issues are resolved by combining train+test and dropping every
ambiguous document entirely, rather than arbitrarily keeping one label per
document — full rationale in `docs/technical-decisions.md`. This leaves:

**8,298 clean, unambiguously-labeled documents**, re-split ourselves
(stratified, ~15% held out as a test set, fixed `random_state=42`):

| Category | Count | Share |
|---|---|---|
| General pathological conditions | 2,394 | 28.9% |
| Neoplasms | 2,195 | 26.5% |
| Cardiovascular diseases | 1,961 | 23.6% |
| Nervous system diseases | 1,049 | 12.6% |
| Digestive system diseases | 699 | 8.4% |

Still comfortably above the challenge's 2,000-sample minimum, with class
imbalance (~3.4x) essentially unchanged from the raw corpus (~3.2x).

## Model

4 staged MLflow experiments (model-selection, feature-engineering,
hyperparameter-tuning, latency-optimization) — full search grids and
reasoning in `docs/technical-decisions.md`. The model chosen,
**ComplementNB**, was *not* the top scorer on the primary classification
metric (F1-macro); it was chosen for a substantially lower rate of
dangerous triage misses (predicting a lower urgency tier than the truth),
an explicit, documented safety-for-accuracy trade — this pattern (the
raw-accuracy winner isn't the safety winner) recurred at every stage of
tuning, not just model choice.

**Final pipeline**: TF-IDF (unigram, 10,000 features) + ComplementNB
(`alpha=0.5, norm=True`).

**Final test-set result** (1,245 held-out documents, evaluated once):

| Metric | Value |
|---|---|
| F1-macro | 0.7786 |
| Accuracy | 0.7880 |
| Undertriage rate (dangerous misses) | 0.0498 |
| Overtriage rate (false alarms) | 0.1446 |

**Known limitation, confirmed via real-world testing, not just assumed**:
the model is trained on formal medical *abstracts* (academic, third-person
register), not real triage phrasing. Testing realistic short/informal
inputs post-deployment found genuine misses — e.g. "Unresponsive, no
detectable pulse, non-breathing" classified as `normal`. Two related bugs
this surfaced were fixed (the keyword-adjustment layer now scales with
signal strength instead of capping at one tier, and `/predict` now flags
`low_confidence: true` for inputs with no real vocabulary overlap), but the
core register mismatch isn't fixable by adding more of the same training
data — it would need real triage-style text, a genuine scope increase. Full
writeup in `docs/technical-decisions.md` and `docs/model-card.md` §
Caveats. This is exactly the kind of limitation `docs/model-card.md`
already exists to document plainly rather than hide.

## Training & experimentation

Every step below is a plain script, runnable directly (`uv sync --group
training` first) — the Airflow DAG (see
[§ Running the DAG locally](#running-the-dag-locally)) automates the last
two of these (`ingest` + `train`) for the retrain-demo use case, but the
scripts underneath are the same ones you can run by hand for development.

| Step | Command | MLflow experiment |
|---|---|---|
| Model selection | `uv run python -m training.model_selection` | `model-selection` |
| Feature engineering | `uv run python -m training.feature_engineering` | `feature-engineering` |
| Hyperparameter tuning | `uv run python -m training.hyperparameter_tuning` | `hyperparameter-tuning` |
| Final train + evaluate | `uv run python -m training.train_final` | `final-model` |
| ONNX export + latency benchmark | `uv run python -m optimization.export_and_benchmark` | `latency-optimization` |

Each of the first four stages was run once, its results reviewed, and the
result fed into the next stage's design — full grids and reasoning for
every stage in `docs/technical-decisions.md`, not just the final numbers
above. `training.train_final` is the one that actually produces
`models/pipeline.joblib` + `models/vocabulary.json` (what the DAG's
`train`/`save_model` tasks call); `optimization.export_and_benchmark`
turns that into `models/pipeline_fp32.onnx` (what the API actually
serves) and produces the latency comparison table below.

Inspect any run's metrics/params/artifacts (confusion matrices, MLflow's
own model registry entries) via MLflow's UI, pointed at the same local
SQLite store all of these write to:

```bash
uv run --group training mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Latency optimization

Exported to ONNX (`skl2onnx`) and compared against dynamic INT8
quantization, benchmarked as full single-document `/predict` latency
(TF-IDF vectorization + inference), 500 requests:

| Variant | P50 | P95 | P99 | F1-macro | Size |
|---|---|---|---|---|---|
| sklearn baseline | 0.595ms | 0.814ms | 1.040ms | 0.7786 | 1,233KB |
| **ONNX FP32 (served)** | **0.135ms** | **0.263ms** | **0.339ms** | 0.7786 | 413KB |
| ONNX INT8 | 0.163ms | 0.281ms | 0.397ms | 0.7803 | 267KB |

**Served: ONNX FP32** — 4.4x faster than the sklearn baseline at P50, 3x
smaller, mathematically exact (identical F1-macro, not an approximation).
INT8 quantization was actually *slower* than FP32 here, not faster — at
this scale the whole model already runs in a fraction of a millisecond,
so quantization's per-call dequantization overhead outweighs its compute
savings. It does deliver a real size win (35% smaller than FP32) if
footprint matters more than latency. Full reasoning in
`docs/technical-decisions.md`.

## AWS architecture: real-time vs. batch

Three deploy patterns exist for a model like this — the right one depends on
what the workload actually needs, not a general preference:

| | Batch | Real-time | Serverless |
|---|---|---|---|
| Latency | High, tolerated | Low, deterministic | Variable (cold start) |
| Cost shape | Concentrated in scheduled runs | Constant (infra always on) | Aligned to demand |
| Fits this project? | No — no recurring bulk workload to run over | **Yes** | No — cold start unacceptable |

**Real-time**, not batch: a laudo comes in and a hospital needs an
urgent/attention/normal answer immediately, not after the next scheduled
run — the whole point of triage is catching the urgent case *now*. Batch
also has nothing to batch here — there's no recurring stream of laudos
queued up for offline processing, just individual reports arriving one at
a time. Serverless (Lambda) is ruled out for the same underlying reason as
batch is ruled in against: per-invocation cold start is incompatible with
consistent low latency in a clinical tool, even though it would otherwise
fit the "one request at a time" shape. The real-time pattern's downside —
cost is constant rather than usage-proportional, since the model has to
stay loaded and warm — is an acceptable, deliberate trade for a use case
where latency has direct clinical consequences.

Within AWS's real-time options specifically:

- **App Runner** (documented target) — always-warm container, no
  instance to provision or patch, point it at an ECR image tag and it
  runs. Closest fit to "real-time inference with minimal ops burden."
- **Raw EC2** — same always-on behavior, but the ops burden (provisioning,
  patching, health checks) that App Runner exists to remove falls back on
  us.
- **Lambda** — ruled out above (cold start).
- **AWS Batch** — wrong shape entirely; built for scheduled/bulk jobs, not
  a standing inference endpoint.
- **SageMaker** — a full managed ML platform (model registry, managed
  endpoints, governance) — real capability, but disproportionate setup for
  a single lightweight classifier.

**Actually deployed on Elastic Beanstalk, single-instance Docker
platform**, not App Runner — a late, real constraint, not a design
change: App Runner isn't part of AWS Free Tier, discovered only when
attempting the real deployment. Beanstalk on a single free-tier EC2
instance is the closest Free-Tier-eligible match to App Runner's intent —
same always-warm, no-cold-start property, and Beanstalk still absorbs most
of the manual EC2 operational burden (provisioning, health checks,
deployment) that App Runner would have handled directly. Specifically the
single-instance environment type, not load-balanced/auto-scaling — that
tier provisions an Elastic Load Balancer, which is billed separately and
isn't Free Tier eligible. The reasoning above is still the *documented*
architecture decision; the deployed service differs from it for this one
budget-driven reason, called out explicitly rather than silently swapped.
Full pivot story in `docs/technical-decisions.md`.

A production-hardened version of this same real-time architecture would
also need API rate limiting (anti-abuse and cost control — real-time's
constant-cost shape makes unbounded traffic a direct cost leak, not just a
security concern) and encryption in transit/at rest for laudo text, since
it's patient data.

## Data & retrain pipeline flow

How a laudo dataset turns into a live deployed model, and how that differs
between this project's local Airflow setup and what a hosted, production
version would look like. Airflow here is **standalone mode, manually
triggered** — it demonstrates retrain orchestration, it isn't wired into
automatic deployment (see `docs/technical-decisions.md` for why that
extra wiring is out of scope here).

**Today: local Airflow, DAG output is a manual hand-off**

```
[Kaggle CSVs] (one-time)
      |
      v
data/ (local)  --dvc add + dvc push-->  S3 (DVC remote)
                                              |
   ==== Airflow DAG boundary (manual trigger) ====
   |                                          |
   |  ingest task: dvc pull  <----------------+
   |       |
   |       v
   |  load_raw() / split_test()  -->  pool (85%) + test (15%, held out)
   |       |
   |       v
   |  train task: fit Pipeline(TF-IDF + ComplementNB) on pool
   |       |
   |       v
   |  evaluate on test  -->  log metrics/params to MLflow
   |       |
   |       v
   |  save_model task: write pipeline.joblib + vocabulary.json (local disk)
   |
   ==== DAG ends here ====
                |
                v  (manual step, not in the DAG)
     optimization/export_and_benchmark.py
                |
                v
     pipeline_fp32.onnx + latency benchmark numbers
                |
     dvc add + dvc push (manual) --> S3
                |
     commit .dvc pointers + git push (manual) --> GitHub main
                |
                v  (automatic from here -- CI/CD already wired)
     ci-cd.yml: lint -> test -> dvc pull (model) -> docker build
                |
                v
     push to ECR --> update Dockerrun.aws.json --> deploy to Elastic Beanstalk
                |
                v
     Live API (/predict, /metrics) --scraped by--> Prometheus --> Grafana
```

The DAG's own output (`pipeline.joblib`) isn't what the live API serves
(`pipeline_fp32.onnx`) — connecting the two today is a manual chain
(export, `dvc push`, `git push`), not automatic.

**Production-level: hosted Airflow, fully wired (not built here)**

```
[Real data source: hospital DB / event stream / scheduled export]
      |
      v  (scheduled OR event-triggered -- no human clicking "run")
   ==== Airflow DAG boundary (hosted: MWAA or EC2, always running) ====
   |
   |  ingest task: query DB / consume event / pull new S3 batch
   |       |
   |       v
   |  train task: fit + evaluate + log to MLflow
   |       |
   |       v
   |  quality gate: compare new metrics vs. current production model
   |       |            (skip deploy if the new model is worse)
   |       v
   |  export task: ONNX export + benchmark  <-- new task, not in today's DAG
   |       |
   |       v
   |  publish task: dvc push (Airflow's own S3 write credentials)
   |       |
   |       v
   |  commit + push task: update .dvc pointers, push to main
   |       |              (bot git identity/token, not a person's)
   |
   ==== DAG ends here, but it just triggered CI/CD ====
                |
                v  (automatic, same pipeline as today)
     ci-cd.yml: lint -> test -> dvc pull -> build -> push ECR -> deploy
                |
                v
     Live API updated automatically, no human in the loop
```

The ML steps look almost the same in both — the real difference is that
today's DAG output is a dead end a human has to manually carry the rest of
the way, while the hosted version closes the loop itself, plus adds a
quality gate that doesn't exist today to stop a worse retrain from
auto-deploying. Full cost/scope reasoning for why this isn't built for
real here is in `docs/technical-decisions.md`.

### Running the DAG locally

**Runs in Docker, not directly on the host.** Apache Airflow doesn't
support native Windows at all (it depends on POSIX-only APIs) — this
isn't the "official production docker-compose is disproportionate for a
3-task demo" tradeoff mentioned above, it's a hard OS incompatibility with
no workaround short of a different OS or a container. `airflow/` holds a
small dedicated Dockerfile + `docker-compose.yml` (standalone mode, one
container) — separate from the root `docker-compose.yml`, since this is a
different concern (orchestrating retraining, not serving/observing the
API) and isn't part of that stack.

The container mounts this repo read-write, so `ingest`/`train`/`save_model`
write real files back into `data/`/`models/` on your machine, same as
running the training scripts directly. It does **not** share the host's
`mlflow.db` — reusing it would try to reuse the `final-model` experiment's
already-recorded artifact path, which is an absolute Windows path and
isn't writable from Linux, so the container gets its own separate MLflow
store instead (`training/mlflow_config.py`, `$MLFLOW_TRACKING_URI`).

```bash
cd airflow
docker compose up --build -d
```

The container prints a generated admin password to its logs on first
boot — `docker compose logs | Select-String "Password for user"`
(PowerShell) or `docker compose logs | grep "Password for user"`
(bash/zsh) — then open http://localhost:8081 and log in as `admin` with
that password.

Trigger a run (UI: click the play button on `train_pipeline_dag`, or CLI):

```bash
docker compose exec airflow airflow dags unpause train_pipeline_dag
docker compose exec airflow airflow dags trigger train_pipeline_dag
```

**Verified working end-to-end**, not just "should work" — run twice against
the real dataset (not a toy/mocked run): both runs completed all three
tasks (`ingest` → `train` → `save_model`) successfully, confirming
idempotency, and the metrics the second run logged and printed matched the
README's reported numbers above **exactly** (`f1_macro=0.7786`,
`undertriage=0.0498`) — genuine reproducibility of the already-reported
result from a completely different OS/environment (Linux container vs.
the native Windows runs that originally produced those numbers), not
coincidence.

Check a run's state:

```bash
docker compose exec airflow airflow tasks states-for-dag-run \
  train_pipeline_dag "<run_id from the trigger output>"
```

Tear down when done (the named volume keeps Airflow's own state — admin
user, DAG pause state, its MLflow store — across restarts; add `-v` to
also wipe that):

```bash
docker compose down
```

## Monitoring

Docker Compose stack: api + prometheus + grafana, dashboard
auto-provisioned (not clicked together manually) with the 3 required
panels: request rate, latency (P50/P95/P99), error rate. Prometheus scrapes
both the local `api` service and the live AWS deployment, so the dashboard
shows real production traffic alongside local dev traffic in the same
panels. Verified end-to-end in an actual browser session with real
generated traffic, not just that Grafana accepted the dashboard JSON —
caught and fixed a real query bug in the process (the error-rate panel
showed ambiguous "No data" instead of an explicit 0% with zero errors).
Full story in `docs/technical-decisions.md`.

### Running it locally

```
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 (`/predict`, `/health`, `/metrics`) |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (anonymous viewer access, no login) |

Generate some traffic to see the dashboard populate — e.g.
`curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "..."}'`,
or point the Streamlit frontend's `API_URL` at `http://localhost:8000`.

### Running it on AWS

**Deployed and verified** — real EC2 instance, real generated traffic,
dashboard confirmed rendering it in an actual browser session, not just
"should work." Unlike the API (always-on, auto-deployed by CI on every
push), this stack runs on a plain EC2 instance stood up specifically for
the grading/demo window, then torn down — see `docs/technical-decisions.md`
for why raw EC2 rather than Beanstalk here (the multi-container setup
doesn't fit Beanstalk's single-container Docker platform without
disproportionate extra config for something temporary).

**How it was stood up** (for reference / standing up a fresh instance):

1. Launch a plain EC2 instance (t2/t3.micro), IAM instance profile with S3
   read (DVC bucket) + SSM permissions, security group open only on
   Grafana's port (3000) — no SSH port needed, access is via SSM Session
   Manager.
2. Install Docker + Compose + `docker-buildx` (user-data script at launch
   — `buildx` specifically, not just the `docker-compose` CLI plugin,
   since `docker compose build` needs it and it isn't installed by
   default).
3. `git clone` this repo, `dvc pull` the model artifacts (via `uv`/`uvx`,
   not plain `pip` — much faster dependency resolution on a small
   instance, see `docs/technical-decisions.md`).
4. `docker compose up -d --build` — the exact same file as local, nothing
   different for the deployed version.
5. Verify at `<instance-public-ip>:3000`.

**Auto-redeploys on push** — `.github/workflows/deploy-monitoring.yml`
watches for pushes touching `docker-compose.yml`, `monitoring/**`,
`app/**`, `Dockerfile`, or `models/*.dvc`, and redeploys via **AWS Systems
Manager Run Command** (no SSH, no keys — the same OIDC role already used
for the API, extended with a scoped `ssm:SendCommand` policy for this one
instance): `git pull` → `dvc pull` → `docker compose up -d --build`, all
on the instance, triggered from CI. Check its status the same way as any
other workflow — GitHub → Actions tab → "Deploy monitoring stack".

### How this would differ in a real production environment

What's simplified here for a 2-week solo challenge vs. what a real
always-on clinical deployment would need:

- **Managed observability, not a single Docker Compose instance**: Amazon
  Managed Service for Prometheus + Amazon Managed Grafana (or a Prometheus
  Operator on EKS) for high availability, proper retention/backup, and no
  single point of failure — one EC2 instance running `docker-compose` is a
  demo convenience, not a production pattern.
- **Alerting, not just dashboards**: Grafana/Alertmanager rules wired to
  actual on-call paging (PagerDuty, Opsgenie, etc.) — a dashboard nobody is
  actively watching doesn't catch an incident at 3am.
- **Authenticated access with RBAC**, not anonymous viewer — used here
  purely for local demo convenience.
- **Model-quality monitoring alongside operational metrics** — this stack
  only covers request count/latency/error rate (infrastructure health);
  real deployment would add drift detection (PSI/KS, mentioned as
  forward-looking in `docs/model-card.md`, not implemented here) to catch
  the model silently degrading even while the service itself stays healthy.
- **Permanent infrastructure, not spin-up/tear-down** — the temporary EC2
  pattern used for this demo would become continuously-provisioned,
  likely autoscaled/HA infrastructure in a real deployment.

## CI/CD

Two independent GitHub Actions workflows — independent because they watch
different trigger paths and target different resources, so they run
concurrently when a push touches both, not sequentially:

| Workflow | Triggers on | Does |
|---|---|---|
| `.github/workflows/ci-cd.yml` ("CI/CD") | Every push to `main` (and PRs, minus the deploy job) | lint → test → build → **smoke test** → push to ECR → deploy to Beanstalk → health check (**auto-rollback** on failure) |
| `.github/workflows/deploy-monitoring.yml` ("Deploy monitoring stack") | Pushes touching `docker-compose.yml`, `monitoring/**`, `app/**`, `Dockerfile`, or `models/*.dvc` | Redeploys the EC2 monitoring stack via SSM (see [§ Running it on AWS](#running-it-on-aws)) |

**Check status**: GitHub → **Actions** tab lists every run, newest first,
by workflow name. A failed `ci-cd.yml` run means either a real code
problem (lint/test failure) or a deploy that got rejected — read the
failing step's log; the deploy step's automated rollback means production
should still be healthy even if the job itself is red.

**What's gated before anything reaches production**:

- `lint`/`test` must pass before `build-and-push` even starts.
- The **smoke test** runs the actual built image and makes a real
  `/predict` call, checking the response is well-formed (right schema,
  known category/urgency values) — before the image is ever pushed to ECR.
  This catches "the container is fundamentally broken" (missing model
  file, crashing code path); it does **not** catch "the model's
  predictions got worse" — that needs ground truth, a different, larger
  problem that was considered and deliberately not built (see
  `docs/technical-decisions.md`).
- After deploy, a health check failure triggers an **automatic rollback**
  to the previously-live version, so a bad deploy self-heals within the
  same run instead of leaving production broken until someone notices.

**Does every push retrain the model?** No — `ci-cd.yml` only `dvc pull`s
whatever model artifact the committed `.dvc` pointer files currently
reference; it never calls the training code. The model in production only
changes when someone deliberately updates those pointers (train → export
to ONNX → `dvc push` → commit the new `.dvc` files → push) — see
[§ Data & retrain pipeline flow](#data--retrain-pipeline-flow) for the
full chain and why it's a manual hand-off today, not automatic.
