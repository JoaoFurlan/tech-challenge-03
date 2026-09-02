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
lint → test → build → push to ECR → deploy to Beanstalk → verify health,
no manual steps.

**Demo frontend** (non-graded extra, `frontend/streamlit_app.py`): a
Streamlit UI over `/predict`, hosted separately on **Streamlit Community
Cloud** (free) rather than a second AWS environment — deliberately kept off
the same AWS account to avoid doubling Free Tier EC2 instance-hours for a
component that's just an HTTP client. Deploy URL to be added here once set
up.

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

### Running it on AWS (for the demo video)

**Not deployed yet as of this writing** — this is the plan, not a
completed step. Unlike the API (always-on, auto-deployed by CI on every
push), this stack is stood up temporarily on a plain EC2 instance
specifically for recording the STAR video, then torn down — see
`docs/technical-decisions.md` for why raw EC2 rather than Beanstalk here
(the multi-container setup doesn't fit Beanstalk's single-container Docker
platform without disproportionate extra config for something temporary).

1. Launch a plain EC2 instance (t2/t3.micro), IAM instance profile with S3
   read (DVC bucket) + SSM permissions, security group open only on
   Grafana's port (3000) — no SSH port needed.
2. Install Docker + Compose (user-data script at launch).
3. `git clone` this repo, `dvc pull` the model artifacts.
4. `docker compose up -d --build` — the exact same file as local, nothing
   different for the deployed version.
5. Verify at `<instance-public-ip>:3000`, record the video segment.
6. Terminate the instance once done.

CI/CD to this instance (via AWS Systems Manager Run Command, no SSH keys
involved — same OIDC role already used for the API) is a planned addition
once the instance exists to target.

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
