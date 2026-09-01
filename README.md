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
