# Performance Monitoring (Etapa 3, discipline 1 of 2)

Course folder: `content/Monitoração de Performance/` (Aulas 01–08). Despite the folder
name, this is mostly a **model-optimization** course (hyperparameter tuning,
compression, architecture patterns) — only Aula 8 is actual Prometheus/Grafana
content. Pairs with `monitoring-services.md` for Etapa 3.

## Latência vs. Throughput fundamentals (Aula 1) — high importance

- **Latency** = time for one request round-trip. **Throughput** = requests handled
  per unit time. They trade off against each other.
- **Never monitor latency with the mean** — long tails (GC pauses, network blips)
  hide behind averages. Use **percentiles**: P50 (typical), P95 (contention
  visible), **P99** (worst 1% — what SLAs should target). Cited stat: every 100ms
  of added latency costs Amazon ~1% in sales.
- **Batching vs. GPU utilization**: a single request massively underuses a GPU
  (~1-5% utilized at batch size 1). Batching raises throughput but adds queueing
  latency — mitigated via **dynamic micro-batching** (buffer until max size OR max
  wait time, whichever comes first).

**Our project:** report **P50/P95/P99** for the Etapa 4 before/after latency
comparison, not a single average. Measure the *whole* `/predict` endpoint (JSON
parsing + TF-IDF vectorization + inference + response), not just
`model.predict()` — preprocessing can be 40-60% of total latency in an
unoptimized system (Amdahl's Law: optimizing only the model won't shrink total
latency if most time is spent elsewhere).

## Hyperparameter tuning & overfitting (Aula 2) — medium-high importance

- **Grid Search** (exhaustive, exponential cost) vs. **Random Search** (usually
  beats Grid Search in high dimensions) vs. **Bayesian Optimization** (Optuna —
  probabilistic model of hyperparameters→performance, converges in fewer trials).
- **Regularization**: **L1 (Lasso)** — zeros many weights, automatic feature
  selection, sparse model; **L2 (Ridge)** — shrinks smoothly, no zeroing;
  **Elastic Net** combines both.
- **Stratified K-Fold** reinforced again — essential for imbalanced classes.

**Our project:** light Random/Bayesian search (Optuna) over regularization
strength `C` for Logistic Regression/SVM — concrete tuning approach and README
vocabulary beyond "we picked defaults."

## Model compression: pruning, quantization, distillation (Aula 3) — core Etapa 4 material, with a caveat

- **Pruning**: unstructured (zero individual weights, high compression but needs
  sparsity-aware hardware to actually speed up) vs. structured (remove whole
  neurons/channels, genuinely faster on any hardware, hurts accuracy more).
- **Quantization**: FP32 → INT8. **PTQ** (quantize an already-trained model
  directly, fast/easy) vs. **QAT** (simulates quantization during training,
  better accuracy retention, industry standard when accuracy matters).
- **Knowledge Distillation**: small "student" mimics large "teacher's" soft
  probability outputs.

**Important caveat:** every technique/example here is framed for deep neural
networks (CNNs, Transformers, PyTorch) — our model (TF-IDF + Logistic
Regression/Linear SVM) is a small linear model, which changes the practical story:
- **ONNX export + dynamic INT8 quantization is the technique that cleanly
  transfers** — `skl2onnx` converts the fitted model to ONNX, ONNX Runtime
  supports post-training dynamic quantization of the linear weight matrix. Real,
  supported, produces measurable improvement.
- **"Pruning" has no clean equivalent for a linear model** (no lottery-ticket
  iterative retraining of a subnetwork). Closest legitimate parallel: **L1-regularized
  Logistic Regression already produces a sparse coefficient vector** by zeroing
  uninformative TF-IDF features — framable as a "pruning-equivalent
  sparsification step" in the README, but not the same mechanism the course
  teaches — claiming it as literal "pruning" without this caveat would look like
  a category error.

**Recommendation:** **ONNX export + quantization** as the primary Etapa 4 path
(really one combined step — quantize the ONNX-exported model); note pruning as
"considered but not directly applicable to our linear-model architecture" rather
than forcing it.

## Unsupervised model optimization (Aula 4) — low relevance

K-Means vs. DBSCAN, Silhouette Score, curse of dimensionality (PCA/UMAP), Spark
MLlib/RAPIDS cuML for scale. Not directly relevant — our classifier is
supervised, single-machine, small dataset. Only interesting if clustering ever
helps explore/design the urgency label taxonomy.

## Batch vs. real-time service architecture (Aula 5) — high importance for framing

- **Lambda Architecture** (separate batch + speed layers, merged at serving) vs.
  **Kappa Architecture** (one continuous stream, no separate batch layer, simpler).
- Reinforces Airflow's DAG/dependency model and fail-fast data-quality gates.
- **Drift detection detail**: **KS test** and **PSI** (finance-industry standard,
  less sensitive to sample-size false positives than raw p-values).

**Our project:** reinforcing vocabulary for the batch-vs-real-time README
argument and for framing the Airflow DAG conceptually (ours will be far simpler).

## Infrastructure for high throughput (Aula 6) — one actionable rule, rest out of scope

- **FastAPI async/await rule, directly actionable**: only use `async def` when
  everything inside actually awaits async I/O. A blocking call (synchronous
  scikit-learn inference) inside `async def` blocks the whole event loop for all
  other requests. FastAPI automatically runs plain `def` functions in a
  threadpool — the correct choice for CPU-bound inference.
- NVIDIA Triton, NGINX load balancing, Kubernetes HPA — beyond a single-container
  FastAPI + docker-compose project.

**Action item:** the `/predict` endpoint should be `def`, not `async def` —
classifier inference is CPU-bound, synchronous work.

## Orchestration & scaling at production scale (Aula 7) — low implementation relevance

Kubernetes pods, KServe scale-to-zero, **Canary Release** (small % traffic to new
model, ramp gradually) and **Shadow Deployment** (mirror all traffic, discard
output, zero user-facing risk) — good narrative for "how would you roll out a new
model safely in a hospital triage tool" in the README, not implemented.

## Monitoring & maintenance (Aula 8) — very high importance, Etapa 3 template

- **Prometheus**: pull-based (app exposes `/metrics`, Prometheus scrapes on
  schedule) — exactly `prometheus-client` + FastAPI.
- **Counter** (only increases) → "request count" panel. **Histogram** (buckets
  observations, gives percentiles) → "latency" panel, P50/P95/P99 for free.
  **Error rate panel** = a Counter labeled by HTTP status, rated via PromQL for
  5xx responses.
- Example code is essentially a template: FastAPI middleware timing every
  request, incrementing Counter/Histogram, exposed at `/metrics`.

**Directly satisfies the challenge's "≥3 panels: request count, latency, error
rate" requirement** with the exact metric types needed — re-read closely when
building the monitoring stack.

## Bottom line

Most load-bearing: Aula 3 (compression, Etapa 4, with linear-model caveat) and
Aula 8 (Prometheus, Etapa 3). Strong secondary: Aula 1 (percentile reporting),
Aula 2 (tuning practice), Aula 5 (Airflow/architecture framing), and the
FastAPI `async`/`def` rule from Aula 6. Low relevance: Aulas 4 and 7.
