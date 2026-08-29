# Serviços de Monitoração (Etapa 3, discipline 2 of 2)

Course folder: `Serviços de Monitoração/` (Aulas 01–08, filenames just
"POSTECH - Aula 01" through "Aula 08"). This is the actual Prometheus/Grafana
observability deep-dive, complementing
`course-notes/monitoracao-performance.md`'s Aula 8 for Etapa 3.

## Observability fundamentals (Aula 01) — high importance

**Monitoring vs. observability** — not synonyms: monitoring is reactive/
threshold-based ("alert if CPU > 90%"); observability is proactive — can you
reconstruct *any* internal state from your outputs, without shipping new
instrumentation to answer a new question? Traces to 1960s control theory
(Kálmán): a system is "observable" if its outputs let you fully reconstruct
internal state.

**Three pillars, each answering a different question:**

| Pillar | Answers | Nature | Cost |
|---|---|---|---|
| Metrics | "how much / when" | Numeric, aggregable | Cheap |
| Logs | "what happened" (per-request) | Textual, detailed | Moderate |
| Traces | "where did time go" (cross-service) | Connected spans | Higher |

**Why ML observability is uniquely hard:** a broken model doesn't throw
exceptions — it keeps returning HTTP 200 while silently getting worse. Two named
failure modes: **data drift** (input distribution shifts) and **concept drift**
(the relationship between features and the correct answer shifts). Detected via
statistical distance metrics: **PSI**, **KL divergence**, **KS test**,
**Jensen-Shannon divergence**. Not implemented in this project, but good
vocabulary for explaining why "the API returns 200" isn't sufficient monitoring
for ML specifically. Cited stat: ~85% of companies running models in production
report silent-degradation incidents quarterly.

**Tooling landscape:** extend existing infra (Prometheus+exporters,
Grafana+ML plugins — our path) vs. dedicated ML-monitoring platforms
(Evidently AI, WhyLabs, Arize, Fiddler) for out-of-the-box drift/explainability.

## Prometheus mechanics (Aula 02) — high importance, exactly our stack

**Architecture: pull-based, not push.** Prometheus scrapes your `/metrics`
endpoint on a schedule rather than the app pushing data to it — the FastAPI app
only needs to expose one `/metrics` endpoint; it doesn't need to know Prometheus
exists.

**Storage:** purpose-built TSDB, each series uniquely identified by metric name +
label set, e.g. `http_requests_total{method="POST", handler="/predict",
status="200"}`.

**The four metric types:**

| Type | Behavior | Use for our project |
|---|---|---|
| **Counter** | Only increases; `rate()` derives a per-second rate | Total request count, error count |
| **Gauge** | Instantaneous value, up or down | In-flight requests, memory use |
| **Histogram** | Buckets observations; enables percentiles | **Latency** — the right type, not an average |
| **Summary** | Client-side quantiles; can't aggregate across instances | Generally skip — Histogram preferred |

**Instrumenting Python (`prometheus_client`):** decorator (`@REQUEST_TIME.time()`),
explicit calls (`counter.inc()` / `histogram.observe()`), or `Info`/`Enum` for
static metadata (model version). This is the library the challenge requires — the
"request count, latency, error rate" panels map to a Counter for requests, a
Counter (or error-labeled Histogram series) for errors, and a Histogram for
latency.

**PromQL essentials:** `rate()` for counters, `histogram_quantile()` for
P95/P99 from a Histogram, composing derived metrics like
`error_rate = rate(errors_total[5m]) / rate(requests_total[5m])`.

**Industry weight:** Prometheus is CNCF's 2nd graduated project after Kubernetes,
~75%+ adoption among container-using orgs — the single most standard combo for
exactly what the challenge wants.

## Grafana dashboard design (Aula 03) — high importance

**Design principles for the ≥3-panel requirement:**
- **Progressive disclosure**: top level = aggregate status, drill-down =
  per-endpoint detail.
- **Tufte's principles** (data-ink ratio, avoid chartjunk) and the
  **"glanceability" rule** — useful info conveyed in under 5 seconds.

**Suggested dashboard architecture** (template for our deliverable):
- Row 1 — Overview: stat panels for total requests, avg latency, error rate
  (exactly the challenge's minimum).
- Row 2 — Infrastructure: CPU/memory time series (optional extra).
- Row 3 — Model-specific: prediction distribution (nice-to-have differentiator).

**Panel types:** Heatmap (latency distribution over time), time series with
threshold annotations (mark deploys), stat panels with sparklines. Cited stat:
Grafana Cloud processes 1.5 trillion metrics/day; notable users include
Bloomberg, JPMorgan, Tesla.

## Aulas 04–08 — scaling beyond our scope (context only)

None of this applies to a docker-compose-only project, but worth knowing exists:

- **Aula 04**: Kubernetes-native Prometheus (Operator pattern, `ServiceMonitor`/
  `PodMonitor` CRDs, HPA driven by custom metrics like queue depth). Not needed —
  no Kubernetes deployment here.
- **Aula 05**: Azure Monitor + OpenTelemetry auto-instrumentation, KQL. Azure-
  specific, irrelevant (we picked AWS).
- **Aula 06** (title/content mismatch: header says "Kubernetes," content is
  actually 100% **AWS CloudWatch** — flagging so it isn't mistaken for a missing
  lecture): custom metrics via boto3, CloudWatch Logs Insights, pricing
  ($0.30/custom-metric/month). Worth one README sentence: in a real AWS
  deployment, CloudWatch would be the natively-integrated alternative to
  self-hosted Prometheus/Grafana — but self-hosted is what the challenge
  explicitly requires, and CloudWatch's pay-per-metric pricing is a good contrast
  point for "why not just CloudWatch."
- **Aula 07**: Thanos/Cortex for multi-cluster/long-term Prometheus storage —
  irrelevant at our single-service scale.
- **Aula 08**: **OpenTelemetry** unifies metrics/traces/logs instrumentation into
  one standard. Traces show hierarchical structure a Prometheus histogram can't
  (e.g., HTTP handler → TF-IDF vectorization → model.predict() → response, each
  with its own latency) — good one-liner for the README's "future work" section.
  **AIOps** (ML applied to ops itself) is pure context, not implementable here.
