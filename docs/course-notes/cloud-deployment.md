# Cloud Deployment (Etapa 1)

Course folder: `content/Deploy em Nuvem/` (Aulas 01–06). Discipline behind Etapa 1 of the
Tech Challenge: architecture decision + initial FastAPI app + Docker + baseline latency.

## 1. The three deploy patterns (Aula 01)

Deploy is a **systems decision**, not an afterthought to training. Three fundamental
patterns:

| | Batch | Real-time | Serverless |
|---|---|---|---|
| What it is | Model runs over stored data, scheduled/triggered | Always-on API, one request = one inference | Runs on-demand per event, provider manages scaling |
| Latency | High, tolerated | Low, deterministic | Variable (cold start) |
| Cost | Lower, no idle infra | Higher, infra always on | Efficient for spiky/intermittent load |
| Scaling | Planned around throughput | Planned around concurrency | Automatic, provider-limited |
| Ops complexity | Low | High | Medium (pushed to provider) |
| Failure tolerance | High — just re-run the job | Low — felt immediately by users | Medium — retries usually automatic |
| Model updates | Easy — swap artifact before next run | Needs care: endpoint versioning, gradual rollout | Easy — functions stateless/ephemeral |

**Key idea:** the right question isn't "which is best" but "what does this specific
component need?" Real systems are usually **hybrid** — batch for reports, real-time
for user-facing APIs, serverless for occasional event-driven tasks.

**Our project:** the hospital triage scenario is a **real-time** deploy — a laudo
comes in, an urgent/attention/normal answer is needed immediately. This is the
architecture argument for the README.

## 2. The model's own behavior drives the deploy pattern (Aula 02)

Algorithm choice isn't independent of infrastructure choice:

- **Linear models** (linear/logistic regression, linear SVM): cheap, fast, predictable
  inference — a few matrix multiplications on a small parameter set. Naturally suited
  to real-time APIs and serverless.
- **Distance-based/clustering models** (KMeans, etc.): need the whole dataset or
  pre-computed structures to produce a result — don't answer one request at a time
  well. Naturally suited to batch jobs.
- **Tree/probabilistic models** (Random Forest, etc.): heavier per-inference cost,
  less predictable latency than linear models.

Other decisive factors: **artifact size / load time** (small models load instantly —
good for serverless cold starts), **preprocessing cost** (must exactly match between
train and inference, or results silently break), and **external dependency needs**
(does inference need only input features, or also global state like cluster centroids?).

**Our project:** this is the reasoning behind leaning toward **TF-IDF + Logistic
Regression / Linear SVM** over Random Forest — cheaper, more predictable inference,
and cleaner ONNX export for the Etapa 4 latency-optimization story. *(Not finalized —
still an open decision as of this note.)*

## 3. Cloud vocabulary — AWS / Azure / GCP (Aulas 03, 04, 05)

Same three patterns + a managed-ML layer, different service names per cloud:

| Concept | AWS | Azure | GCP |
|---|---|---|---|
| Container image registry | ECR | ACR | Artifact Registry |
| Real-time, full control (VM) | EC2 | Azure VM | Compute Engine |
| Serverless / event-driven | AWS Lambda | Container Apps | Cloud Run |
| Batch / scheduled jobs | AWS Batch | Container Apps Jobs | Cloud Run Jobs |
| Object storage | S3 | Blob Storage | Cloud Storage |
| Managed ML platform | SageMaker | Azure ML | Vertex AI |
| Access control | IAM | Managed Identities | IAM + Service Accounts |

**Common pattern across all three clouds:**
1. Containerize everything (model + code + deps in one Docker image) — same artifact
   can run as a VM service, serverless function, or batch job with no code changes.
2. Push the image to a registry (ECR/ACR/Artifact Registry) — versioned, access-controlled.
3. Pick an execution service matching the workload.
4. Object storage glues the pipeline together (datasets in, models/results out).
5. Identity-based access (IAM roles / managed identities / service accounts) instead
   of hardcoded credentials.
6. Managed ML platforms (SageMaker/Azure ML/Vertex AI) add model registry,
   versioning, managed endpoints, governance — more setup, less low-level control.

**EC2 vs ECR — not alternatives, different layers:** ECR only *stores and distributes*
the Docker image (a warehouse); EC2 is *compute that runs* the container (the store/
truck). The flow is: build image → push to ECR → EC2 (or Lambda/Batch/SageMaker)
pulls from ECR and runs it. ECR is a required stop on the way to any execution option,
not a competing choice to EC2.

**Why EC2 for this project:** matches the real-time pattern — full control, model
stays loaded in memory permanently, predictable low latency. Lambda has cold-start
(bad for consistent low latency in a clinical tool); AWS Batch is for scheduled/bulk
processing, wrong shape; SageMaker is a full managed ML platform, heavier setup than
needed. EC2 is what the course explicitly teaches for "always-on API," so it also
keeps README vocabulary aligned with course content.

**CI/CD tie-in:** GitHub Actions can build the Docker image and push it to ECR
automatically on every push (auth → build → tag → push), so there's no need to do
it locally. Best practice: authenticate via **OIDC federation** (short-lived,
scoped IAM role) rather than long-lived AWS access keys stored as GitHub secrets —
ties back to the "no hardcoded credentials" principle from Aula 06.

## 4. FinOps and Security (Aula 06)

Cost and security share the same foundation: **governance, visibility, control.** A
misconfiguration is very often *both* a security hole and a cost leak (e.g., no rate
limiting → abuse risk + unbounded compute cost).

**FinOps:**
- Not just cost-cutting — maximizing value per dollar, tracked continuously.
- ML cost is inherently variable: training spikes, inference runs continuously,
  experimentation burns money without guaranteed value.
- Practices: tag/label resources (project, environment, owner) for cost attribution;
  budgets and alerts; kill idle resources (unused VMs, zero-traffic endpoints);
  separate dev/staging/prod so experiments don't inflate production costs.
- Deploy pattern drives cost shape: real-time = constant cost, serverless = cost
  aligned to demand, batch = cost concentrated in discrete runs.

**Security, ML-specific:**
- Least-privilege access control.
- Data protection: encryption at rest/in transit, masking/anonymization (relevant
  for medical laudos).
- **Model security as its own category** — the trained model is an asset to protect:
  - **Model extraction attacks**: repeated queries used to reverse-engineer/clone
    the model.
  - **Data poisoning attacks**: malicious data injected during training corrupts
    model quality.
  - Mitigations: rate limiting, authentication, input validation.
- Continuous monitoring: logs/metrics catch both cost anomalies and access anomalies
  — often the same signal.
- Governance: environment separation, standardized tagging, automated policy
  enforcement, balanced log retention (audit needs vs. storage cost).

**Our project:** no live AWS deployment, but this feeds the README's architecture
section — e.g., noting that a production version would need API rate limiting
(dual-purpose: anti-abuse + cost control) and encryption for laudo text (patient
data). Also strengthens the real-time-over-batch argument: Aula 06 explicitly frames
real-time as constant-but-higher cost, an acceptable tradeoff for a clinical triage
tool where latency has real consequences.

## Open decisions still on the table

- Exact urgency label-mapping strategy for the dataset.
- Final classifier choice (leaning TF-IDF + Linear SVM/LogReg, not committed).
- ONNX vs. quantization vs. pruning for Etapa 4.
- Git/GitHub setup timing (init now vs. later in the "clean" repo).
