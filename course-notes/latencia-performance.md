# Latência e Performance em Modelos de Dados Não Estruturados (Etapa 4)

Course folder: `Latência e Performance em Modelos de Dados Não Estruturados/`
(Aulas 01–08). Genuinely graduate-level "ML Systems Engineering" content — written
for someone optimizing deep neural networks / Transformers / LLMs on GPU clusters,
not a TF-IDF + linear classifier on CPU. Only two of the eight lectures transfer
meaningfully.

## Relevance triage

| Aula | Topic | Relevance |
|---|---|---|
| 01 | Fundamentals of latency/throughput | **High** — universal, directly usable |
| 02 | NLP/audio performance (attention, KV-cache, Whisper) | None — Transformer/audio-specific |
| 03 | Computer vision performance | None — different modality |
| 04 | Pruning & Quantization | **Medium-high, with a caveat** |
| 05 | Transfer learning (LoRA/QLoRA) | None — for billion-parameter LLMs |
| 06 | GPU/TPU hardware acceleration | None — our model never touches a GPU |
| 07 | Distributed inference (LLM serving at scale) | None — multi-GPU LLM serving |
| 08 | Multimodal orchestration at scale | None — Kubernetes-scale serving |

## Aula 01 — Fundamentals of Latency and Performance (directly useful)

- **Latency vs. throughput**, formalized by **Little's Law**: `L = λ × W`
  (average requests in system = arrival rate × average time in system) — useful
  for our API's capacity planning.
- **Percentiles over averages** — P50/P95/P99 characterize real user experience
  (reinforces Monitoração de Performance). Real industry standard, reads as more
  rigorous in the README/video.
- **Amdahl's Law**: max speedup from optimizing one part is capped by the
  fraction of the system that stays unaffected. Even a perfectly ONNX-optimized
  model won't reduce total latency much if most time is spent in non-model code
  (JSON parsing, TF-IDF vectorization, network overhead) — measure the *whole*
  `/predict` pipeline for the before/after comparison, not just `model.predict()`.
- **Queueing theory (M/M/1)**: latency diverges as utilization approaches 100%;
  keep utilization 60-80% for stable latency — background vocabulary only.
- Pre/post-processing and serialization can be 40-60% of total latency in
  unoptimized systems.

## Aula 04 — Pruning & Quantization (the important caveat, confirmed across three lectures now)

Both techniques framed entirely in a deep-learning context:
- **Pruning** — removing individual weights or whole structures from a neural
  network (Lottery Ticket Hypothesis: a dense randomly-initialized network
  contains a sparse "winning ticket" subnetwork that trains to equal accuracy).
  Structured pruning (whole filters/heads) gets real speedups on standard
  hardware; unstructured pruning needs specialized sparse-matrix hardware.
- **Quantization** — FP32 → INT8/INT4 via calibration. Combined pipelines claim
  10-50x compression in the neural-net examples given.

**Confirmed final recommendation** (this is now consistent across
`course-notes/monitoracao-performance.md` Aula 3 and this folder's Aula 4 — our
model candidate is a linear model, not a neural network, which changes the
practical story):

1. **ONNX export + dynamic INT8 quantization is the technique that cleanly
   transfers.** `skl2onnx` converts the fitted model to ONNX; ONNX Runtime
   supports post-training dynamic quantization of the linear weight matrix to
   INT8. Real, supported, genuine measurable improvement.
2. **"Pruning" has no clean equivalent** for a linear model unless drawing an
   explicit parallel: **L1-regularized Logistic Regression** (or `LinearSVC`
   with L1 penalty) already produces a sparse coefficient vector by zeroing
   uninformative TF-IDF features during training. Legitimate to mention as a
   conceptual parallel ("we applied L1 regularization as a pruning-equivalent
   sparsification step, then ONNX + INT8 quantization for the inference-latency
   optimization"), but not the mechanism the course teaches (no lottery ticket,
   no iterative subnetwork retraining) — shouldn't be presented as literal
   "pruning."

**Updates the framing of "ONNX vs. quantization vs. pruning" from CLAUDE.md's
open decisions list**: not three independent equal options for us — quantization
and ONNX-export are the same optimization step (quantize the ONNX-exported
model), and pruning doesn't cleanly apply to a linear model at all.

## Aulas 02, 03, 05–08 — background only, not actionable

- **Aula 02 (NLP/Audio)**: self-attention complexity, Flash Attention, KV-Cache,
  audio pipelines (Whisper/wav2vec2). None applies — TF-IDF isn't a Transformer,
  no audio processing.
- **Aula 03 (Computer Vision)**: CNN/ViT trade-offs, YOLO, video pipelines. Zero
  overlap with a text classifier.
- **Aula 05 (Transfer Learning)**: LoRA/QLoRA for fine-tuning billion-parameter
  LLMs cheaply. We're training a small model from scratch on ~2000 samples.
- **Aula 06 (GPU/TPU Hardware)**: Tensor Cores, Roofline model, Mixed Precision
  Training — assumes GPU-bound deep learning; scikit-learn inference runs
  trivially on CPU.
- **Aula 07 (Distributed Inference)**: vLLM, tensor/pipeline parallelism for
  multi-GPU LLM serving — wildly out of scope for one lightweight model on one
  EC2 instance.
- **Aula 08 (Multimodal Orchestration)**: SRE concepts (SLIs/SLOs/error budgets
  — reusable vocabulary for the monitoring section), Kubernetes ML-serving
  frameworks (KServe/Seldon/BentoML), canary deployments. SLI/SLO framing worth
  a passing README mention; the Kubernetes-scale architecture is far beyond this
  project.
