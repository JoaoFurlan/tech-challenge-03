# Model Card — Hospital Report Triage Classifier

Following the structure proposed in Mitchell et al., *Model Cards for Model
Reporting* (2019). Not required by the Tech Challenge — added because
documenting a model this way is standard professional practice for anything
touching clinical decisions, even at demo/academic scale. Quantitative
sections are placeholders until the training pipeline runs; see
`architecture.md` and `technical-decisions.md` for the full design rationale.

## Model Details

- **Developed by:** solo project, Tech Challenge Fase 3 (POS Tech / MLET).
- **Model date:** 2026-09.
- **Model type:** text classifier — TF-IDF vectorizer (unigram,
  `max_features=10000`, `min_df=2`, `stop_words="english"`) + **Complement
  Naive Bayes** (`alpha=0.5, norm=True`), chosen via the `model-selection`
  MLflow experiment over Logistic Regression, LinearSVC, Multinomial NB,
  and Random Forest — *not* the raw F1-macro leader (LinearSVC), chosen
  instead for a substantially lower dangerous-miss rate, an explicit
  documented safety-for-accuracy trade. Full reasoning in
  `technical-decisions.md`.
- **What it predicts directly:** one of 5 disease categories (neoplasms,
  cardiovascular diseases, nervous system diseases, digestive diseases,
  general pathological conditions) from a medical report's free text.
- **What it outputs after post-processing:** an urgency tier —
  normal / attention / urgent — derived deterministically from the predicted
  category plus a keyword adjustment over the report text (see
  `architecture.md` § Urgency mapping). The urgency tier is **not** learned;
  only the category prediction is a machine-learned output. As of the
  domain-shift fixes below, the response also includes `low_confidence:
  bool`, signaling when the category prediction has no real vocabulary
  evidence behind it.
- **Optimization:** exported to ONNX (FP32) — chosen over the sklearn
  pipeline for a 4.4x P50 latency win at zero accuracy cost. INT8
  quantization was tested and rejected: counterintuitively *slower* than
  FP32 at this model's scale (dequantization overhead exceeds the compute
  savings for a model this small) — a documented negative finding, not
  assumed to help because the plan called for it. See
  `technical-decisions.md` § Latency optimization.
- **License / paper:** none published for this project; underlying dataset is
  the Medical Abstracts TC Corpus (Kaggle /
  `sebischair/Medical-Abstracts-TC-Corpus`).

## Intended Use

- **Primary intended use:** academic demonstration of an ML deployment
  pipeline (CI/CD, orchestration, monitoring, latency optimization) for a
  postgraduate tech challenge. The "hospital triage" framing is the exercise's
  scenario, not a validated clinical deployment.
- **Primary intended users:** the project author and course graders; secondary
  audience is anyone reviewing the repo as a portfolio/reference piece.
- **Out-of-scope uses:** **this model must not be used for real clinical
  triage or any real patient-facing decision.** It is trained on a public
  research abstract dataset, not on real hospital laudo intake text, has not
  been clinically validated, and the urgency-mapping layer is a documented
  heuristic (category baseline + keyword rule), not a clinically-derived
  scoring system.

## Factors

- **Relevant factors:** report length and vocabulary style (the dataset is
  medical *abstracts* — condensed, technical academic writing — which may
  differ systematically from how a real hospital laudo is phrased).
- **Evaluation factors:** performance is evaluated per disease category
  (5-way), not per urgency tier, since urgency has no ground truth in this
  dataset — see § Metrics.

## Metrics

- **Primary model-selection metric: F1-macro** across the 5 disease
  categories — chosen specifically so a model can't win by only being good at
  the largest category. See `technical-decisions.md` § Evaluation metrics for
  the full reasoning.
- **Reported alongside every model:** per-class precision/recall, confusion
  matrix, accuracy (context only, not decisive).
- **Clinically-motivated secondary check:** recall on the cardiovascular
  category specifically (our "urgent" baseline) is confirmed explicitly after
  the F1-macro winner is chosen — a false negative there is the costliest
  failure mode, since it would silently fall through to a lower urgency tier
  downstream.
- **Latency metrics** (separate from classification quality): P50/P95/P99
  response time over the full `/predict` pipeline, original model vs.
  optimized (ONNX + quantization/pruning).
- **Metrics considered and not used:** ROC-AUC, PR-AUC, Matthews Correlation
  Coefficient — see `technical-decisions.md` for why.

## Training Data

- **Source:** Medical Abstracts TC Corpus (Kaggle), 14,438 labeled records.
- **Class distribution:**

| Category | Count |
|---|---|
| General pathological conditions | 4,805 |
| Neoplasms | 3,163 |
| Cardiovascular diseases | 3,051 |
| Nervous system diseases | 1,925 |
| Digestive system diseases | 1,494 |

- **Preprocessing:** TF-IDF vectorization (configuration chosen via the
  `feature-engineering` MLflow experiment); no external embeddings.
- **Split:** ~80–85% used for cross-validated model-selection,
  feature-engineering, and hyperparameter-tuning (Stratified K-Fold); ~15–20%
  held out as a test set, untouched until final evaluation. Details in
  `technical-decisions.md` § Train/validation/test split.

## Evaluation Data

Same source distribution as training data — the held-out test set described
above, drawn from the same Medical Abstracts TC Corpus via a single
stratified split. No separate out-of-distribution evaluation set is used;
this is a known limitation (see § Caveats).

## Quantitative Analyses

Final pipeline (ComplementNB `alpha=0.5, norm=True` + TF-IDF as above),
evaluated once on the 1,245-document held-out test set — see
`architecture.md` and `technical-decisions.md` for the full staged-experiment
process (model-selection → feature-engineering → hyperparameter-tuning) that
led here, including every rejected alternative and why.

| Metric | Value |
|---|---|
| F1-macro | 0.7786 |
| Accuracy | 0.7880 |
| Tier accuracy (urgency, not just category) | 0.8056 |
| Undertriage rate (predicted tier below true tier — dangerous) | 0.0498 |
| Overtriage rate (predicted tier above true tier — costly, not dangerous) | 0.1446 |

Per-class (test set):

| Category | Recall | Precision |
|---|---|---|
| Cardiovascular diseases | 0.939 | 0.758 |
| Neoplasms | 0.927 | 0.831 |
| Digestive system diseases | 0.781 | 0.788 |
| Nervous system diseases | 0.709 | 0.747 |
| General pathological conditions | 0.574 | 0.792 |

Cardiovascular recall (0.939) was the explicit clinically-motivated check
from § Metrics — confirmed high, consistent with the safety-first model
choice. General pathological conditions has the lowest recall (0.574) but
the highest precision among the weaker classes (0.792) — consistent with
ComplementNB's documented tilt away from confidently predicting the
"normal"-mapped category unless the evidence is strong, the same mechanism
that drove the model-selection choice.

**Latency** (ONNX FP32 vs. sklearn baseline, single-document `/predict`,
500-request benchmark):

| Variant | P50 | P95 | P99 | Size |
|---|---|---|---|---|
| sklearn baseline | 0.595ms | 0.814ms | 1.040ms | 1,233KB |
| ONNX FP32 (served) | 0.135ms | 0.263ms | 0.339ms | 413KB |
| ONNX INT8 (tested, not served) | 0.163ms | 0.281ms | 0.397ms | 267KB |

INT8 was slower than FP32, not faster — see `technical-decisions.md` § Latency
optimization for why, and § Model Details above.

## Ethical Considerations

- **Not a validated medical device.** No clinical, regulatory, or
  human-subjects review has been performed. Explicitly not for real triage
  use — see § Intended Use.
- **Urgency labels are a documented heuristic, not ground truth.** The
  category→urgency baseline table and the escalate/de-escalate keyword lists
  (`architecture.md` § Urgency mapping) were designed by the project author
  based on general clinical reasoning, not derived from labeled urgency data
  or reviewed by a clinician. They should be read as a transparent, auditable
  business rule — intentionally simple and inspectable — not as a validated
  triage protocol.
- **Failure mode of highest concern:** a report from a genuinely urgent
  category (cardiovascular) misclassified into a lower-urgency category would
  silently under-triage. This is why confusion-matrix review and
  cardiovascular recall are treated as required checks, not optional
  diagnostics, throughout the modeling pipeline.
- **Data provenance:** the training data is published medical *abstracts*
  (research/academic text), not real patient intake records — no PII/PHI is
  involved, but this also means the text style may not transfer cleanly to
  real hospital laudo language (see § Caveats).

## Caveats and Recommendations

- **Domain shift risk — confirmed via real-world testing, not just
  theoretical.** Medical abstracts (condensed, third-person, academic
  register) read differently than real triage phrasing (short, informal,
  clinical-shorthand). Post-deployment manual testing confirmed this
  concretely: "Unresponsive, no detectable pulse, non-breathing" — a
  textbook cardiac-arrest description — was classified `normal`. The
  words are individually in the training vocabulary, but the model never
  learned to associate this *register* with urgency, because it was never
  shown text written that way. **This is not fixable by adding more of
  the existing training data** — the Medical Abstracts TC Corpus is the
  only data source available, and more abstracts would only improve
  performance on abstract-style text, not teach the model a register it's
  never seen. A real fix needs training examples in that different
  register (real or realistically synthesized triage notes), which is a
  genuine scope increase, not a quick follow-up. Full investigation,
  including three other concrete misclassification examples and the two
  bugs it did surface and get fixed (a keyword-adjustment cap, and a
  missing low-confidence signal for near-empty-vocabulary inputs), in
  `technical-decisions.md` § Real-world testing surfaced a genuine
  domain-shift limitation.
- **Low-confidence inputs are now flagged, not silently trusted.** As of
  the fix above, `/predict` returns `low_confidence: true` when the input
  shares no vocabulary with the training data at all (e.g. very short or
  colloquial text) — the category prediction is then driven by the
  classifier's structural bias, not real evidence. Urgency is **fixed to
  `attention`** in that case (not merely floored) — a raw `urgent` guess
  is exactly as ungrounded as `normal` when there's no real evidence
  behind it, so neither extreme is trusted; a `message` field is also
  returned, guiding the caller to provide more detail. This narrows one
  failure mode (zero-signal inputs) but does not address the broader
  register-mismatch risk above (in-vocabulary text in an unfamiliar
  register still gets a confident-looking, potentially wrong answer).
- **No drift monitoring implemented.** In a real deployment, input
  distribution drift and concept drift (see `course-notes/monitoring-
  services.md`) would need active monitoring; this project's monitoring
  stack (Prometheus/Grafana) covers operational metrics (request count,
  latency, error rate) only, not model-quality drift.
- **Single held-out test set.** Results reflect one stratified split; no
  repeated/bootstrapped test-set estimates are computed, so reported test
  metrics carry some sampling variance not captured by a single point
  estimate.
- **Recommendation for any future real-world adaptation:** replace the
  training data with real (de-identified) hospital laudo text, have the
  urgency-mapping rule reviewed and validated by a clinician rather than
  relying on the author's heuristic, and add the drift-monitoring and
  human-in-the-loop review practices referenced in `course-notes/training-
  pipeline.md` before considering any real clinical use.
