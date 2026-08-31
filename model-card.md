# Model Card — Hospital Report Triage Classifier

Following the structure proposed in Mitchell et al., *Model Cards for Model
Reporting* (2019). Not required by the Tech Challenge — added because
documenting a model this way is standard professional practice for anything
touching clinical decisions, even at demo/academic scale. Quantitative
sections are placeholders until the training pipeline runs; see
`architecture.md` and `technical-decisions.md` for the full design rationale.

## Model Details

- **Developed by:** solo project, Tech Challenge Fase 3 (POS Tech / MLET).
- **Model date:** TBD (filled in once training completes).
- **Model type:** text classifier — TF-IDF vectorizer + a linear or tree-based
  scikit-learn classifier (final algorithm chosen empirically via the
  `model-selection` MLflow experiment; candidates: Logistic Regression,
  LinearSVC, Multinomial Naive Bayes, Complement Naive Bayes, Random Forest).
- **What it predicts directly:** one of 5 disease categories (neoplasms,
  cardiovascular diseases, nervous system diseases, digestive diseases,
  general pathological conditions) from a medical report's free text.
- **What it outputs after post-processing:** an urgency tier —
  normal / attention / urgent — derived deterministically from the predicted
  category plus a keyword adjustment over the report text (see
  `architecture.md` § Urgency mapping). The urgency tier is **not** learned;
  only the category prediction is a machine-learned output.
- **Optimization:** exported to ONNX, with quantization (linear model) or
  cost-complexity pruning (tree model) applied as a latency-optimization step
  — see `technical-decisions.md` § Latency optimization.
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

## Quantitative Analyses — TBD

To be filled in after `training/model_selection.py`,
`training/feature_engineering.py`, `training/hyperparameter_tuning.py`, and
`training/train_final.py` run:

- [ ] Winning model + configuration
- [ ] Final F1-macro on the held-out test set
- [ ] Per-class precision/recall/F1 on the held-out test set
- [ ] Confusion matrix (test set)
- [ ] Cardiovascular-class recall (test set)
- [ ] Original vs. optimized (ONNX) latency — P50/P95/P99, and model size

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

- **Domain shift risk:** medical abstracts (condensed, third-person, academic
  register) likely read differently than a real hospital laudo (often
  first-person clinical notes, abbreviations, structured fields). A model
  performing well on this dataset should not be assumed to generalize to real
  hospital intake text without further validation on in-domain data.
- **No drift monitoring implemented.** In a real deployment, input
  distribution drift and concept drift (see `course-notes/servicos-de-
  monitoracao.md`) would need active monitoring; this project's monitoring
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
  human-in-the-loop review practices referenced in `course-notes/pipeline-
  de-treino.md` before considering any real clinical use.
