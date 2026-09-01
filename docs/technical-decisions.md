# Technical Decisions

Record of the decisions made for the Tech Challenge Fase 3 project and the
reasoning behind each one — especially where we chose a simpler version than
"the ideal" production setup, and why. Complements `architecture.md` (which
describes the *what*; this document focuses on the *why*).

## Dataset and urgency labeling

**Choice:** Medical Abstracts TC Corpus (Kaggle), 14,438 labeled reports across
5 disease categories (neoplasms, cardiovascular diseases, nervous system
diseases, digestive diseases, general pathological conditions).

**The problem:** the dataset has no real urgency labels (normal/attention/
urgent) — only disease categories. Training directly against synthetic urgency
labels (invented by us) would be scientifically fragile: evaluation metrics
would measure how well the model learned our own heuristic, not real urgency.

**Decision:** train the classifier on the 5 real categories (genuine ground
truth) and apply a deterministic category→urgency mapping layer on top of the
prediction, adjusted by keywords present in the report's own text. This keeps
model evaluation honest (metrics against real labels) and documents the
urgency logic as an explicit, auditable business rule rather than something
"learned" opaquely.

**Trade-off accepted:** final triage quality depends on both the category
classifier's accuracy and the mapping rule's quality — an error in either part
can produce an incorrect urgency. That's why the confusion matrix is treated
as a required artifact in every experiment: a cardiovascular report
misclassified as "general pathological condition," for example, would
silently fall through to "normal" downstream — this is the system's most
dangerous failure mode, and it's what the confusion matrix exists to catch.

## Re-splitting the dataset and dropping ambiguous labels

**The problem:** Kaggle ships this corpus pre-split into train (11,550 rows)
and test (2,888 rows). Combining them to build our own split (rather than
using the shipped one — see split rationale below) surfaced two data-quality
issues invisible from either file alone:

1. **988 abstracts appear in both the original train and test files.** A
   model trained on the shipped train split and evaluated on the shipped test
   split could have been partly scoring on memorized examples — the official
   split has train/test leakage built in.
2. **2,929 abstracts appear more than once with *conflicting* category
   labels** — same document text, different `condition_label`. Critically,
   *zero* duplicate groups repeat with a matching label: every single
   duplicate is a genuine conflict. Inspecting examples confirmed why:
   "general pathological conditions" is involved in the large majority of
   conflicting pairs (e.g. `cardiovascular diseases` + `general pathological
   conditions`, 738 pairs), consistent with the source corpus having
   originally multi-labeled some documents (a case report can plausibly span
   a specific disease system *and* the generic bucket), which this
   single-label Kaggle release exploded into separate rows — one per label —
   rather than preserving as multi-label.

**Why this matters beyond data hygiene:** it collides directly with the
urgency-mapping design. "General pathological conditions" maps to baseline
**normal**; several of its most common conflict partners
(`cardiovascular diseases`) map to **urgent**. Whichever label got kept for
an ambiguous document would silently decide its urgency tier — arbitrarily.

**Decision:** combine train+test, then **drop all 2,929 ambiguous documents
entirely** rather than keeping one label per document. Considered and
rejected: (a) keeping the first-occurring row — simplest, preserves the full
11,227-document count, but the kept label is incidental to Kaggle's row
order, effectively injecting label noise into ~26% of the corpus with no
principled justification; (b) a safety-biased tiebreak resolving conflicts
toward whichever label maps to the higher urgency tier — ties the cleanup
decision to the triage-safety narrative used elsewhere in this document, but
adds a bespoke rule that's harder to justify as *data cleaning* rather than
*model behavior in disguise*. Dropping ambiguous documents outright keeps
every remaining label unambiguous ground truth, is the easiest of the three
to explain and defend, and still leaves 8,298 documents — comfortably above
the challenge's 2,000-sample floor — with class balance essentially
unchanged (~3.4x vs. the original ~3.2x). The held-out test set is then
carved from this clean pool with a fixed `random_state`, per the
train/validation/test split decision below.

## Candidate models

**Choice:** Logistic Regression, LinearSVC, Multinomial Naive Bayes, Complement
Naive Bayes, and Random Forest, all with `class_weight="balanced"` where
supported.

**Why not just Random Forest** (the challenge PDF's literal suggestion): for
high-dimensional, sparse TF-IDF text, linear models tend to perform better and
more predictably than tree ensembles — and export more cleanly to ONNX. Random
Forest still enters the comparison though: testing it empirically and showing
why (or whether) it loses to the linear models is a stronger narrative than
dismissing it without evidence.

**Why not embeddings (Word2Vec/BERT/ClinicalBERT):** the challenge itself asks
for a "lightweight NLP model" — embeddings/transformers would work against
that requirement and against the latency-optimization story we built around a
small linear model. Considered and deliberately rejected, not overlooked.

## Model selection: simplified feature engineering first

**Decision:** rather than searching for the optimal TF-IDF configuration per
model (a full, expensive search across all 5 candidates), we test only 2
representative configurations ("conservative" and "rich") across all 5 models
in the model-selection stage. The full feature search (36 combinations) runs
afterward, only on the winning model.

**Why this is acceptable:** it's a pragmatic simplification of an already
standard industry practice — "model bake-off" / "spot-checking algorithms":
screen candidates cheaply first, invest heavy tuning effort only in the
winner. The fully automated version of this (AutoML, joint Bayesian search
over model+features+hyperparameters) would need more infrastructure than two
weeks allows, and — more importantly — a joint automated search produces a
far less explainable story for the README/video than a pipeline with clear,
staged decisions. We chose interpretability over marginal additional rigor
here, deliberately, not as a hidden limitation.

## Evaluation metrics

**Decision metric: F1-macro.** Unweighted mean across the 5 classes — a model
can't win purely by being good at the majority class ("general pathological
condition," 4,805 samples).

**Why not accuracy alone:** with moderate class imbalance (~3.2x between the
largest and smallest class), accuracy can be misleading — reported for
context, never as the deciding criterion.

**Why cardiovascular recall is checked separately:** the cardiovascular
category is our baseline for "urgent." A false negative here (a
cardiovascular report misclassified) carries a higher clinical cost than a
false positive. So after picking the F1-macro winner, we explicitly confirm
cardiovascular recall wasn't sacrificed — a deliberate two-step check, more
transparent than trying to bake that clinical weighting into a single
automatic composite metric.

**Why not ROC-AUC/PR-AUC/MCC:** for a multi-class problem, ROC-AUC requires
extra decisions (One-vs-Rest, macro/weighted averaging) and `predict_proba`,
which LinearSVC doesn't have natively, requiring extra `CalibratedClassifierCV`
just for that one candidate. F1-macro + per-class recall + confusion matrix
already cover the same ground with less added complexity. Considered and
consciously dropped, not overlooked — discussed and weighed explicitly before
the final call.

## Train/validation/test split and data leakage

**Decision:** a test set (~15–20%) is carved out once at the start, never
touched during model-selection, feature-engineering, or hyperparameter-tuning.
All experimentation stages use Stratified K-Fold over the remaining data. The
final pipeline (winning model + feature config + hyperparameters) is retrained
on the full train+validation pool and evaluated exactly once on the test set —
that's the number reported in the README.

**Why this matters:** repeatedly evaluating against the same test set at every
decision stage, even without directly training on it, creates indirect
overfitting to that test set. Setting aside a single test set and using it
only once, at the end, avoids that trap.

**Use of `sklearn.pipeline.Pipeline`:** the TF-IDF vectorizer and classifier
are bundled into a single `Pipeline` object. This isn't just deployment
convenience — it structurally prevents data leakage: when the `Pipeline` is
passed to `cross_validate`/`GridSearchCV`, the vectorizer is refit only on
each fold's training data, never on validation data. The same trained
`Pipeline` object is saved, loaded by the FastAPI service, and exported to
ONNX — one artifact, identical behavior everywhere, no risk of
training-serving skew.

## Latency optimization (Etapa 4)

**Decision:** the technique applied depends on which model wins selection:
- Linear model winner: ONNX export + dynamic INT8 quantization.
- Random Forest winner: ONNX export + cost-complexity pruning (`ccp_alpha`)
  and/or reduced `n_estimators`.

**Why not force quantization either way:** quantization reduces the numeric
precision of dense weight matrices (matrix multiplication) — it doesn't
meaningfully apply to tree ensembles, which have no such structure. Forcing
this technique onto a Random Forest would be a category error. Tree pruning
(`ccp_alpha`), on the other hand, is literally where the term "pruning"
historically comes from (predating its use in neural networks) — the more
appropriate technique here, not a lesser alternative.

**Metric reported:** P50/P95/P99 over the whole `/predict` pipeline
(preprocessing + inference + response), not just `model.predict()` —
preprocessing can be 40–60% of total latency in an unoptimized system.

## Tooling: uv, DVC, MLflow

**uv:** replaces `requirements.txt`/Poetry, no meaningful adoption cost.

**DVC + S3 (not just local):** even with a static dataset (unchanged
throughout the project), we chose to host it on S3 via real DVC rather than
locally only — this reflects how it's actually done in industry and avoids
treating the practice as decorative.

**MLflow, 4 separate experiments, from the start:** `model-selection`,
`feature-engineering`, `hyperparameter-tuning`, `latency-optimization`. Local
tracking, no dedicated server — viewed via `mlflow ui` in the browser.
Separating these stages into distinct experiments (common industry practice)
keeps each stage's search space clean and comparable.

**SQLite backend instead of the plain filesystem store:** originally planned
as file-based tracking (`mlruns/` only, no database). In practice, the
installed MLflow version (3.15.2) has put the plain filesystem backend into
maintenance mode — `mlflow ui` refuses to start against `./mlruns` at all,
raising `MlflowException` and pointing at a database backend instead (it's
possible to force the old behavior via `MLFLOW_ALLOW_FILE_STORE=true`, but
running the graded deliverable against a backend MLflow itself says "will
not receive further updates" felt like the wrong tradeoff for a project
meant to reflect current practice). Switched to the MLflow-recommended local
SQLite backend (`mlflow.db`) for run/experiment metadata instead —
`training/mlflow_config.py` centralizes the tracking URI so every
experiment script points at the same store. Artifacts (confusion matrices,
classification reports) still land under `mlruns/` regardless of backend —
only the metadata store moved. Still fully local, still no tracking-server
infra to run or maintain — the substance of the original plan is unchanged,
this is a version-driven correction, not a design change.

## CI/CD and AWS

**Real push to ECR via GitHub Actions:** authentication via OIDC (federation
with `token.actions.githubusercontent.com`), no static AWS keys stored as
GitHub secrets — standard industry security practice.

**App Runner, not EC2/Lambda/Batch/SageMaker, for real-time inference:** the
scenario requires an immediate response (hospital triage), which rules out
Lambda's per-invocation cold start — the model needs to stay loaded in memory
continuously. The challenge only requires this decision to be written up in
the README (not deployed), but we chose to actually deploy it: App Runner
gives the same always-warm container behavior as EC2 without EC2's
operational overhead — no instance to provision, no SSH, no security groups,
no OS patching. Point it at a tagged image in ECR and it runs. Full
justification (App Runner vs. Lambda vs. Batch vs. SageMaker vs. raw EC2)
still goes in the README, since that section is graded regardless of whether
deployment is real.

**Why deploy for real when the challenge doesn't require it:** a live,
demoable endpoint is a stronger STAR-video "Result" than a localhost screen
recording, and App Runner's low operational cost (nothing to manage or patch)
makes the extra realism cheap enough to justify. Torn down after the
grading/demo window — no need to keep it running afterward.

## Airflow in standalone mode

**Decision:** `airflow standalone` (SQLite backend), not the official
production docker-compose (Postgres + Redis + webserver + scheduler + worker).

**Why:** the challenge asks for a "simple" DAG simulating train/retrain — the
official production compose is disproportionate for a 3-task demo. The DAG is
triggered manually (there's no real stream of new data feeding this project),
demonstrating orchestration capability rather than an actual recurring
retraining need in this specific context.

## Streamlit frontend (extra, not graded)

Separate, simple application calling the API's `/predict` endpoint over HTTP
(no duplicated model logic) — added purely to make the video demo more visual
than showing Swagger docs or a curl command. Explicitly marked as not
required.

## Out of scope (deliberately)

Kubernetes/HPA/KEDA, Canary/Shadow deployment, implemented drift detection
(PSI/KS — mentioned in the README as future work, not built), word/transformer
embeddings, joint automated search (AutoML) over model+features+
hyperparameters, ROC-AUC/PR-AUC/MCC, any real production traffic. All
considered and consciously dropped for the reasons above — not from lack of
awareness of what exists, but because they aren't justified within this
challenge's scope and timeline.
