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

**Amendment: urgency-tier metrics added, and used as co-decisive.** Running
model-selection surfaced a structural problem with F1-macro for this
specific system: 3 of the 5 categories (neoplasms, nervous, digestive) map
to the *same* urgency tier ("attention"). F1-macro penalizes confusing those
three exactly as much as a genuinely dangerous error (e.g. cardiovascular →
general, a 2-tier drop to "normal"), even though the former has **zero**
effect on the actual triage output. `training/urgency.py` maps each
category to its baseline tier; `training/model_selection.py` now also logs
`tier_accuracy`, `undertriage_rate` (predicted tier < true tier — the
dangerous direction), and `overtriage_rate` (predicted tier > true tier —
costly but safe) for every run, evaluated against the pool via the same
Stratified K-Fold used for F1-macro. Per the cardiovascular-recall check
already established above, we'd already committed to not treating F1-macro
as the sole criterion — this generalizes that same principle across all
categories via the tier mapping instead of singling out one category.

**Model-selection result: ComplementNB (conservative TF-IDF) chosen over
the F1-macro leader.** LinearSVC/rich had the top F1-macro (0.798), but its
margin over LogisticRegression/rich (0.793) was smaller than either model's
fold-to-fold standard deviation (~0.009–0.013) — statistically a tie, not a
real difference. ComplementNB/conservative trailed on F1-macro (0.765, a
real ~3-point gap) but had a substantially lower `undertriage_rate` (0.053
vs. 0.089 for LinearSVC/rich — a ~40% relative reduction). Checked the
mechanism before trusting the number: ComplementNB's cardiovascular recall
(0.933) well exceeds its precision (0.774), and its general-pathological
precision (0.783) well exceeds its recall (0.562) — a consistent, genuine
directional tilt away from the "normal" tier when uncertain, not
indiscriminate over-prediction (precision stays reasonable across the
board). This is expected behavior for Complement Naive Bayes specifically:
unlike standard Multinomial NB (which estimates each class's word
probabilities from only that class's own data and is known to bias toward
majority classes on imbalanced datasets — visible here in its collapsed
digestive/nervous recall, 0.18–0.39), ComplementNB estimates each class's
parameters from every *other* class's data, which structurally counteracts
that imbalance bias — matching our moderately imbalanced dataset
(~3.4x). **Trade-off accepted, not hidden:** ComplementNB's total
tier-error rate is actually slightly higher than LinearSVC/rich's (19.8% vs.
17.4%) — it doesn't reduce mistakes overall, it redistributes them toward
the safe direction (`overtriage_rate` 0.145 vs. 0.085). For a hospital
triage system, more false alarms are an acceptable operational cost in
exchange for meaningfully fewer dangerous misses; this is a deliberate
safety-for-accuracy trade, documented as exactly that rather than presented
as a strictly better model.

## Feature-engineering result: unigram-only, confirmed not just assumed

**36-combination grid** (`ngram_range` x `max_features` x `min_df` x
`sublinear_tf`) with ComplementNB fixed as the model surfaced the same
F1-macro-vs-undertriage tension as model-selection, one level down: the
F1-macro-best config (`ngram_range=(1,2)`, `max_features=20000`,
`min_df=1`, `sublinear_tf=False`, F1=0.782) has `undertriage_rate=0.066`,
meaningfully worse than the unigram-only region (`ngram_range=(1,1)`,
undertriage clustered at 0.052-0.058 across the whole sub-grid — a
consistent ~20% relative gap, not a cherry-picked pair). Bigrams improve
the model's ability to positively identify "general pathological
conditions" (recall climbs from ~0.54-0.59 to ~0.60-0.64), which is
exactly the safety-favorable reluctance that made ComplementNB attractive
in model-selection — bigrams erode it. Kept unigram-only
(`ngram_range=(1,1)`) for the same safety-first reasoning already applied
to the model choice.

**Follow-up sweep** (`max_df` in {1.0, 0.7, 0.5} x `stop_words` in
{"english", None}, anchored on `max_features=10000, min_df=2,
sublinear_tf=False`): `max_df` had no measurable effect at any tested
value (F1-macro and undertriage_rate both flat to within 0.0001/0) — no
single unigram term is common enough across this corpus to matter once
English stop words are already removed, so `max_df` isn't a useful lever
here and wasn't added as a permanent config knob. `stop_words="english"`
beat `stop_words=None` consistently across every `max_df` value tested,
both on F1-macro and on undertriage_rate (0.052-0.053 vs. 0.055-0.055) —
digestive and nervous recall (2 of the 3 "attention"-tier categories) both
drop without stop-word removal, and that recall loss is what drives the
extra undertriage. This confirms a choice we'd already made by default
(`stop_words="english"` was fixed throughout model-selection) with actual
evidence, rather than leaving it untested.

**Final TF-IDF config**: `ngram_range=(1,1)`, `max_features=10000`,
`min_df=2`, `sublinear_tf=False`, `stop_words="english"` — F1-macro 0.773,
undertriage_rate 0.0525. Essentially unchanged from the original
"conservative" config used in model-selection (F1=0.765,
undertriage=0.0526); the grid search mostly *confirmed* that starting
point was already close to the safety frontier, while explaining why
(unigram-only is what matters, and the conservative config was already
there).

## Hyperparameter-tuning result and an MLflow autolog bug

**Tooling note:** planned as `GridSearchCV` + `mlflow.sklearn.autolog()`
(per `architecture.md`), but autolog's per-candidate child-run creation
throws internally on the installed MLflow version
(`'NoneType' object has no attribute '_to_mlflow_entity'`) — confirmed
only 1 of 24 expected runs was actually logged, despite all 24 candidates
being evaluated correctly under the hood (`cv_results_` was intact,
only MLflow visibility was broken). Switched to the same manual
per-combination logging already used in `model_selection.py` /
`feature_engineering.py`, verified to reproduce identical metrics before
discarding the `GridSearchCV` run. Same category of issue as the
filesystem-tracking-backend deprecation earlier — a version-driven
tooling correction, not a design change.

**Finding: `fit_prior` has zero effect on ComplementNB.** Every
`fit_prior=True`/`False` pair produced bit-for-bit identical metrics
across all 12 `alpha`/`norm` combinations. This isn't a bug in our
pipeline — sklearn's `ComplementNB` implementation doesn't incorporate
the class-prior term into its decision rule at all, per the original
paper's formulation (unlike `MultinomialNB`, where `fit_prior` does
matter). Confirmed empirically rather than assumed; not worth keeping as
a tuning dimension going forward.

**Result: `alpha=0.5, norm=True`** — F1-macro 0.769, `undertriage_rate`
0.048. The same F1-macro-vs-undertriage tension recurred at this third
level (after model choice and TF-IDF config): the F1-macro-best point
(`alpha=0.1, norm=False`, F1=0.777) has *worse* undertriage than our
prior baseline (0.056 vs 0.053) — reopening the exact tension we'd
already resolved in favor of safety at model-selection. `norm=True`
(ComplementNB's optional weight-renormalization step from the original
paper) is consistently what buys undertriage improvement across the
whole grid, at a real F1-macro cost. Of the three points on that
tradeoff frontier — max-F1 (`alpha=0.1, norm=False`), max-safety
(`alpha=0.05, norm=True`, undertriage 0.038 but F1 down to 0.751), and
this balanced middle — chose the middle: a meaningful undertriage
improvement over the pre-tuning baseline (~9% relative reduction) without
the steepest F1 cost of the max-safety point. Consistent with the
safety-first lean established at model-selection, without over-rotating
into it a second time.

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

**ComplementNB is neither branch, and was treated as the linear one.**
The winning model (see model-selection above) is Naive Bayes, not
anticipated by either branch above. Its decision rule — a dot product
against a dense per-class weight matrix (`feature_log_prob_`) — is
architecturally the same shape as a linear model's `coef_`, so it was
exported and quantized the same way as the linear branch. Random Forest's
pruning technique has no analogue here: pruning operates on tree
structure, and ComplementNB has none.

**Result** (500-request benchmark, single document at a time, `models/`):

| Variant | P50 | P95 | P99 | F1-macro | Size |
|---|---|---|---|---|---|
| sklearn baseline | 0.595ms | 0.814ms | 1.040ms | 0.7786 | 1,233KB |
| ONNX FP32 | **0.135ms** | **0.263ms** | **0.339ms** | 0.7786 (exact) | 413KB |
| ONNX INT8 (dynamic) | 0.163ms | 0.281ms | 0.397ms | 0.7803 | 267KB |

**Chosen: ONNX FP32 as the served artifact.** ONNX export alone is the
dominant win — 4.4x faster at P50, 3x smaller, and mathematically exact
(F1-macro unchanged, not approximated). INT8 quantization is
counterintuitively *slower* than FP32 here (0.163ms vs. 0.135ms P50), not
faster — at this scale, the whole model already runs in a fraction of a
millisecond, so the dequantization overhead added around each quantized
op outweighs the compute savings from smaller integer math. Quantization
only pays off on latency when compute time dominates over per-call
overhead, which isn't the case for a model this small. What INT8 does
deliver is a real size reduction (35% smaller than FP32) — a legitimate
choice if container/memory footprint is the priority, just not the
latency win it's usually reached for. This is reported as a genuine
negative finding on quantization, not glossed over as a win because the
plan called for it.

**Two tooling snags getting quantization to run at all**, both specific
to `skl2onnx`'s text-pipeline graph (`TfIdfVectorizer` + Naive Bayes ops
aren't the vision/NLP graphs onnxruntime's quantization tooling is
built/tested against):
1. `quant_pre_process`'s full symbolic shape inference throws
   (`"Incomplete symbolic shape inference"`) on this graph — worked
   around with `skip_symbolic_shape=True`, which still runs basic shape
   inference + model optimization.
2. `quantize_dynamic` itself then fails
   (`"Unable to find data type for weight_name='sum_result'"`) on an
   intermediate tensor from ComplementNB's decision-rule subgraph the
   quantizer's type inference can't resolve — worked around with
   `extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT}`. Needed
   adding `sympy` as an explicit dependency (required by the symbolic
   shape inference step, not declared as a transitive dependency by
   `onnxruntime` itself).

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

**Actually deployed on Elastic Beanstalk, not App Runner.** Discovered only
when attempting the real deployment: App Runner isn't part of AWS Free Tier.
Rather than quietly incur charges or silently rewrite the "why App Runner"
reasoning above (which is still sound — it's the *documented* choice, and
what the README's written justification is about), switched the actually
*deployed* service to Elastic Beanstalk running Docker on a single free-tier
EC2 instance: same always-warm, no-cold-start property App Runner offered
(Beanstalk keeps the instance running continuously, doesn't scale to zero
between requests), rides free-tier EC2 hours, and Beanstalk still absorbs
most of the manual EC2 ops burden (provisioning, health checks, deployment)
that App Runner would have avoided — closest free-tier-eligible match to
the original intent. Specifically the **single-instance** environment type,
not "load balanced, auto scaling" — that tier provisions an Elastic Load
Balancer, which is billed separately and isn't Free Tier eligible, which
would have defeated the entire point of switching. This is a real,
documented pivot forced by a budget constraint discovered late, not a
design change — same category as the MLflow filesystem-backend correction
and the `dvc`/`pygtrie` CI dependency fix earlier in this document.

**CI auto-deploys to Beanstalk on every push to main — and the IAM policy
for it ended up needing to be AWS-managed, not hand-scoped.** After the
first successful manual deployment, extended the `build-and-push` job to
rewrite `Dockerrun.aws.json`'s image tag, create a new Beanstalk
application version, and update the environment automatically — verified
by polling `describe-environments` afterward and failing the job if health
isn't `Green`, so a broken deploy is visible in CI rather than requiring a
manual console check.

Getting the IAM permissions right took several rounds, each surfacing a
genuinely new requirement rather than a mistake in the previous fix:
`elasticbeanstalk:UpdateEnvironment` itself, then `s3:CreateBucket` on
Beanstalk's own storage bucket (needed even though the bucket already
existed — IAM authorization happens before S3's idempotent "you already
own this" check), then `s3:PutBucketOwnershipControls` (likely reflecting
an S3 default-security change made after older example policies were
written). Research at that point turned up that a properly-scoped policy
for this actually needs wildcard-resource `autoscaling:*`/
`cloudformation:*`/`ec2:*` too, since Beanstalk provisions those services
under the hood via CloudFormation even for a single-instance environment —
at which point hand-scoping had lost its point. Switched to the AWS-managed
`AdministratorAccess-AWSElasticBeanstalk` policy (the current replacement
for the now-deprecated `AWSElasticBeanstalkFullAccess`) instead of
continuing to chase individual permissions one at a time. Broader than the
scoped-policy approach used for ECR/DVC's S3 access, but the pragmatic
choice given Beanstalk's own API surface is broad and evolving in ways a
hand-maintained policy can't keep pace with — an explicit, deliberate
trade-off, not the path of least resistance taken by default.

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

**Hosted on Streamlit Community Cloud, not the same AWS account as the
API.** Considered running it as a second Elastic Beanstalk environment
(consistent with the rest of the stack) but rejected: that would mean a
second continuously-running EC2 instance for a component that does no
inference at all, just an HTTP client — doubling Free Tier instance-hour
consumption for no real benefit, straight after having to pivot the API's
own deployment specifically *because of* that same budget constraint (see
"Actually deployed on Elastic Beanstalk, not App Runner" above). Streamlit
Cloud is free, deploys straight from this repo, and keeps the frontend and
API genuinely decoupled — it reaches the API the same way any external
client would, over the public `medsys.us-east-1.elasticbeanstalk.com`
endpoint, not via any AWS-internal networking. One real cost: Streamlit
Cloud doesn't support `uv`'s `pyproject.toml`/`uv.lock` format (misreads
`pyproject.toml` as Poetry format), so `frontend/requirements.txt` exists
as a plain, minimally-scoped dependency list just for this one deployment
target — a small, contained bit of duplication versus a second AWS
environment's worth of ops overhead.

## Real-world testing surfaced a genuine domain-shift limitation

After deployment, manual testing against realistic (not abstract-style)
triage phrasing surfaced concrete misclassifications, confirming a risk
`model-card.md` had already flagged theoretically ("Domain shift risk")
before any evidence existed. Four examples, and what they revealed:

| Input | Predicted (before fix) | Root cause |
|---|---|---|
| "Unresponsive, no detectable pulse, non-breathing." | `normal` | Words *are* in vocabulary (`unresponsive`, `pulse`, `breathing`), but the model never learned to associate this register with urgency — trained only on formal PubMed-style abstracts, never on clinical shorthand. |
| "stomachache" | `cardiovascular` / `urgent` | Genuinely **zero** TF-IDF features — `stomachache` never appears in the training vocabulary at all (only `stomach` does; TF-IDF doesn't do subword matching). With no real evidence, the prediction is driven entirely by ComplementNB's structural class bias (the same safety-tilt that won model-selection) applied to an empty vector, not a real judgment. |
| "Acute respiratory distress, ... severe facial/airway swelling, blood pressure 80/50 mmHg following a bee sting." | `attention` (anaphylaxis, should be `urgent`) | Keyword-adjustment design bug: capped at exactly one tier regardless of how many escalate words matched. 2 hits (`acute`, `severe`) only moved normal→attention, not further. |
| "Asymptomatic patient requesting a routine prescription renewal for hypertension medication; mild ... rash ..." | `attention` (should be closer to `normal`) | Category-level miss (`cardiovascular`, likely from "hypertension" dominating the TF-IDF signal despite being mentioned as routine background, not the active complaint) compounded by the same one-tier cap masking the 2 de-escalate hits (`routine`, `mild`) that should have corrected further. |

**Two of these are genuine, code-only bugs — fixed, not just documented:**

1. **Keyword-adjustment now scales with the net score instead of capping at
   one tier** (`app/urgency.py`): `tier = clamp(baseline + net, 0, 2)`
   instead of `tier = baseline ± 1`. Verified against the real cases:
   anaphylaxis now reaches `urgent` (was `attention`); the routine-renewal
   case now reaches `normal` (was `attention`) — the de-escalate signal was
   strong enough to fully correct what would otherwise have been a
   dangerous over-triage from the flawed category prediction. No
   retraining involved — pure inference-time logic change.

2. **A low-confidence guard now flags near-empty-signal inputs**
   (`app/model.py::has_known_vocabulary`, wired into `/predict`'s response
   as `low_confidence: bool`). Checks token overlap against the
   vectorizer's vocabulary (exported once, as plain JSON, from the
   already-fitted `models/pipeline.joblib` — reading a fitted model's
   learned vocabulary isn't training, so this touches nothing that would
   risk test-set leakage). Verified: "stomachache" now returns
   `low_confidence: true`.

   **Revised after initial deployment**: first version *floored* urgency
   at `attention` (raised a low guess up, left a high one unchanged) —
   for "stomachache," the raw category (`cardiovascular`) already implied
   `urgent`, so the floor was a no-op and the UI displayed `URGENT` right
   next to a warning about low confidence, which read as contradictory
   even though each part was individually correct. Caught via user
   testing of the deployed fix. The real issue was the floor's premise: a
   raw `urgent` guess isn't actually safer or more justified than
   `normal` when there's zero real evidence — it's equally ungrounded, in
   the other direction. Changed to a **fixed override**: `low_confidence`
   now always forces `urgency = "attention"` outright, discarding the raw
   guess entirely rather than taking its max against a floor — a
   deliberate "flag for human review" signal, not a hedge in either
   direction. Also added a `message` field (populated only when
   `low_confidence`) giving the caller concrete guidance to resubmit with
   more clinical detail — surfaced in the Streamlit UI's warning too, not
   just the API response.

**One is a genuine limitation, not fixable by either change — documented,
not silently accepted.** The cardiac-arrest example (`normal`,
`low_confidence: false`) fails because its vocabulary genuinely does
overlap with training data, just not in a way the model learned to
associate with urgency — a real register mismatch between the *Medical
Abstracts TC Corpus* (formal, third-person, academic case-report writing)
and how urgency is actually communicated in short clinical/triage
phrasing. **More training data of the same kind would not fix this** —
the corpus is the only data source available, and more abstracts would
only make the model marginally better at abstracts, not teach it a
register it was never shown. A real fix would need training examples in
that different register — real (or realistically synthesized) triage-note
phrasing, not more academic abstracts — which is a genuine scope increase
beyond what this project's timeline supports, not a quick follow-up. See
`model-card.md` § Caveats for how this is now recorded as a validated,
evidenced limitation rather than a theoretical one.

## Out of scope (deliberately)

Kubernetes/HPA/KEDA, Canary/Shadow deployment, implemented drift detection
(PSI/KS — mentioned in the README as future work, not built), word/transformer
embeddings, joint automated search (AutoML) over model+features+
hyperparameters, ROC-AUC/PR-AUC/MCC, any real production traffic. All
considered and consciously dropped for the reasons above — not from lack of
awareness of what exists, but because they aren't justified within this
challenge's scope and timeline.
