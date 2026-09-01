# MedSys — Hospital Laudo Triage Classifier

Postgraduate tech challenge (POS Tech / MLET, Fase 3): automatic triage of
medical text reports (laudos médicos) into urgency tiers — **normal /
attention / urgent** — served as a REST API, with a CI/CD pipeline,
monitoring stack, and a latency-optimized model.

This README is filled in progressively as the project is built. See
`docs/architecture.md` (the *what*) and `docs/technical-decisions.md` (the
*why*) for the full build plan and rationale.

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
