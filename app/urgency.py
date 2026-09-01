"""Category -> urgency mapping: baseline tier + keyword adjustment.

Deterministic, documented business rule layered on top of the trained
classifier's category prediction — see docs/architecture.md and
docs/technical-decisions.md for why urgency isn't learned directly.

BASELINE_URGENCY/TIER_ORDER mirror training/urgency.py (used there for
evaluation, not the deployed API) — kept as a separate, self-contained
copy rather than a cross-package import so the serving container doesn't
need the training pipeline's dependencies. tests/test_urgency.py asserts
the two stay in sync.
"""

import re

TIER_ORDER = ["normal", "attention", "urgent"]

BASELINE_URGENCY = {
    "cardiovascular diseases": "urgent",
    "nervous system diseases": "attention",
    "neoplasms": "attention",
    "digestive system diseases": "attention",
    "general pathological conditions": "normal",
}

ESCALATE_WORDS = {"acute", "emergency", "severe", "critical", "sudden", "life-threatening"}
DE_ESCALATE_WORDS = {"chronic", "stable", "routine", "mild", "follow-up", "long-term"}

_ESCALATE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in ESCALATE_WORDS) + r")\b", re.IGNORECASE
)
_DE_ESCALATE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in DE_ESCALATE_WORDS) + r")\b", re.IGNORECASE
)


def keyword_adjustment(text: str) -> int:
    """Net escalate/de-escalate keyword hits: positive nudges urgency up."""
    escalate_hits = len(_ESCALATE_PATTERN.findall(text))
    de_escalate_hits = len(_DE_ESCALATE_PATTERN.findall(text))
    return escalate_hits - de_escalate_hits


def predict_urgency(category: str, text: str) -> str:
    """Category baseline, nudged one tier by keyword adjustment (capped)."""
    baseline_tier = TIER_ORDER.index(BASELINE_URGENCY[category])
    net = keyword_adjustment(text)

    if net > 0:
        tier = min(baseline_tier + 1, len(TIER_ORDER) - 1)
    elif net < 0:
        tier = max(baseline_tier - 1, 0)
    else:
        tier = baseline_tier

    return TIER_ORDER[tier]
