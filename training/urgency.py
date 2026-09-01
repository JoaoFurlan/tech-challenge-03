"""Category -> baseline urgency tier mapping, per docs/architecture.md.

Kept separate from the keyword-adjustment layer (which belongs to the
serving app, app/urgency.py, once built) — this is only the piece needed to
evaluate models against the actual triage objective, not just raw category
accuracy. Three of the five categories collapse to the same "attention"
tier, so confusing them has zero effect on the real triage output.
"""

TIER_ORDER = ["normal", "attention", "urgent"]

BASELINE_URGENCY = {
    "cardiovascular diseases": "urgent",
    "nervous system diseases": "attention",
    "neoplasms": "attention",
    "digestive system diseases": "attention",
    "general pathological conditions": "normal",
}


def tier_rank(category: str) -> int:
    return TIER_ORDER.index(BASELINE_URGENCY[category])


CATEGORIES = sorted(BASELINE_URGENCY)
