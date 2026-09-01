"""Tests for app/urgency.py: baseline mapping, keyword adjustment."""

from app.urgency import (
    BASELINE_URGENCY as APP_BASELINE_URGENCY,
)
from app.urgency import (
    TIER_ORDER as APP_TIER_ORDER,
)
from app.urgency import keyword_adjustment, predict_urgency
from training.urgency import BASELINE_URGENCY as TRAINING_BASELINE_URGENCY
from training.urgency import TIER_ORDER as TRAINING_TIER_ORDER

NEUTRAL_TEXT = "a report describing findings observed in the patient sample"


def test_app_and_training_urgency_mappings_stay_in_sync():
    assert APP_BASELINE_URGENCY == TRAINING_BASELINE_URGENCY
    assert APP_TIER_ORDER == TRAINING_TIER_ORDER


def test_baseline_urgency_with_no_keyword_hits():
    assert predict_urgency("cardiovascular diseases", NEUTRAL_TEXT) == "urgent"
    assert predict_urgency("nervous system diseases", NEUTRAL_TEXT) == "attention"
    assert predict_urgency("general pathological conditions", NEUTRAL_TEXT) == "normal"


def test_escalate_keyword_nudges_tier_up():
    text = "an acute presentation with sudden onset"
    assert predict_urgency("nervous system diseases", text) == "urgent"


def test_deescalate_keyword_nudges_tier_down():
    text = "a chronic, stable, routine follow-up"
    assert predict_urgency("nervous system diseases", text) == "normal"


def test_urgent_capped_not_pushed_past_top_tier():
    text = "an acute, severe, critical, life-threatening emergency"
    assert predict_urgency("cardiovascular diseases", text) == "urgent"


def test_normal_floored_not_pushed_below_bottom_tier():
    text = "a chronic, stable, mild, long-term case"
    assert predict_urgency("general pathological conditions", text) == "normal"


def test_equal_escalate_and_deescalate_hits_cancel_out():
    text = "acute but stable"
    assert keyword_adjustment(text) == 0
    assert predict_urgency("nervous system diseases", text) == "attention"


def test_keyword_matching_is_case_insensitive():
    assert keyword_adjustment("ACUTE presentation") == 1


def test_keyword_matching_respects_word_boundaries():
    assert keyword_adjustment("an inaccurate measurement") == 0
