"""Tests for scripts/legibility/census_trigger.py — periodic legibility
census trigger evaluator + census-state reader (task 2579 / PRD task ζ).

See plans/confusion-reduction-prd.md §6 (task ζ fire logic), §7.4 (census
config block), §7.5 (census state contract), §8.5 (boundary-test matrix).

Imported as a namespace package (`from legibility import census_trigger`)
since scripts/legibility/ is a subdir of scripts/ (on sys.path via
scripts/tests/conftest.py) with no __init__.py — same convention as
test_codebook.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from legibility import census_trigger as ct

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# step-1: RED — CensusConfig defaults + from_mapping merge
# ---------------------------------------------------------------------------

def test_census_config_defaults_match_prd_section_7_4():
    config = ct.CensusConfig()
    assert config.max_interval_days == 10
    assert config.tasks_landed_threshold == 120
    assert config.tasks_landed_min_days == 7
    assert config.novelty_spike_count == 4
    assert config.novelty_spike_window_hours == 72
    assert config.floor_days == 5


def test_census_config_from_mapping_merges_partial_overrides_over_defaults():
    config = ct.CensusConfig.from_mapping(
        {"max_interval_days": 3, "novelty_spike": {"count": 9}}
    )
    assert config.max_interval_days == 3
    assert config.novelty_spike_count == 9
    # untouched fields keep their §7.4 defaults
    assert config.tasks_landed_threshold == 120
    assert config.tasks_landed_min_days == 7
    assert config.novelty_spike_window_hours == 72
    assert config.floor_days == 5


def test_census_config_from_mapping_none_returns_defaults():
    assert ct.CensusConfig.from_mapping(None) == ct.CensusConfig()


def test_census_config_from_mapping_empty_dict_returns_defaults():
    assert ct.CensusConfig.from_mapping({}) == ct.CensusConfig()


# ---------------------------------------------------------------------------
# step-3: RED — evaluate() condition (a): max_interval_days
# ---------------------------------------------------------------------------

def _evaluate_a(*, last_census_at, tasks_landed=None, candidate_first_seens=None, config=None):
    """Helper fixing the args condition (a)'s tests hold constant: has
    censused, no spike, caller-controlled tasks_landed/config."""
    return ct.evaluate(
        now=NOW,
        last_census_at=last_census_at,
        never_censused=False,
        tasks_landed=tasks_landed,
        candidate_first_seens=candidate_first_seens or [],
        config=config or ct.CensusConfig(),
    )


def test_evaluate_condition_a_day_9_no_fire():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=9))
    assert decision.fire is False


def test_evaluate_condition_a_day_10_fires():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=10))
    assert decision.fire is True
    assert any("max-interval" in r for r in decision.reasons)


def test_evaluate_condition_a_day_12_fires():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=12))
    assert decision.fire is True
    assert any("max-interval" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# step-5: RED — evaluate() condition (b): tasks_landed
# ---------------------------------------------------------------------------

def test_evaluate_condition_b_day_7_130_landed_fires():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=7), tasks_landed=130)
    assert decision.fire is True
    assert any("tasks-landed" in r for r in decision.reasons)


def test_evaluate_condition_b_day_7_below_threshold_no_fire():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=7), tasks_landed=100)
    assert decision.fire is False


def test_evaluate_condition_b_day_6_min_days_not_met_no_fire():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=6), tasks_landed=130)
    assert decision.fire is False


def test_evaluate_condition_b_delta_unavailable_no_fire():
    decision = _evaluate_a(last_census_at=NOW - timedelta(days=9), tasks_landed=None)
    assert decision.fire is False
