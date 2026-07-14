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

from legibility import census_trigger as ct


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
