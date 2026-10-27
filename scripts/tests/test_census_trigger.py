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

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from legibility import census_trigger as ct
from legibility.config import Census as LegibilityCensus
from shared.testing_mcp_stub import StubAnswer, StubMcpServer

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=UTC)


def _anchor(kind, days_ago):
    return ct.FloorAnchor(kind=kind, at=NOW - timedelta(days=days_ago))


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
    assert config.novelty_spike_multiple == 2
    assert config.novelty_spike_baseline_days == 30
    assert config.floor_days == 5


def test_census_config_from_mapping_merges_partial_overrides_over_defaults():
    config = ct.CensusConfig.from_mapping(
        {"max_interval_days": 3, "novelty_spike": {"count": 9, "baseline_days": 14}}
    )
    assert config.max_interval_days == 3
    assert config.novelty_spike_count == 9
    assert config.novelty_spike_baseline_days == 14
    assert config.novelty_spike_multiple == 2
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
# task 4085 (amendment pass, review finding #1): the `census:` block is
# hand-edited -- the same operator-authored-input class as the hand-seeded
# `last_census_done_count` baseline this task hardens -- so an unusable
# threshold VALUE is an expected input. Unvalidated, a quoted `'10'` reaches
# `evaluate()`'s `days_since >= config.max_interval_days` and raises
# TypeError straight out of `load_census_config` and `decide_for_project`,
# both of which document a never-raises contract, and into `census.main`,
# which calls the latter unguarded.
#
# Same rule as the baseline guard: VALIDATE, never coerce. `int('10')` would
# silently bless a malformed config and could arm a ~$100 census on an
# unaudited number.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,key,mapping_key",
    [
        ("max_interval_days", "max_interval_days", None),
        ("tasks_landed_threshold", "tasks_landed_threshold", None),
        ("tasks_landed_min_days", "tasks_landed_min_days", None),
        ("floor_days", "floor_days", None),
        ("novelty_spike_count", "count", "novelty_spike"),
        ("novelty_spike_window_hours", "window_hours", "novelty_spike"),
        ("novelty_spike_multiple", "multiple", "novelty_spike"),
        ("novelty_spike_baseline_days", "baseline_days", "novelty_spike"),
    ],
)
@pytest.mark.parametrize("bad", ["10", "", 10.0, 1.5, True, False, -1, None, [10], {"n": 10}])
def test_census_config_from_mapping_rejects_unusable_threshold_values(
    field, key, mapping_key, bad, caplog
):
    """Each of the six §7.4 thresholds falls back to its OWN default, alone,
    with one WARNING naming the value and its type. `True`/`False` are in the
    parameter list deliberately: `bool` is an `int` subclass, so an unquoted
    YAML `true` would otherwise mean 1 -- a daily census -- in total silence.
    A negative is definitionally impossible for a day/count threshold."""
    override = {key: bad} if mapping_key is None else {mapping_key: {key: bad}}

    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping(override)

    # The bad field fell back; nothing else moved.
    assert config == ct.CensusConfig()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert key in message
    assert type(bad).__name__ in message


def test_census_config_from_mapping_rejects_a_non_mapping_novelty_spike(caplog):
    """`census: {novelty_spike: [4, 72]}` used to raise AttributeError from
    the `.get` on a list -- a raise from inside the config reader, which is
    the one place that must not have one."""
    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping({"novelty_spike": [4, 72], "floor_days": 3})

    assert config.novelty_spike_count == ct.CensusConfig().novelty_spike_count
    assert config.novelty_spike_window_hours == ct.CensusConfig().novelty_spike_window_hours
    # ...and the sibling override alongside it is still honoured.
    assert config.floor_days == 3

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "novelty_spike" in warnings[0].getMessage()


def test_census_config_from_mapping_reports_every_bad_value_in_one_warning(caplog):
    """Per-FIELD fallback, per-BLOCK warning. The six thresholds are
    independent, so one typo must not disarm the other five; but a single
    hand-edited block is a single operator fault, and this WARNING is one
    nightly journal line, so all of them are named in one message."""
    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping(
            {
                "max_interval_days": "10",
                "floor_days": -1,
                "tasks_landed_threshold": 200,  # the one good override
                "novelty_spike": {"window_hours": None, "multiple": "2"},
            }
        )

    assert config.tasks_landed_threshold == 200
    assert config.max_interval_days == ct.CensusConfig().max_interval_days
    assert config.floor_days == ct.CensusConfig().floor_days
    assert config.novelty_spike_window_hours == ct.CensusConfig().novelty_spike_window_hours
    assert config.novelty_spike_multiple == ct.CensusConfig().novelty_spike_multiple

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for named in ("max_interval_days", "floor_days", "window_hours", "multiple"):
        assert named in message
    assert "tasks_landed_threshold" not in message


def test_census_config_from_mapping_accepts_every_valid_int_silently(caplog):
    """Over-rejection guard. `0` is aggressive but legal (fire every day) and
    must not be mistaken for a missing value, and there is no upper bound."""
    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping(
            {
                "max_interval_days": 0,
                "floor_days": 0,
                "tasks_landed_threshold": 10_000,
                "novelty_spike": {
                    "count": 0, "window_hours": 1, "multiple": 0, "baseline_days": 0,
                },
            }
        )

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    assert config.max_interval_days == 0
    assert config.floor_days == 0
    assert config.tasks_landed_threshold == 10_000
    assert config.novelty_spike_count == 0
    assert config.novelty_spike_window_hours == 1
    assert config.novelty_spike_multiple == 0
    assert config.novelty_spike_baseline_days == 0


# ---------------------------------------------------------------------------
# amendment pass (review finding #3): CensusConfig defaults must not
# silently drift from task β's legibility.config.Census model, now that β
# has landed and lives alongside this module in scripts/legibility/.
# ---------------------------------------------------------------------------

def test_census_config_defaults_match_legibility_config_census_model():
    """Regression guard for the reuse fix: CensusConfig's field defaults are
    sourced from legibility.config.Census (task β), not independently
    hardcoded, so the two schemas cannot silently drift apart."""
    beta_defaults = LegibilityCensus()
    config = ct.CensusConfig()
    assert config.max_interval_days == beta_defaults.max_interval_days
    assert config.tasks_landed_threshold == beta_defaults.tasks_landed_threshold
    assert config.tasks_landed_min_days == beta_defaults.tasks_landed_min_days
    assert config.novelty_spike_count == beta_defaults.novelty_spike.count
    assert config.novelty_spike_window_hours == beta_defaults.novelty_spike.window_hours
    assert config.novelty_spike_multiple == beta_defaults.novelty_spike.multiple
    assert config.novelty_spike_baseline_days == beta_defaults.novelty_spike.baseline_days
    assert config.floor_days == beta_defaults.floor_days


# ---------------------------------------------------------------------------
# step-3: RED — evaluate() condition (a): max_interval_days
# ---------------------------------------------------------------------------

def _evaluate_a(*, last_census_at, tasks_landed=None, candidate_first_seens=None, config=None):
    """Helper fixing the args condition (a)'s tests hold constant: has
    censused, no spike, caller-controlled tasks_landed/config."""
    return ct.evaluate(
        now=NOW,
        last_census_at=last_census_at,
        floor_anchor=ct.FloorAnchor(ct.FloorAnchorKind.SESSION_WATERMARK, last_census_at),
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


# ---------------------------------------------------------------------------
# relative novelty spike against a trailing baseline
# (plans/census-incremental-prd.md §4.7 (b), §4.8 row 13)
# ---------------------------------------------------------------------------

_TODAY = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=UTC)


def _first_seens_over_a_baseline_of_40(current_72h):
    """35 days of history whose every 72h window holds exactly 40 candidates
    (period-3 daily counts 13, 13, 14), then today topped up so the window
    ending at NOW holds `current_72h`."""
    daily = {days_ago: (13, 13, 14)[days_ago % 3] for days_ago in range(1, 36)}
    daily[0] = daily[3] + (current_72h - 40)
    return [_TODAY - timedelta(days=days_ago) for days_ago, n in daily.items() for _ in range(n)]


def _quiet_history():
    return [NOW - timedelta(days=40)]


def _evaluate_novelty(first_seens, config=None, now=NOW):
    return ct.evaluate(
        now=now,
        last_census_at=now - timedelta(days=6),
        floor_anchor=ct.FloorAnchor(ct.FloorAnchorKind.SESSION_WATERMARK, now - timedelta(days=6)),
        tasks_landed=None,
        candidate_first_seens=first_seens,
        config=config or ct.CensusConfig(),
    )


def _novelty_line(decision):
    (line,) = [r for r in decision.reasons if r.startswith("novelty-spike:")]
    return line


def test_novelty_spike_43_against_baseline_median_40_does_not_fire():
    decision = _evaluate_novelty(_first_seens_over_a_baseline_of_40(43))
    assert decision.fire is False
    assert _novelty_line(decision) == "novelty-spike: 43 within 72h (baseline median 40, x2)"


def test_novelty_spike_90_against_baseline_median_40_fires():
    decision = _evaluate_novelty(_first_seens_over_a_baseline_of_40(90))
    assert decision.fire is True
    assert (
        _novelty_line(decision) == "novelty-spike: 90 within 72h (baseline median 40, x2) -> FIRE"
    )


def test_novelty_spike_needs_the_absolute_minimum_too():
    three = _evaluate_novelty(_quiet_history() + [_TODAY] * 3)
    assert three.fire is False
    assert _novelty_line(three) == "novelty-spike: 3 within 72h (baseline median 0, x2)"

    four = _evaluate_novelty(_quiet_history() + [_TODAY] * 4)
    assert four.fire is True
    assert _novelty_line(four).endswith("-> FIRE")


def test_novelty_spike_is_na_with_less_than_baseline_days_of_history():
    ten_days = [_TODAY - timedelta(days=i % 11) for i in range(90)]
    decision = _evaluate_novelty(ten_days)
    assert decision.fire is False
    line = _novelty_line(decision)
    assert "N/A" in line
    assert "30d" in line
    assert "10.5d" in line


def test_novelty_spike_window_counts_72h_inclusive():
    # 2026-07-11 is 84h before NOW (excluded); NOW - 72h exactly is included.
    dated = [
        datetime.fromisoformat(d) for d in ("2026-07-14", "2026-07-13", "2026-07-12", "2026-07-11")
    ]
    decision = _evaluate_novelty(_quiet_history() + dated + [NOW - timedelta(hours=72)])
    assert _novelty_line(decision).startswith("novelty-spike: 4 within 72h")


def test_novelty_spike_baseline_days_zero_degenerates_to_the_absolute_rule():
    decision = _evaluate_novelty(
        [_TODAY] * 4, config=ct.CensusConfig(novelty_spike_baseline_days=0)
    )
    assert decision.fire is True
    assert "baseline median 0" in _novelty_line(decision)


def test_novelty_spike_multiple_is_honoured():
    decision = _evaluate_novelty(
        _first_seens_over_a_baseline_of_40(90), config=ct.CensusConfig(novelty_spike_multiple=3)
    )
    assert decision.fire is False
    assert _novelty_line(decision).endswith("(baseline median 40, x3)")


# Histogram of candidates[].first_seen in docs/legibility/confusion-codebook.yaml
# at 92af0716e8 (1,132 candidates).
_RECORDED_DAILY_NEW_CANDIDATES = {
    "2026-07-13": 4, "2026-07-14": 4, "2026-07-15": 5, "2026-07-16": 2, "2026-07-17": 2,
    "2026-07-18": 1, "2026-07-19": 6, "2026-07-20": 11, "2026-07-21": 6, "2026-07-22": 35,
    "2026-07-23": 5, "2026-07-24": 2, "2026-07-25": 3, "2026-07-26": 1, "2026-07-28": 3,
    "2026-07-29": 7, "2026-07-30": 32, "2026-07-31": 24, "2026-08-01": 8, "2026-08-02": 23,
    "2026-08-03": 20, "2026-08-04": 9, "2026-08-05": 17, "2026-08-06": 15, "2026-08-07": 16,
    "2026-08-08": 15, "2026-08-09": 16, "2026-08-10": 17, "2026-08-11": 7, "2026-08-12": 20,
    "2026-08-13": 2, "2026-08-15": 18, "2026-08-16": 18, "2026-08-17": 1, "2026-08-18": 24,
    "2026-08-19": 22, "2026-08-21": 11, "2026-08-22": 23, "2026-08-23": 20, "2026-08-24": 28,
    "2026-08-25": 28, "2026-08-26": 14, "2026-08-27": 23, "2026-08-28": 24, "2026-08-29": 16,
    "2026-08-31": 16, "2026-09-01": 8, "2026-09-02": 7, "2026-09-03": 20, "2026-09-04": 15,
    "2026-09-05": 12, "2026-09-06": 13, "2026-09-07": 21, "2026-09-08": 31, "2026-09-09": 21,
    "2026-09-10": 20, "2026-09-11": 63, "2026-09-12": 25, "2026-09-13": 18, "2026-09-14": 1,
    "2026-09-15": 21, "2026-09-16": 18, "2026-09-17": 22, "2026-09-18": 20, "2026-09-19": 3,
    "2026-09-20": 16, "2026-09-21": 25, "2026-09-22": 12, "2026-09-23": 15, "2026-09-24": 6,
    "2026-09-25": 21, "2026-09-26": 11, "2026-09-27": 18, "2026-09-28": 1, "2026-10-02": 15,
    "2026-10-03": 9,
}


def _recorded_72h_count(day):
    return sum(
        _RECORDED_DAILY_NEW_CANDIDATES.get((day - timedelta(days=back)).isoformat(), 0)
        for back in range(3)
    )


def test_novelty_spike_replay_fires_on_the_relative_rule_not_the_absolute_one():
    recorded = {
        "version": 2,
        "entries": [],
        "candidates": _candidates_from_dates(
            [day for day, n in _RECORDED_DAILY_NEW_CANDIDATES.items() for _ in range(n)]
        ),
    }
    _, first_seens = ct.codebook_signal(recorded)

    lines, fired, absolute_days = {}, {}, set()
    day = date(2026, 8, 15)
    while day <= date(2026, 10, 3):
        now = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=12)
        decision = _evaluate_novelty(first_seens, now=now)
        lines[day] = _novelty_line(decision)
        if decision.fire:
            fired[day] = lines[day]
        if _recorded_72h_count(day) >= 4:
            absolute_days.add(day)
        day += timedelta(days=1)

    assert fired == {
        date(2026, 9, 11): "novelty-spike: 104 within 72h (baseline median 43.5, x2) -> FIRE",
        date(2026, 9, 12): "novelty-spike: 108 within 72h (baseline median 44.5, x2) -> FIRE",
        date(2026, 9, 13): "novelty-spike: 106 within 72h (baseline median 46, x2) -> FIRE",
    }
    assert lines[date(2026, 8, 25)] == "novelty-spike: 76 within 72h (baseline median 43.5, x2)"
    assert len(absolute_days) == 48
    assert set(fired) < absolute_days


# ---------------------------------------------------------------------------
# hard floor: measured from its anchor, blocks a/b/c, always reported
# (plans/census-incremental-prd.md §4.7 Floor)
# ---------------------------------------------------------------------------

def _evaluate_floor(*, last_census_days_ago, floor_anchor, tasks_landed=None, config=None):
    return ct.evaluate(
        now=NOW,
        last_census_at=NOW - timedelta(days=last_census_days_ago),
        floor_anchor=floor_anchor,
        tasks_landed=tasks_landed,
        candidate_first_seens=[],
        config=config or ct.CensusConfig(),
    )


def _floor_lines(decision):
    return [r for r in decision.reasons if r.startswith("floor:")]


def test_evaluate_floor_blocks_inside_floor_days_of_its_anchor():
    decision = _evaluate_floor(
        last_census_days_ago=12,
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 3),
    )
    assert decision.fire is False
    assert _floor_lines(decision) == [
        "floor: 3.0d since session watermark (floor 5d) -> BLOCKS all conditions"
    ]


def test_evaluate_floor_blocks_tasks_landed_too():
    decision = _evaluate_floor(
        last_census_days_ago=8,
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 4),
        tasks_landed=999,
    )
    assert decision.fire is False
    assert any("BLOCKS" in line for line in _floor_lines(decision))


def test_evaluate_floor_clear_still_names_its_anchor():
    decision = _evaluate_floor(
        last_census_days_ago=12,
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 6),
    )
    assert decision.fire is True
    assert _floor_lines(decision) == ["floor: 6.0d since session watermark (floor 5d)"]


@pytest.mark.parametrize(
    "kind,label",
    [
        (ct.FloorAnchorKind.LAST_CENSUS_AT, "since last_census_at (no watermark yet)"),
        (ct.FloorAnchorKind.EARLIEST_CODEBOOK_DATE, "since earliest codebook date (never censused)"),
        (ct.FloorAnchorKind.SESSION_WATERMARK, "since session watermark"),
    ],
)
def test_evaluate_floor_line_names_each_anchor_kind(kind, label):
    decision = _evaluate_floor(last_census_days_ago=12, floor_anchor=_anchor(kind, 3))
    floor_lines = _floor_lines(decision)
    assert len(floor_lines) == 1
    assert label in floor_lines[0]
    assert "BLOCKS" in floor_lines[0]


def test_evaluate_floor_applies_to_a_never_censused_project():
    decision = _evaluate_floor(
        last_census_days_ago=3,
        floor_anchor=_anchor(ct.FloorAnchorKind.EARLIEST_CODEBOOK_DATE, 3),
        config=ct.CensusConfig(max_interval_days=1),
    )
    assert decision.fire is False
    (floor_line,) = _floor_lines(decision)
    assert "earliest codebook date (never censused)" in floor_line


def test_evaluate_floor_without_an_anchor_blocks_and_says_so():
    decision = _evaluate_floor(last_census_days_ago=12, floor_anchor=None)
    assert decision.fire is False
    (floor_line,) = _floor_lines(decision)
    assert floor_line.startswith("floor: no anchor")
    assert "BLOCKS" in floor_line


def test_evaluate_floor_never_fires_by_itself():
    decision = _evaluate_floor(
        last_census_days_ago=9,
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 100),
    )
    assert decision.fire is False
    (floor_line,) = _floor_lines(decision)
    assert "BLOCKS" not in floor_line


# ---------------------------------------------------------------------------
# step-11: RED — load_census_state() three-valued (ok/missing/malformed)
# ---------------------------------------------------------------------------

def test_load_census_state_missing_file_is_missing_with_no_warning(tmp_path, caplog):
    path = tmp_path / "census-state.json"

    with caplog.at_level(logging.WARNING):
        status, data = ct.load_census_state(path)

    assert status == "missing"
    assert data is None
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize(
    "content",
    [
        "not json{",
        "null",
        "[]",
        '{"last_census_at": "not-a-date"}',
    ],
    ids=["invalid-json", "json-null", "non-dict-list", "unparseable-last-census-at"],
)
def test_load_census_state_malformed_variants_are_malformed_with_one_warning(
    tmp_path, caplog, content
):
    path = tmp_path / "census-state.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        status, data = ct.load_census_state(path)

    assert status == "malformed"
    assert data is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_load_census_state_valid_file_is_ok_with_no_warning(tmp_path, caplog):
    path = tmp_path / "census-state.json"
    path.write_text(
        json.dumps(
            {
                "last_census_at": "2026-07-01T00:00:00+00:00",
                "last_census_report": "plans/confusion-census-2026-07-01.md",
                "last_census_done_count": 500,
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        status, data = ct.load_census_state(path)

    assert status == "ok"
    # tuple[str, dict | None] — None only for the missing/malformed statuses.
    assert data is not None
    assert data["last_census_at"] == "2026-07-01T00:00:00+00:00"
    assert data["last_census_report"] == "plans/confusion-census-2026-07-01.md"
    assert data["last_census_done_count"] == 500
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# task 6397: the six-key census state, and the legacy shapes it replaces
# ---------------------------------------------------------------------------

_LEGACY_STATE = {
    "last_census_at": "2026-10-03",
    "last_census_report": "/home/leo/src/dark-factory/plans/confusion-census-2026-10-03.md",
    "last_census_done_count": 2872,
}

_SIX_KEY_STATE = {
    "last_census_at": "2026-10-06",
    "last_census_run_id": "census-dark_factory-20261006",
    "last_census_report": "plans/confusion-census-2026-10-06.md",
    "last_census_as_of_sha": "a" * 40,
    "session_watermark": "2026-10-06T00:00:00+00:00",
    "last_census_done_count": 2900,
}


@pytest.mark.parametrize(
    "state",
    [
        _LEGACY_STATE,
        _SIX_KEY_STATE,
        {"last_census_at": "2026-10-01T09:30:00+00:00", "last_census_report": "x.md"},
        {**_SIX_KEY_STATE, "session_watermark": None},
    ],
    ids=["legacy", "six-key", "hand-seeded-datetime", "null-watermark"],
)
def test_load_census_state_accepts_every_supported_shape(tmp_path, caplog, state):
    path = tmp_path / "census-state.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        status, data = ct.load_census_state(path)

    assert (status, data) == ("ok", state)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("watermark", ["yesterday", 5])
def test_load_census_state_unparseable_watermark_is_malformed_with_one_warning(
    tmp_path, caplog, watermark,
):
    path = tmp_path / "census-state.json"
    path.write_text(
        json.dumps({**_SIX_KEY_STATE, "session_watermark": watermark}), encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        status, data = ct.load_census_state(path)

    assert (status, data) == ("malformed", None)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "session_watermark" in warnings[0].getMessage()
    assert repr(watermark) in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# step-13: RED — codebook_signal() + load_census_config()
# ---------------------------------------------------------------------------

def _sighting(date, session="s"):
    return {
        "date": date,
        "project": "dark_factory",
        "session": session,
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
    }


def test_codebook_signal_empty_codebook_returns_none_and_empty_list():
    codebook = {"version": 2, "entries": [], "candidates": []}
    earliest, first_seens = ct.codebook_signal(codebook)
    assert earliest is None
    assert first_seens == []


def test_codebook_signal_collects_earliest_across_entries_and_candidates():
    codebook = {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "t",
                "severity": "low",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [_sighting("2026-06-20", "s1"), _sighting("2026-07-01", "s2")],
            }
        ],
        "candidates": [
            {
                "id": "cand-20260705-1",
                "title": "x",
                "first_seen": "2026-07-05",
                "disposition": "pending",
                "sightings": [_sighting("2026-07-06", "s3")],
            },
            {
                "id": "cand-20260710-1",
                "title": "y",
                "first_seen": "2026-07-10",
                "disposition": "pending",
                "sightings": [],
            },
        ],
    }
    earliest, first_seens = ct.codebook_signal(codebook)
    assert earliest == datetime.fromisoformat("2026-06-20")
    assert first_seens == [
        datetime.fromisoformat("2026-07-05"),
        datetime.fromisoformat("2026-07-10"),
    ]


def test_load_census_config_absent_file_returns_defaults(tmp_path):
    config = ct.load_census_config(tmp_path)
    assert config == ct.CensusConfig()


def test_load_census_config_partial_override_merges_over_defaults(tmp_path):
    legibility_dir = tmp_path / "docs" / "legibility"
    legibility_dir.mkdir(parents=True)
    (legibility_dir / "legibility.yaml").write_text(
        "census:\n  max_interval_days: 3\n  novelty_spike:\n    count: 9\n",
        encoding="utf-8",
    )
    config = ct.load_census_config(tmp_path)
    assert config.max_interval_days == 3
    assert config.novelty_spike_count == 9
    assert config.floor_days == 5  # untouched -- still default


def test_load_census_config_malformed_file_returns_defaults_with_one_warning(tmp_path, caplog):
    legibility_dir = tmp_path / "docs" / "legibility"
    legibility_dir.mkdir(parents=True)
    (legibility_dir / "legibility.yaml").write_text("{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = ct.load_census_config(tmp_path)

    assert config == ct.CensusConfig()
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


# ---------------------------------------------------------------------------
# step-15: RED — compute_tasks_landed() fail-safe delta + default_status_fetcher()
# ---------------------------------------------------------------------------

def _wrapped_fetcher(statuses):
    """A fake status_fetcher mimicking get_statuses' wrapped envelope
    (fused-memory/src/fused_memory/server/tools.py:2665): `{"statuses": {id: status}}`."""
    return lambda: {"statuses": statuses}


def test_compute_tasks_landed_happy_path_returns_delta_over_baseline():
    state = {"last_census_done_count": 500}
    statuses = {str(i): "done" for i in range(630)}
    statuses.update({str(i): "pending" for i in range(630, 645)})

    result = ct.compute_tasks_landed(state=state, status_fetcher=_wrapped_fetcher(statuses))

    assert result == 130


def test_compute_tasks_landed_no_fetcher_returns_none_with_one_warning(caplog):
    state = {"last_census_done_count": 500}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(state=state, status_fetcher=None)

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_compute_tasks_landed_raising_fetcher_returns_none_with_one_warning(caplog):
    state = {"last_census_done_count": 500}

    def _raising_fetcher():
        raise RuntimeError("get_statuses unreachable")

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(state=state, status_fetcher=_raising_fetcher)

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_compute_tasks_landed_missing_baseline_returns_none_with_one_warning(caplog):
    state = {"last_census_at": "2026-07-01T00:00:00+00:00"}  # no last_census_done_count

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=_wrapped_fetcher({"1": "done"})
        )

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


# ---------------------------------------------------------------------------
# task 3291: extract_done_count() — the ONE place a get_statuses payload
# becomes a number, and the one place the silent-zero hole is closed.
#
# The idiom this replaces -- `(payload.get("statuses") or {})`, duplicated at
# census_trigger.compute_tasks_landed and census.run_census -- coerced EVERY
# unusable payload to a done-count of 0 with no warning and no exception.
# That is how a fabricated 0 was persisted as a real census baseline on
# 2026-07-24 and again on 2026-07-31.
#
# Be precise about the harm, because the arithmetic matters: while the fetch
# was also broken the poisoned baseline was self-cancelling (`current_done`
# was zeroed by the same defect, so the delta was `0 - 0` and condition (b)
# did NOT fire -- measured by replaying the 2026-07-31 decision against the
# pre-task code). It is unsound because it ARMS (b) with a delta of ~2872,
# ~24x its 120 threshold, the instant the fetch is repaired. Which is why
# the payload guard, the absolute-project_root fix and the on-disk baseline
# repair all had to land together. See census_trigger's module docstring.
# ---------------------------------------------------------------------------

# fused-memory's `_normalize_project_root` hard-rejects a relative path with
# exactly this payload. Verified live against localhost:8002 with
# `{"project_root": "."}` -- and critically, the JSON-RPC envelope carries
# `isError: false`, so `_extract_tool_result` unwraps this dict as though it
# were a genuine tool result and hands it straight to the caller.
_TOOL_ERROR_ENVELOPE = {
    "error": "project_root must be a non-empty absolute path, got: '.'",
    "error_type": "ValidationError",
}


def test_extract_done_count_counts_done_values():
    assert ct.extract_done_count({"statuses": {"1": "done", "2": "done", "3": "pending"}}) == 2


def test_extract_done_count_empty_statuses_is_a_valid_zero():
    """A project with zero done tasks is a real, expected state and MUST stay
    distinguishable from a failed call -- it is precisely the case
    `advance_census_state`'s "0 is never dropped as falsy" contract exists
    for. The distinction that matters is presence-of-shape, not
    emptiness-of-content."""
    assert ct.extract_done_count({"statuses": {}}) == 0


def test_extract_done_count_raises_on_fused_memory_tool_error_envelope():
    """The regression that poisoned the live baseline. `@mcp_tool_errors`
    returns this dict with `isError: false` at the JSON-RPC layer, so
    `_extract_tool_result` unwraps it happily; the old
    `(payload.get("statuses") or {})` idiom then read it as a done-count of
    0 -- silently, with no warning and no exception."""
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_done_count(_TOOL_ERROR_ENVELOPE)

    # structured-facts-at-failure: the operator must see the REAL cause, not
    # a generic shape complaint, so the server's own error text is quoted.
    assert "project_root must be a non-empty absolute path" in str(excinfo.value)


def test_extract_done_count_raises_when_statuses_key_absent():
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_done_count({})

    assert "statuses" in str(excinfo.value)


def test_extract_done_count_prefers_a_present_statuses_key_over_an_error_key():
    """Pin the documented precedence of the error-envelope guard: it is
    conditioned on `"statuses" not in payload`, so a payload that DOES carry a
    real status snapshot is counted even if some stray `error` key rides along.
    Untested, a future reorder of the two guards would flip this silently --
    and the wrong direction (rejecting a usable snapshot) reintroduces exactly
    the never-observable baseline this module exists to avoid."""
    assert ct.extract_done_count({"statuses": {"1": "done"}, "error": "x"}) == 1


def test_extract_done_count_raises_when_statuses_is_not_a_mapping():
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_done_count({"statuses": ["1", "2"]})

    # The message must name the offending shape, not just that something failed.
    assert "statuses" in str(excinfo.value)
    assert "list" in str(excinfo.value)


@pytest.mark.parametrize("payload", [None, [], "oops", 0])
def test_extract_done_count_raises_on_non_dict_payload(payload):
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_done_count(payload)

    assert type(payload).__name__ in str(excinfo.value)


# ---------------------------------------------------------------------------
# task 6404: extract_statuses() — the validated {task id: status} mapping a
# census run reads ONCE and derives both its outcome feed and its done-count
# from. extract_done_count builds on it, so the two cannot drift.
# ---------------------------------------------------------------------------


def test_extract_statuses_returns_the_validated_mapping():
    assert dict(ct.extract_statuses({"statuses": {"12": "done", "13": "pending"}})) == {
        "12": "done",
        "13": "pending",
    }


def test_extract_statuses_empty_statuses_is_a_valid_empty_mapping():
    assert dict(ct.extract_statuses({"statuses": {}})) == {}


def test_extract_statuses_raises_on_fused_memory_tool_error_envelope():
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_statuses(_TOOL_ERROR_ENVELOPE)

    assert "project_root must be a non-empty absolute path" in str(excinfo.value)


def test_extract_statuses_raises_when_statuses_key_absent():
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_statuses({})

    assert "statuses" in str(excinfo.value)


def test_extract_statuses_raises_when_statuses_is_not_a_mapping():
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_statuses({"statuses": ["1", "2"]})

    assert "statuses" in str(excinfo.value)
    assert "list" in str(excinfo.value)


@pytest.mark.parametrize("payload", [None, [], "oops", 0])
def test_extract_statuses_raises_on_non_dict_payload(payload):
    with pytest.raises(ct.StatusFetchUnavailable) as excinfo:
        ct.extract_statuses(payload)

    assert type(payload).__name__ in str(excinfo.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"statuses": {}},
        {"statuses": {"1": "done", "2": "done", "3": "pending"}},
        {"statuses": {"1": "done"}, "error": "x"},
        {"statuses": {"7": "cancelled", "8": "done", "9": "in-progress"}},
    ],
)
def test_extract_done_count_is_the_done_count_of_extract_statuses(payload):
    statuses = ct.extract_statuses(payload)

    assert ct.extract_done_count(payload) == sum(
        1 for status in statuses.values() if status == "done"
    )


def test_compute_tasks_landed_tool_error_envelope_returns_none_with_one_warning(caplog):
    """Before task 3291 this returned `0 - 500 == -500` with ZERO warnings --
    a silent negative delta from a payload that was never a status snapshot
    at all. A shape failure must land in the same "one WARNING, return None"
    fail-safe branch as an unreachable server."""
    state = {"last_census_done_count": 500}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=lambda: _TOOL_ERROR_ENVELOPE
        )

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_compute_tasks_landed_payload_without_statuses_returns_none_with_one_warning(caplog):
    state = {"last_census_done_count": 500}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(state=state, status_fetcher=lambda: {})

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_compute_tasks_landed_empty_project_computes_a_real_zero_delta(caplog):
    """Regression guard in the OPPOSITE direction: collapsing "empty result"
    into "failed call" would break the first baseline of a newly-onboarded
    project with no done tasks. A real 0 baseline against a real empty
    snapshot is a real delta of 0, not a fail-safe None, and warns nothing."""
    state = {"last_census_done_count": 0}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(state=state, status_fetcher=_wrapped_fetcher({}))

    assert result == 0
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# ---------------------------------------------------------------------------
# task 4085: the BASELINE side of the same fail-safe contract.
#
# task 3291 (above) made an unusable get_statuses PAYLOAD fail safe. The other
# input to the same subtraction -- the `last_census_done_count` baseline read
# off disk -- was left unvalidated, and it is the one an operator hand-edits
# (skills/census/SKILL.md documents seeding it by hand), so malformed values
# are an expected input rather than a hypothetical. Three measured behaviours,
# all from the single unguarded `current_done - baseline`:
#
#   * `"2872"` (a quoted JSON int) raises TypeError out of a function whose
#     docstring promises a fail-safe None. Via the `evaluate` CLI, main()'s
#     catch-all swallows it into NO-FIRE + exit 0, which discards the ENTIRE
#     evaluation -- so conditions (a) max-interval and (c) novelty-spike are
#     suppressed by a fault in (b), taking out the very backstop that exists
#     to survive a broken (b);
#   * `true` returns `current_done - 1` with ZERO warnings, because
#     `isinstance(True, int)` is True in Python -- arming (b) with essentially
#     every done task ever. Strictly more dangerous than the crash: nothing
#     signals it at all;
#   * `1.5` returns a fractional delta that flows into evaluate()'s
#     `>= threshold` comparison as a real number.
#
# The contract pinned here: a baseline that is not a non-negative int fails
# SAFE (return None, condition (b) inert, (a)/(c) unaffected) with exactly one
# WARNING naming the value and its type. VALIDATE, never coerce -- `int("2872")`
# would silently bless a malformed state file and arm a ~$100 census run on an
# unaudited number, which is precisely the defect class task 3291 closed on the
# fetch side.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "baseline",
    ["2872", "", 2872.0, 1.5, True, False, [2872], {"n": 2872}, object()],
    ids=["str-int", "str-empty", "float-whole", "float-frac", "bool-true", "bool-false",
         "list", "dict", "object"],
)
def test_compute_tasks_landed_non_int_baseline_fails_safe_with_one_warning(baseline, caplog):
    """Every non-int baseline lands in the same "one WARNING, return None"
    branch the fetch and payload faults already use.

    `True`/`False`/`2872.0` are the interesting parameters: they are not
    TypeErrors, they are SILENT wrong answers today (`True` == 1, so the delta
    is `current_done - 1`), which is why a bare `except TypeError` would not
    have been a fix."""
    state = {"last_census_done_count": baseline}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=_wrapped_fetcher(_done_statuses(3000))
        )

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # Both facts, because either alone is ambiguous in a journal line: the
    # repr says WHICH value is wrong, the type name says WHY it is wrong
    # (`"2872"` and `2872` are visually near-identical without it).
    assert repr(baseline) in message
    assert type(baseline).__name__ in message


def test_compute_tasks_landed_negative_baseline_fails_safe_with_one_warning(caplog):
    """A done-count cannot be negative. Today `-5` silently inflates the delta
    in the FIRE-ward direction (`current_done + 5`), so it belongs in the same
    guard as the type faults rather than being treated as a usable int."""
    state = {"last_census_done_count": -5}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=_wrapped_fetcher(_done_statuses(600))
        )

    assert result is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_compute_tasks_landed_bad_baseline_warns_before_touching_the_fetcher(caplog):
    """Pins the guard's POSITION, not just its existence.

    The baseline is a state-side fault and must be reported BEFORE the
    `status_fetcher is None` branch and before any fetch. That ordering is what
    makes a mis-seeded baseline visible on the DEFAULT nightly trickle wiring,
    where `run_nightly` passes `status_fetcher=None` (nightly.py) and the
    fetcher guard would otherwise return first -- leaving a corrupt state file
    unreported in total silence for as long as it sits there."""
    calls = []

    def _recording_fetcher():
        calls.append(1)
        return {"statuses": {"1": "done"}}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state={"last_census_done_count": "2872"}, status_fetcher=_recording_fetcher
        )

    assert result is None
    assert calls == []
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1

    # ...and with no fetcher at all, the single warning is still the specific
    # baseline complaint, not the generic "no status_fetcher configured" one.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state={"last_census_done_count": "2872"}, status_fetcher=None
        )

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "last_census_done_count" in warnings[0].getMessage()


def test_compute_tasks_landed_bad_baseline_warning_is_bounded(caplog):
    """The WARNING is re-emitted into the nightly systemd journal as ONE line,
    and census-state.json is hand-editable, so an arbitrarily large value must
    not dump itself into the log (`_bounded_repr`'s whole reason to exist)."""
    state = {"last_census_done_count": "9" * 5000}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=_wrapped_fetcher(_done_statuses(10))
        )

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert len(message) < 1000
    assert "repr truncated" in message


@pytest.mark.parametrize("baseline,expected", [(0, 600), (500, 100), (2872, -2272)])
def test_compute_tasks_landed_int_baseline_still_computes_delta(baseline, expected, caplog):
    """Over-rejection guard: a real non-negative int -- including a real 0 and a
    baseline larger than the current count -- still computes its delta and warns
    nothing. Complements
    `test_compute_tasks_landed_empty_project_computes_a_real_zero_delta`, which
    pins the same property from the payload side."""
    state = {"last_census_done_count": baseline}

    with caplog.at_level(logging.WARNING):
        result = ct.compute_tasks_landed(
            state=state, status_fetcher=_wrapped_fetcher(_done_statuses(600))
        )

    assert result == expected
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# ---------------------------------------------------------------------------
# task 4085: `_bounded_repr` / `_bounded_keys` — direct coverage.
#
# CHARACTERIZATION, with no paired implementation change: both helpers are
# already correct. They had no direct tests at all (exercised only indirectly
# through `extract_done_count`'s messages), and the baseline guard above
# promotes `_bounded_repr` from an incidental helper to the mechanism that
# keeps a malformed hand-seeded baseline -- an arbitrarily large hand-edited
# value -- to ONE nightly journal line.
#
# Scope, per the amendment pass (review finding #3): these pin the PROPERTY
# each helper exists for -- output stays bounded, the elision is marked, and
# the true size is still reported -- not the exact wording of a private
# format string, and not the `<=`/`>` boundary arithmetic, which would only
# re-derive the implementation. The end-to-end guarantee that a malformed
# baseline cannot dump itself into a journal line is pinned separately and
# behaviourally by
# `test_compute_tasks_landed_bad_baseline_warning_is_bounded` above; these
# three are the unit-level backstop for the helpers it leans on.
#
# The two constants are read from the module rather than hard-coded, so
# retuning 200/20 does not produce a false failure here.
# ---------------------------------------------------------------------------

def test_bounded_repr_bounds_an_arbitrarily_large_value():
    """The point of the helper: census-state.json and a get_statuses payload
    are both arbitrarily large, and neither may dump itself into a journal
    line. A value small enough to print in full is left alone."""
    assert "repr truncated" not in ct._bounded_repr({"n": 1})

    value = "x" * 100_000
    result = ct._bounded_repr(value)

    # Bounded, marked as elided, and the TRUE size still stated -- that number
    # is how an operator tells "slightly over" from "a whole payload".
    assert len(result) < ct._MAX_REPR_CHARS + 100
    assert "repr truncated" in result
    assert str(len(repr(value))) in result


def test_bounded_keys_bounds_an_arbitrarily_wide_mapping():
    """Same property for the key list: a get_statuses payload over a big
    project has thousands of keys, and the shape is all diagnosis needs."""
    count = ct._MAX_REPORTED_KEYS + 5
    mapping = {f"k{i:02d}": i for i in range(count)}

    result = ct._bounded_keys(mapping)

    shown = sum(1 for i in range(count) if f"'k{i:02d}'" in result)
    assert shown == ct._MAX_REPORTED_KEYS
    # The suffix reports the TRUE key count, not the number shown.
    assert str(count) in result


def test_bounded_keys_survives_mixed_type_keys():
    """The invariant the `key=str` sort exists to protect. A bare
    `sorted({1: ..., "b": ..., None: ...})` raises TypeError comparing int to
    str -- i.e. the error path would fail while reporting a failure, which is
    the worst possible place for it. get_statuses payloads are server-supplied,
    so a mixed-key dict is exactly the malformed shape this reports on."""
    result = ct._bounded_keys({1: "a", "b": "c", None: "d"})

    assert "1" in result
    assert "'b'" in result
    assert "None" in result


# ---------------------------------------------------------------------------
# default_status_fetcher against a real-socket stand-in for fused-memory,
# reached through FUSED_MEMORY_MCP_URL.
# ---------------------------------------------------------------------------

def _done_statuses(count):
    return {str(i): "done" for i in range(count)}


def _fetch_from(server, monkeypatch, project_root):
    monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
    return ct.default_status_fetcher(project_root)()


def _get_statuses_arguments(server):
    """The arguments of the one tools/call *server* received, which must be
    get_statuses."""
    [call] = [
        r.body for r in server.requests
        if isinstance(r.body, dict) and r.body.get("method") == "tools/call"
    ]
    assert call["params"]["name"] == "get_statuses"
    return call["params"]["arguments"]


def test_default_status_fetcher_returns_the_unwrapped_get_statuses_reply(tmp_path, monkeypatch):
    with StubMcpServer(stateful=False, payload={"statuses": {"1": "done"}}) as server:
        reply = _fetch_from(server, monkeypatch, tmp_path)

    assert reply == {"statuses": {"1": "done"}}


def test_default_status_fetcher_raises_status_fetch_unavailable_when_unreachable(
    tmp_path, monkeypatch,
):
    """A refused connection surfaces as StatusFetchUnavailable naming the
    URL, never as a raw transport exception."""
    with StubMcpServer(stateful=False) as server:
        pass
    monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
    fetcher = ct.default_status_fetcher(tmp_path)

    with pytest.raises(ct.StatusFetchUnavailable, match=re.escape(f"unreachable at {server.url}")):
        fetcher()


def test_default_status_fetcher_raises_status_fetch_unavailable_on_a_tool_error(
    tmp_path, monkeypatch,
):
    """An isError reply is a failed fetch, not a done-count of 0."""
    with (
        StubMcpServer(stateful=False, answer=StubAnswer.IS_ERROR) as server,
        pytest.raises(ct.StatusFetchUnavailable) as excinfo,
    ):
        _fetch_from(server, monkeypatch, tmp_path)

    assert server.url in str(excinfo.value)


def test_default_status_fetcher_sends_absolute_project_root_for_dot(tmp_path, monkeypatch):
    """fused-memory's `_normalize_project_root` rejects any relative path: the
    MCP argument is resolved by the SERVER's cwd, not the client's. The
    `evaluate` CLI's "." default reaches this resolve."""
    monkeypatch.chdir(tmp_path)

    with StubMcpServer(stateful=False, payload={"statuses": {}}) as server:
        _fetch_from(server, monkeypatch, ".")
        sent = _get_statuses_arguments(server)["project_root"]

    assert sent == str(tmp_path.resolve())
    assert Path(sent).is_absolute()


def test_default_status_fetcher_sends_absolute_project_root_for_relative_subdir(
    tmp_path, monkeypatch,
):
    """Second case so the fix cannot pass by special-casing "." alone."""
    (tmp_path / "sub" / "dir").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with StubMcpServer(stateful=False, payload={"statuses": {}}) as server:
        _fetch_from(server, monkeypatch, "sub/dir")
        sent = _get_statuses_arguments(server)["project_root"]

    assert sent == str((tmp_path / "sub" / "dir").resolve())
    assert Path(sent).is_absolute()


def test_default_status_fetcher_leaves_absolute_project_root_unchanged(tmp_path, monkeypatch):
    """An already-absolute root crosses the wire unchanged. fused-memory keys
    its `_MAIN_CHECKOUT_CACHE` on the path it is handed, so the resolve must
    normalise relative paths and rewrite nothing else."""
    monkeypatch.chdir(tmp_path)
    absolute_root = str(tmp_path)

    with StubMcpServer(stateful=False, payload={"statuses": {}}) as server:
        _fetch_from(server, monkeypatch, absolute_root)
        sent = _get_statuses_arguments(server)["project_root"]

    assert sent == absolute_root


def test_default_status_fetcher_feeds_compute_tasks_landed_the_correct_delta(
    tmp_path, monkeypatch,
):
    """The fetcher's reply is exactly the shape compute_tasks_landed reads:
    630 done of 645 against a 500 baseline is a delta of 130."""
    statuses = _done_statuses(630)
    statuses.update({str(i): "pending" for i in range(630, 645)})

    with StubMcpServer(stateful=False, payload={"statuses": statuses}) as server:
        monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
        result = ct.compute_tasks_landed(
            state={"last_census_done_count": 500},
            status_fetcher=ct.default_status_fetcher(tmp_path),
        )

    assert result == 130


# ---------------------------------------------------------------------------
# step-17: RED — decide_for_project() end-to-end over the full §8.5 matrix
# ---------------------------------------------------------------------------

def _write_codebook(project_root, entries=None, candidates=None):
    codebook = {"version": 2, "entries": entries or [], "candidates": candidates or []}
    legibility_dir = project_root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "confusion-codebook.yaml").write_text(
        yaml.safe_dump(codebook), encoding="utf-8"
    )


def _write_census_state(project_root, **fields):
    legibility_dir = project_root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "census-state.json").write_text(json.dumps(fields), encoding="utf-8")


def _candidates_from_dates(dates):
    return [
        {"id": f"cand-{i}", "title": "x", "first_seen": d, "disposition": "pending", "sightings": []}
        for i, d in enumerate(dates)
    ]


def _done_statuses(count):
    return {str(i): "done" for i in range(count)}


def test_decide_for_project_row1_missing_state_fires_from_earliest_sighting(tmp_path):
    _write_codebook(
        tmp_path,
        entries=[
            {
                "id": "entry-a",
                "title": "t",
                "severity": "low",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [_sighting("2026-07-02")],  # NOW - 12 days
            }
        ],
    )
    # No census-state.json written -- never censused.

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is True
    assert any("max-interval" in r for r in decision.reasons)
    assert (
        "floor: 12.5d since earliest codebook date (never censused) (floor 5d)"
        in decision.reasons
    )


def _write_legibility_yaml(project_root, text):
    legibility_dir = project_root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "legibility.yaml").write_text(text, encoding="utf-8")


def test_decide_for_project_legacy_state_floor_falls_back_to_last_census_at(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=3)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
    )
    _write_legibility_yaml(tmp_path, "census:\n  max_interval_days: 1\n")

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    assert (
        "floor: 3.0d since last_census_at (no watermark yet) (floor 5d) -> BLOCKS all conditions"
        in decision.reasons
    )


def test_decide_for_project_never_censused_floor_blocks_from_the_earliest_codebook_date(
    tmp_path,
):
    _write_codebook(
        tmp_path,
        entries=[
            {
                "id": "entry-a",
                "title": "t",
                "severity": "low",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [_sighting((NOW - timedelta(days=3)).date().isoformat())],
            }
        ],
    )
    _write_legibility_yaml(tmp_path, "census:\n  max_interval_days: 1\n")

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    floor_lines = [r for r in decision.reasons if r.startswith("floor:")]
    assert len(floor_lines) == 1
    assert "since earliest codebook date (never censused)" in floor_lines[0]
    assert "BLOCKS" in floor_lines[0]


def _watermarked_state(*, last_census_days_ago, watermark_days_ago) -> dict[str, str | None]:
    return {
        "last_census_at": (NOW - timedelta(days=last_census_days_ago)).isoformat(),
        "last_census_report": "plans/confusion-census-prior.md",
        "last_census_run_id": "census-dark_factory-20260702",
        "session_watermark": (NOW - timedelta(days=watermark_days_ago)).isoformat(),
    }


def test_decide_for_project_floor_reads_the_session_watermark_not_last_census_at(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=12, watermark_days_ago=3)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert "max-interval: 12.0d since last census (threshold 10d) -> FIRE" in decision.reasons
    assert decision.fire is False
    assert (
        "floor: 3.0d since session watermark (floor 5d) -> BLOCKS all conditions"
        in decision.reasons
    )


def test_decide_for_project_floor_clears_on_an_old_watermark(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=12, watermark_days_ago=12)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is True
    assert any(r.startswith("max-interval:") and "-> FIRE" in r for r in decision.reasons)
    assert "floor: 12.0d since session watermark (floor 5d)" in decision.reasons


@pytest.mark.parametrize("watermark_present", [True, False], ids=["null", "absent"])
def test_decide_for_project_null_or_absent_watermark_falls_back_to_last_census_at(
    tmp_path, watermark_present
):
    state = _watermarked_state(last_census_days_ago=12, watermark_days_ago=12)
    if watermark_present:
        state["session_watermark"] = None
    else:
        del state["session_watermark"]
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **state)

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is True
    assert "floor: 12.0d since last_census_at (no watermark yet) (floor 5d)" in decision.reasons


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"last_census_report": "plans/confusion-census-prior.md"},
        {"last_census_at": None, "session_watermark": None},
    ],
    ids=["empty", "report-only", "null-timestamps"],
)
def test_decide_for_project_ok_state_without_timestamps_blocks_on_no_anchor(tmp_path, state):
    _write_spike_codebook(tmp_path)
    _write_census_state(tmp_path, **state)

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )
    assert decision.fire is False
    assert (
        "floor: no anchor (no session watermark, last_census_at or codebook date)"
        " -> BLOCKS all conditions" in decision.reasons
    )


def test_decide_for_project_row2_day9_no_spike_low_delta_no_fire(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=9)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )
    def fetcher():  # delta 50 < 120
        return {"statuses": _done_statuses(550)}

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=fetcher)

    assert decision.fire is False


def test_decide_for_project_row3_day7_130_landed_fires(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=7)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )
    def fetcher():  # delta 130 >= 120
        return {"statuses": _done_statuses(630)}

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=fetcher)

    assert decision.fire is True
    assert any("tasks-landed" in r for r in decision.reasons)


def _write_spike_codebook(project_root):
    _write_codebook(
        project_root,
        candidates=_candidates_from_dates(
            [fs.date().isoformat() for fs in _first_seens_over_a_baseline_of_40(90)]
        ),
    )


def test_decide_for_project_row4_novelty_spike_fires(tmp_path):
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=6)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is True
    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )


def test_decide_for_project_row5_floor_blocks_novelty_spike_no_fire(tmp_path):
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=3)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    assert (
        "floor: 3.0d since session watermark (floor 5d) -> BLOCKS all conditions"
        in decision.reasons
    )


def test_decide_for_project_row6_malformed_state_no_fire_one_warning(tmp_path, caplog):
    _write_codebook(tmp_path)
    legibility_dir = tmp_path / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "census-state.json").write_text("not json{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_decide_for_project_survives_a_raising_compute_tasks_landed(
    tmp_path, caplog, monkeypatch
):
    """`decide_for_project` must never propagate — and must degrade condition
    (b) ALONE, keeping (a) and (c) alive.

    The callee is monkeypatched to raise rather than fed a bad baseline, and
    that is deliberate: a bad baseline no longer raises, so a state-file-driven
    test would silently become a duplicate of the compute_tasks_landed tests
    above and this guard would ship untested. Injecting a raising callee pins
    the contract independently of which inner fault happens to be possible
    today.

    The stakes are in the degradation granularity. Today the whole call chain
    is unguarded, so via the `evaluate` CLI a single fault in (b) is caught only
    by main()'s outermost catch-all, which discards the ENTIRE evaluation — the
    max-interval backstop that exists precisely to survive a broken (b) is taken
    out by the same fault."""
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=9)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )

    def _boom(*, state, status_fetcher):
        raise RuntimeError("baseline exploded")

    monkeypatch.setattr(ct, "compute_tasks_landed", _boom)

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(
            tmp_path, now=NOW, status_fetcher=_wrapped_fetcher(_done_statuses(550))
        )

    assert isinstance(decision, ct.Decision)
    assert decision.fire is False

    # Degraded to a fail-safe None delta, NOT to a fabricated number: this is
    # the exact line evaluate() emits for `tasks_landed=None`.
    assert any("tasks-landed: delta unavailable" in r for r in decision.reasons)
    # ...while conditions (a) and (c) still evaluated — the whole point of
    # degrading (b) alone.
    assert any("max-interval" in r for r in decision.reasons)
    assert any("novelty-spike" in r for r in decision.reasons)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()
    assert "baseline exploded" in warnings[0].getMessage()


def test_decide_for_project_survives_a_quoted_max_interval_days(tmp_path, caplog):
    """task 4085 (amendment): the end-to-end pin for the config guard, and the
    one that shows why per-field fallback is the right degradation.

    `census: {max_interval_days: '10'}` in a hand-edited legibility.yaml used
    to reach `evaluate()`'s `days_since >= config.max_interval_days` and raise
    `TypeError: '>=' not supported between instances of 'float' and 'str'`
    straight out of this function -- verified live before the fix -- defeating
    its documented never-raises contract and, via the `evaluate` CLI's
    outermost catch-all, discarding the WHOLE evaluation.

    Here the state is day-11, so with the rejected field back at its default
    of 10 the (a) max-interval backstop still FIRES. That is the property that
    matters: a typo in one threshold must not disarm the backstop that exists
    precisely to survive everything else being broken."""
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=11)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )
    (tmp_path / "docs" / "legibility" / "legibility.yaml").write_text(
        "census:\n  max_interval_days: '10'\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(
            tmp_path, now=NOW, status_fetcher=_wrapped_fetcher(_done_statuses(550))
        )

    assert isinstance(decision, ct.Decision)
    assert decision.fire is True
    assert any("max-interval" in r for r in decision.reasons)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "max_interval_days" in message
    assert "str" in message


def test_decide_for_project_bounds_a_huge_tasks_landed_failure(
    tmp_path, caplog, monkeypatch
):
    """task 4085 (amendment): the guard above must keep this module's
    one-WARNING-one-line discipline. A `StatusFetchUnavailable` chained from a
    big get_statuses payload carries an arbitrarily large message, and this
    WARNING lands in the nightly journal as a single line."""
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=9)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )

    def _boom(*, state, status_fetcher):
        raise ct.StatusFetchUnavailable("payload " + "x" * 50_000 + "\nsecond line")

    monkeypatch.setattr(ct, "compute_tasks_landed", _boom)

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(
            tmp_path, now=NOW, status_fetcher=_wrapped_fetcher(_done_statuses(550))
        )

    assert decision.fire is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert len(message) < 1000
    assert "\n" not in message
    assert "StatusFetchUnavailable" in message
    assert "repr truncated" in message


# ---------------------------------------------------------------------------
# step-19: RED — CLI `evaluate` subcommand (always exits 0, fail-safe)
# ---------------------------------------------------------------------------

def test_cli_evaluate_dark_factory_like_project_prints_no_fire_and_exits_0(tmp_path, capsys):
    # Mirrors the live dark_factory codebook today: candidates: [], no
    # dated sightings, no census-state.json.
    _write_codebook(tmp_path)

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION: NO-FIRE" in captured.out
    assert len(captured.out.strip().splitlines()) > 1  # decision line + >=1 reason line


def test_cli_evaluate_fire_inducing_fixture_prints_fire_and_exits_0(tmp_path, capsys):
    twelve_days_ago = (datetime.now(UTC) - timedelta(days=12)).strftime("%Y-%m-%d")
    _write_codebook(
        tmp_path,
        entries=[
            {
                "id": "entry-a",
                "title": "t",
                "severity": "low",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [_sighting(twelve_days_ago)],
            }
        ],
    )

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION: FIRE" in captured.out
    assert "max-interval" in captured.out


def test_cli_evaluate_malformed_state_never_crashes_and_exits_0(tmp_path, capsys):
    _write_codebook(tmp_path)
    legibility_dir = tmp_path / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "census-state.json").write_text("not json{", encoding="utf-8")

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION:" in captured.out


def _cli_floor_line(tmp_path, capsys):
    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    (floor_line,) = [
        line for line in captured.out.splitlines() if line.startswith("floor:")
    ]
    return floor_line


def test_cli_evaluate_prints_a_floor_line_naming_its_anchor(tmp_path, capsys):
    six_days_ago = datetime.now(UTC) - timedelta(days=6)
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=six_days_ago.date().isoformat(),
        last_census_report="plans/confusion-census-prior.md",
    )

    assert "since last_census_at (no watermark yet)" in _cli_floor_line(tmp_path, capsys)

    _write_census_state(
        tmp_path,
        last_census_at=six_days_ago.date().isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        session_watermark=six_days_ago.isoformat(),
    )

    assert "since session watermark" in _cli_floor_line(tmp_path, capsys)

