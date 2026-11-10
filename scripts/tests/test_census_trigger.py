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
import socket
import subprocess
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


RUN_ID = "census-p-20261101"
_FIRING = ct.Completion(ct.CompletionStatus.FIRE, RUN_ID, 70, share=0.8235)
_NOT_YET = ct.Completion(ct.CompletionStatus.NOT_YET, RUN_ID, 70, share=0.5)
_ERRORED = ct.Completion(
    ct.CompletionStatus.ERROR, RUN_ID, 70, detail="exit 2, store_unreachable"
)


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


def test_census_config_completion_threshold_pct_defaults_to_seventy():
    assert ct.CensusConfig().completion_threshold_pct == 70


@pytest.mark.parametrize("value", [1, 85, 100])
def test_census_config_from_mapping_accepts_a_completion_threshold_in_range_silently(
    value, caplog
):
    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping({"completion_threshold_pct": value})

    assert config.completion_threshold_pct == value
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.parametrize("bad", [0, 101, -1, "70", 70.0, True, None, [70]])
def test_census_config_from_mapping_rejects_a_completion_threshold_outside_1_to_100(
    bad, caplog
):
    """check_run_completion rejects a threshold outside (0, 1] with a usage
    error, so an unvalidated 0 would become a nightly completion ERROR
    streak instead of one config WARNING naming the bound."""
    with caplog.at_level(logging.WARNING):
        config = ct.CensusConfig.from_mapping({"completion_threshold_pct": bad})

    assert config == ct.CensusConfig()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "completion_threshold_pct" in message
    assert type(bad).__name__ in message
    assert "1..100" in message


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
    assert config.completion_threshold_pct == beta_defaults.completion_threshold_pct == 70


# ---------------------------------------------------------------------------
# relative novelty spike against a trailing baseline
# (plans/census-incremental-prd.md §4.7 (b), §4.8 row 13)
# ---------------------------------------------------------------------------

_TODAY = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=UTC)


def _first_seens_over_a_baseline_of_40(current_72h, today=_TODAY):
    """35 days of history whose every 72h window holds exactly 40 candidates
    (period-3 daily counts 13, 13, 14), then *today* topped up so the window
    ending at *today* noon holds `current_72h`."""
    daily = {days_ago: (13, 13, 14)[days_ago % 3] for days_ago in range(1, 36)}
    daily[0] = daily[3] + (current_72h - 40)
    return [today - timedelta(days=days_ago) for days_ago, n in daily.items() for _ in range(n)]


def _quiet_history():
    return [NOW - timedelta(days=40)]


def _evaluate_novelty(first_seens, config=None, now=NOW):
    return ct.evaluate(
        now=now,
        floor_anchor=ct.FloorAnchor(ct.FloorAnchorKind.SESSION_WATERMARK, now - timedelta(days=6)),
        completion=_NOT_YET,
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

def _evaluate_floor(*, floor_anchor, completion=_NOT_YET, config=None):
    return ct.evaluate(
        now=NOW,
        floor_anchor=floor_anchor,
        completion=completion,
        candidate_first_seens=[],
        config=config or ct.CensusConfig(),
    )


def _floor_lines(decision):
    return [r for r in decision.reasons if r.startswith("floor:")]


def test_evaluate_floor_blocks_inside_floor_days_of_its_anchor():
    decision = _evaluate_floor(
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 3), completion=_FIRING,
    )
    assert decision.fire is False
    assert _floor_lines(decision) == [
        "floor: 3.0d since session watermark (floor 5d) -> BLOCKS all conditions"
    ]


def test_evaluate_floor_clear_still_names_its_anchor():
    decision = _evaluate_floor(
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, 6), completion=_FIRING,
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
    decision = _evaluate_floor(floor_anchor=_anchor(kind, 3), completion=_FIRING)
    floor_lines = _floor_lines(decision)
    assert len(floor_lines) == 1
    assert label in floor_lines[0]
    assert "BLOCKS" in floor_lines[0]


def test_evaluate_floor_applies_to_a_never_censused_project():
    decision = _evaluate_floor(
        floor_anchor=_anchor(ct.FloorAnchorKind.EARLIEST_CODEBOOK_DATE, 3),
        completion=_FIRING,
    )
    assert decision.fire is False
    (floor_line,) = _floor_lines(decision)
    assert "earliest codebook date (never censused)" in floor_line


def test_evaluate_floor_without_an_anchor_blocks_and_says_so():
    decision = _evaluate_floor(floor_anchor=None, completion=_FIRING)
    assert decision.fire is False
    (floor_line,) = _floor_lines(decision)
    assert floor_line.startswith("floor: no anchor")
    assert "BLOCKS" in floor_line


def test_evaluate_floor_never_fires_by_itself():
    decision = _evaluate_floor(
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
# `isError: false`, so the MCP client unwraps this dict as though it were a
# genuine tool result and hands it straight to the caller.
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
    returns this dict with `isError: false` at the JSON-RPC layer, so the
    MCP client unwraps it happily; the old
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
# completion — condition (a): the previous run's weighted completion, read
# through scripts/check_run_completion.py (plans/census-incremental-prd.md
# C7 (a), D9, row 13)
# ---------------------------------------------------------------------------

def _verdict_json(verdict, share, *, error=None, vacuous=False):
    record = {
        "run": RUN_ID, "verdict": verdict, "share": share, "threshold": 0.7,
        "landed": None, "pending": None, "total": None, "vacuous": vacuous,
        "by_priority": None,
    }
    if error is not None:
        record["error"] = error
    return json.dumps(record, separators=(",", ":"))


class _ScriptedRunner:
    """A completion runner answering one canned exit code and stdout, or
    raising *raises*, and recording every argv it was handed."""

    def __init__(self, returncode=0, stdout="", *, raises=None):
        self.returncode = returncode
        self.stdout = stdout
        self.raises = raises
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(list(args), self.returncode, stdout=self.stdout)


def _check(runner, project_root, *, run_id=RUN_ID, threshold_pct=70):
    return ct.check_completion(
        run_id, project_root=project_root, threshold_pct=threshold_pct, runner=runner,
    )


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_check_completion_exit_0_fires_on_the_observed_project_only(tmp_path, caplog):
    runner = _ScriptedRunner(0, "run x: 3 tasks\n" + _verdict_json("pass", 0.8235) + "\n")

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.FIRE
    assert completion.fires is True
    assert completion.reason == f"completion: 0.8235 of {RUN_ID} (threshold 0.70) -> FIRE"
    assert runner.calls == [[
        "--run", RUN_ID, "--threshold", "0.7", "--project-root", str(tmp_path.resolve()),
    ]]
    assert _warnings(caplog) == []


def test_check_completion_exit_75_is_not_yet(tmp_path, caplog):
    runner = _ScriptedRunner(75, _verdict_json("not_yet", 0.5) + "\n")

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.NOT_YET
    assert completion.fires is False
    assert completion.reason == f"completion: 0.5 of {RUN_ID} (threshold 0.70)"
    assert _warnings(caplog) == []


def test_check_completion_vacuous_pass_fires_and_says_vacuous(tmp_path):
    runner = _ScriptedRunner(0, _verdict_json("pass", None, vacuous=True))

    completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.FIRE
    assert "vacuous" in completion.reason
    assert completion.reason.endswith("-> FIRE")


def test_check_completion_passes_the_configured_threshold_as_a_fraction(tmp_path):
    runner = _ScriptedRunner(0, _verdict_json("pass", 0.9))

    completion = _check(runner, tmp_path, threshold_pct=85)

    assert runner.calls[0][2:4] == ["--threshold", "0.85"]
    assert "(threshold 0.85)" in completion.reason


def test_check_completion_empty_run_is_filed_nothing(tmp_path, caplog):
    runner = _ScriptedRunner(2, "run x: could not check\n" + _verdict_json("error", None, error="empty_run"))

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.FILED_NOTHING
    assert completion.fires is False
    assert completion.reason == f"completion: run {RUN_ID} filed no tasks -> N/A"
    assert len(_warnings(caplog)) == 1


def test_check_completion_store_unreachable_is_an_error(tmp_path, caplog):
    runner = _ScriptedRunner(2, _verdict_json("error", None, error="store_unreachable"))

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.ERROR
    assert completion.fires is False
    assert "2" in completion.reason
    assert "store_unreachable" in completion.reason
    assert completion.reason.endswith("-> N/A")
    assert len(_warnings(caplog)) == 1


def test_check_completion_garbage_stdout_is_an_error(tmp_path, caplog):
    runner = _ScriptedRunner(1, "Traceback (most recent call last):\n  boom\n")

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.ERROR
    assert completion.reason.endswith("-> N/A")
    assert len(_warnings(caplog)) == 1


def test_check_completion_exit_2_without_a_verdict_line_is_an_error_not_filed_nothing(
    tmp_path, caplog,
):
    """A usage error also exits 2 but prints no JSON verdict, so exit 2 alone
    never means the run filed nothing."""
    runner = _ScriptedRunner(2, "")

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.ERROR
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "raised",
    [subprocess.TimeoutExpired(["check_run_completion"], 120), OSError("no such interpreter")],
    ids=["timeout", "oserror"],
)
def test_check_completion_a_raising_runner_is_an_error_and_never_raises(
    tmp_path, caplog, raised,
):
    runner = _ScriptedRunner(raises=raised)

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path)

    assert completion.status is ct.CompletionStatus.ERROR
    assert type(raised).__name__ in completion.reason
    assert len(_warnings(caplog)) == 1


def test_check_completion_without_a_previous_run_id_never_runs_the_check(tmp_path, caplog):
    runner = _ScriptedRunner(0, _verdict_json("pass", 1.0))

    with caplog.at_level(logging.WARNING):
        completion = _check(runner, tmp_path, run_id=None)

    assert completion.status is ct.CompletionStatus.NO_PREVIOUS_RUN
    assert completion.fires is False
    assert "previous run id" in completion.reason
    assert completion.reason.endswith("-> N/A")
    assert runner.calls == []
    assert len(_warnings(caplog)) == 1


def _closed_port_url():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def _no_live_fused_memory(monkeypatch):
    """A decision that reaches the real completion check unasked meets a
    closed port, never the host's live fused-memory; a test serving a stub
    sets FUSED_MEMORY_MCP_URL itself."""
    monkeypatch.setenv("FUSED_MEMORY_MCP_URL", _closed_port_url())


class _RunStore:
    """A find_tasks_by_metadata backend serving *rows* for every lookup, and
    recording each lookup's arguments."""

    def __init__(self, rows):
        self.rows = rows
        self.lookups = []

    def __call__(self, name, arguments):
        assert name == "find_tasks_by_metadata"
        self.lookups.append(dict(arguments))
        return {
            "results": self.rows,
            "pagination": {"has_more": False},
            "tasks_with_key": len(self.rows),
        }


def _landed_run_rows():
    return [
        {"task_id": 1, "priority": "high", "status": "done"},
        {"task_id": 2, "priority": "high", "status": "done"},
        {"task_id": 3, "priority": "low", "status": "pending"},
    ]


def test_run_completion_check_runs_the_real_script_against_the_observed_project(
    tmp_path, monkeypatch,
):
    store = _RunStore(_landed_run_rows())

    with StubMcpServer(stateful=False, tools=store) as server:
        monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
        completion = _check(ct.run_completion_check, tmp_path)

    assert completion.status is ct.CompletionStatus.FIRE, completion.reason
    assert store.lookups
    assert {lookup["key"] for lookup in store.lookups} == {"x_finding_run"}
    assert {lookup["value"] for lookup in store.lookups} == {RUN_ID}
    assert {lookup["project_root"] for lookup in store.lookups} == {str(tmp_path.resolve())}


def test_run_completion_check_against_a_closed_port_is_a_store_unreachable_error(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("FUSED_MEMORY_MCP_URL", _closed_port_url())

    completion = _check(ct.run_completion_check, tmp_path)

    assert completion.status is ct.CompletionStatus.ERROR
    assert "store_unreachable" in completion.reason


# ---------------------------------------------------------------------------
# the trigger decides on completion (a) and the novelty spike (b), both under
# the floor; max-interval and tasks-landed are retired
# (plans/census-incremental-prd.md C7, row 13)
# ---------------------------------------------------------------------------

def _evaluate_completion(completion, *, watermark_days_ago=30):
    return ct.evaluate(
        now=NOW,
        floor_anchor=_anchor(ct.FloorAnchorKind.SESSION_WATERMARK, watermark_days_ago),
        completion=completion,
        candidate_first_seens=_quiet_history(),
        config=ct.CensusConfig(),
    )


def test_evaluate_a_firing_completion_fires_clear_of_the_floor():
    decision = _evaluate_completion(_FIRING)

    assert decision.fire is True
    assert decision.completion is _FIRING


@pytest.mark.parametrize("completion", [_NOT_YET, _ERRORED], ids=["not-yet", "error"])
def test_evaluate_a_concluded_or_failed_non_firing_completion_does_not_fire(completion):
    decision = _evaluate_completion(completion)

    assert decision.fire is False
    assert decision.completion is completion


def test_evaluate_the_floor_blocks_a_firing_completion():
    decision = _evaluate_completion(_FIRING, watermark_days_ago=3)

    assert decision.fire is False
    assert decision.reasons[-1] == (
        "floor: 3.0d since session watermark (floor 5d) -> BLOCKS all conditions"
    )


def test_evaluate_reasons_are_completion_novelty_floor_and_nothing_retired():
    decision = _evaluate_completion(_NOT_YET)

    completion_line, novelty_line, floor_line = decision.reasons
    assert completion_line == _NOT_YET.reason
    assert novelty_line.startswith("novelty-spike:")
    assert floor_line.startswith("floor:")
    assert not any(
        r.startswith(("max-interval", "tasks-landed")) for r in decision.reasons
    )


_PREVIOUS_RUN = "census-p-20261001"


def _completion_state(**overrides):
    return {
        "last_census_at": (NOW - timedelta(days=30)).isoformat(),
        "last_census_report": "plans/confusion-census-prior.md",
        "last_census_run_id": _PREVIOUS_RUN,
        "session_watermark": (NOW - timedelta(days=30)).isoformat(),
        **overrides,
    }


def test_decide_for_project_fires_when_the_previous_run_has_landed(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(0, _verdict_json("pass", 0.9))

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.fire is True
    [args] = runner.calls
    assert args[:2] == ["--run", _PREVIOUS_RUN]
    assert decision.completion.status is ct.CompletionStatus.FIRE


def test_decide_for_project_does_not_fire_while_the_previous_run_is_landing(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(75, _verdict_json("not_yet", 0.5))

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert f"completion: 0.5 of {_PREVIOUS_RUN} (threshold 0.70)" in decision.reasons


def test_decide_for_project_does_not_fire_when_the_check_fails(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(2, _verdict_json("error", None, error="store_unreachable"))

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert any(
        r.startswith("completion:") and r.endswith("-> N/A") for r in decision.reasons
    )


def test_decide_for_project_a_stale_census_without_a_run_id_no_longer_fires(tmp_path):
    """The old max-interval FIRE case: 30 days since the last census and the
    floor clear. Elapsed time alone fires nothing any more."""
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state(last_census_run_id=None))
    runner = _ScriptedRunner(0, _verdict_json("pass", 1.0))

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert decision.completion.status is ct.CompletionStatus.NO_PREVIOUS_RUN
    assert runner.calls == []
    assert not any(r.startswith("max-interval") for r in decision.reasons)


def test_decide_for_project_a_never_censused_project_has_no_previous_run(tmp_path):
    _write_codebook(tmp_path, entries=[_entry_sighted("2026-06-01")])
    runner = _ScriptedRunner(0, _verdict_json("pass", 1.0))

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.completion.status is ct.CompletionStatus.NO_PREVIOUS_RUN
    assert runner.calls == []


def _entry_sighted(day):
    return {
        "id": "entry-a",
        "title": "t",
        "severity": "low",
        "status": "open",
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
        "sightings": [_sighting(day)],
    }


def _real_now_state(**overrides):
    month_ago = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    return {
        "last_census_at": month_ago,
        "last_census_report": "plans/confusion-census-prior.md",
        "session_watermark": month_ago,
        **overrides,
    }


def test_cli_evaluate_without_a_previous_run_prints_a_completion_na_line(tmp_path, capsys):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_real_now_state())

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert any(
        line.startswith("completion:") and line.endswith("-> N/A")
        for line in out.splitlines()
    ), out
    assert "max-interval" not in out


def test_cli_evaluate_prints_the_previous_runs_completion_share(
    tmp_path, capsys, monkeypatch,
):
    """The task's user-observable signal, through the real default runner:
    2 high done and 1 low pending weigh 6 landed of 7."""
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_real_now_state(last_census_run_id=_PREVIOUS_RUN))

    with StubMcpServer(stateful=False, tools=_RunStore(_landed_run_rows())) as server:
        monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
        exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"completion: 0.8571 of {_PREVIOUS_RUN} (threshold 0.70) -> FIRE" in out
    assert "DECISION: FIRE" in out


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
    _write_spike_codebook(tmp_path)
    # No census-state.json written -- never censused.

    decision = ct.decide_for_project(tmp_path, now=NOW)

    assert decision.fire is True
    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )
    assert (
        "floor: 35.5d since earliest codebook date (never censused) (floor 5d)"
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

    decision = ct.decide_for_project(tmp_path, now=NOW)

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

    decision = ct.decide_for_project(tmp_path, now=NOW)

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


def _passing_runner():
    return _ScriptedRunner(0, _verdict_json("pass", 0.9))


def _landing_runner():
    return _ScriptedRunner(75, _verdict_json("not_yet", 0.5))


def test_decide_for_project_floor_reads_the_session_watermark_not_last_census_at(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=12, watermark_days_ago=3)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=_passing_runner())

    assert decision.completion.fires is True
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

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=_passing_runner())

    assert decision.fire is True
    assert any(r.startswith("completion:") and r.endswith("-> FIRE") for r in decision.reasons)
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

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=_passing_runner())

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

    decision = ct.decide_for_project(tmp_path, now=NOW)

    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )
    assert decision.fire is False
    assert (
        "floor: no anchor (no session watermark, last_census_at or codebook date)"
        " -> BLOCKS all conditions" in decision.reasons
    )


def _write_spike_codebook(project_root, today=_TODAY):
    _write_codebook(
        project_root,
        candidates=_candidates_from_dates(
            [fs.date().isoformat() for fs in _first_seens_over_a_baseline_of_40(90, today)]
        ),
    )


def test_decide_for_project_row4_novelty_spike_fires(tmp_path):
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=6)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=_landing_runner())

    assert decision.fire is True
    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )


def test_decide_for_project_row5_floor_blocks_novelty_spike_no_fire(tmp_path):
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=3)
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=_landing_runner())

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

    runner = _ScriptedRunner(0, _verdict_json("pass", 1.0))

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert decision.completion is None
    assert runner.calls == []
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_decide_for_project_survives_a_quoted_completion_threshold(tmp_path, caplog):
    """A quoted `completion_threshold_pct: '70'` in a hand-edited
    legibility.yaml is rejected alone, with one config WARNING, and the
    evaluation still decides on the default threshold."""
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    _write_legibility_yaml(tmp_path, "census:\n  completion_threshold_pct: '70'\n")
    runner = _passing_runner()

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(tmp_path, now=NOW, completion_runner=runner)

    assert isinstance(decision, ct.Decision)
    assert decision.fire is True
    assert f"completion: 0.9 of {_PREVIOUS_RUN} (threshold 0.70) -> FIRE" in decision.reasons

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "completion_threshold_pct" in message
    assert "str" in message


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
    today = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
    _write_spike_codebook(tmp_path, today=today)

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION: FIRE" in captured.out
    assert any(
        line.startswith("novelty-spike:") and line.endswith("-> FIRE")
        for line in captured.out.splitlines()
    ), captured.out


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

