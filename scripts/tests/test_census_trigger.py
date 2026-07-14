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
from datetime import datetime, timedelta, timezone

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# step-7: RED — evaluate() condition (c): novelty spike
# ---------------------------------------------------------------------------

def _evaluate_c(*, candidate_first_seens):
    """last_census_at=now-6d: > floor(5), < max_interval(10), < min_days(7)
    -- so only condition (c) can possibly fire. first_seen values are
    passed as YYYY-MM-DD date strings, exactly as the codebook writes them
    (§7.1 candidates[].first_seen), parsed to datetimes here (evaluate()
    itself takes already-parsed datetimes)."""
    return ct.evaluate(
        now=NOW,
        last_census_at=NOW - timedelta(days=6),
        never_censused=False,
        tasks_landed=None,
        candidate_first_seens=[datetime.fromisoformat(s) for s in candidate_first_seens],
        config=ct.CensusConfig(),
    )


def test_evaluate_condition_c_four_within_72h_fires():
    # 2026-07-14/13/12 (twice) are 0h/0h/36h/60h before NOW -- all <= 72h.
    decision = _evaluate_c(
        candidate_first_seens=["2026-07-14", "2026-07-14", "2026-07-13", "2026-07-12"]
    )
    assert decision.fire is True
    assert any("novelty-spike" in r and "4" in r for r in decision.reasons)


def test_evaluate_condition_c_only_three_within_72h_no_fire():
    decision = _evaluate_c(candidate_first_seens=["2026-07-14", "2026-07-13", "2026-07-12"])
    assert decision.fire is False


def test_evaluate_condition_c_four_but_one_outside_window_counts_as_three_no_fire():
    # 2026-07-11 is 84h before NOW -- outside the 72h window, so only 3
    # of these 4 candidates count.
    decision = _evaluate_c(
        candidate_first_seens=["2026-07-14", "2026-07-13", "2026-07-12", "2026-07-11"]
    )
    assert decision.fire is False


# ---------------------------------------------------------------------------
# step-9: RED — hard floor (blocks a/b/c) + never-censused exemption
# ---------------------------------------------------------------------------

_SPIKE_4_IN_72H = ["2026-07-14", "2026-07-14", "2026-07-13", "2026-07-12"]


def test_evaluate_floor_blocks_spike_when_censused():
    decision = ct.evaluate(
        now=NOW,
        last_census_at=NOW - timedelta(days=4),
        never_censused=False,
        tasks_landed=None,
        candidate_first_seens=[datetime.fromisoformat(s) for s in _SPIKE_4_IN_72H],
        config=ct.CensusConfig(),
    )
    assert decision.fire is False
    assert any("floor" in r for r in decision.reasons)


def test_evaluate_floor_blocks_tasks_landed_and_spike_when_censused():
    decision = ct.evaluate(
        now=NOW,
        last_census_at=NOW - timedelta(days=4),
        never_censused=False,
        tasks_landed=999,
        candidate_first_seens=[datetime.fromisoformat(s) for s in _SPIKE_4_IN_72H],
        config=ct.CensusConfig(),
    )
    assert decision.fire is False
    assert any("floor" in r for r in decision.reasons)


def test_evaluate_floor_exempt_when_never_censused():
    decision = ct.evaluate(
        now=NOW,
        last_census_at=NOW - timedelta(days=2),
        never_censused=True,
        tasks_landed=None,
        candidate_first_seens=[datetime.fromisoformat(s) for s in _SPIKE_4_IN_72H],
        config=ct.CensusConfig(),
    )
    assert decision.fire is True


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
    assert data["last_census_at"] == "2026-07-01T00:00:00+00:00"
    assert data["last_census_report"] == "plans/confusion-census-2026-07-01.md"
    assert data["last_census_done_count"] == 500
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


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


def test_default_status_fetcher_raises_status_fetch_unavailable_when_unreachable(tmp_path):
    fetcher = ct.default_status_fetcher(tmp_path)

    with pytest.raises(ct.StatusFetchUnavailable):
        fetcher()


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


def test_decide_for_project_row2_day9_no_spike_low_delta_no_fire(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=9)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
        last_census_done_count=500,
    )
    fetcher = lambda: {"statuses": _done_statuses(550)}  # delta 50 < 120

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
    fetcher = lambda: {"statuses": _done_statuses(630)}  # delta 130 >= 120

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=fetcher)

    assert decision.fire is True
    assert any("tasks-landed" in r for r in decision.reasons)


def test_decide_for_project_row4_day6_novelty_spike_fires(tmp_path):
    _write_codebook(tmp_path, candidates=_candidates_from_dates(_SPIKE_4_IN_72H))
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=6)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is True
    assert any("novelty-spike" in r for r in decision.reasons)


def test_decide_for_project_row5_day4_floor_blocks_spike_no_fire(tmp_path):
    _write_codebook(tmp_path, candidates=_candidates_from_dates(_SPIKE_4_IN_72H))
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=4)).isoformat(),
        last_census_report="plans/confusion-census-prior.md",
    )

    decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    assert any("floor" in r for r in decision.reasons)


def test_decide_for_project_row6_malformed_state_no_fire_one_warning(tmp_path, caplog):
    _write_codebook(tmp_path)
    legibility_dir = tmp_path / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    (legibility_dir / "census-state.json").write_text("not json{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        decision = ct.decide_for_project(tmp_path, now=NOW, status_fetcher=None)

    assert decision.fire is False
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


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
    twelve_days_ago = (datetime.now(timezone.utc) - timedelta(days=12)).strftime("%Y-%m-%d")
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
