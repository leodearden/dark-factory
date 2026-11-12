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
from legibility.config import Census, LegibilityConfig, NoveltySpike, load_config
from shared.testing_mcp_stub import StubAnswer, StubMcpServer
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
        config=config or Census(),
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
        [_TODAY] * 4, config=Census(novelty_spike=NoveltySpike(baseline_days=0))
    )
    assert decision.fire is True
    assert "baseline median 0" in _novelty_line(decision)


def test_novelty_spike_multiple_is_honoured():
    decision = _evaluate_novelty(
        _first_seens_over_a_baseline_of_40(90),
        config=Census(novelty_spike=NoveltySpike(multiple=3)),
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
        config=config or Census(),
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
# step-13: RED — codebook_signal()
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


# ---------------------------------------------------------------------------
# extract_statuses() — the validated {task id: status} mapping census.py's
# outcome feed reads once per run (census_run.py::_read_statuses).
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


# ---------------------------------------------------------------------------
# `bounded_repr` / `bounded_keys` (task 4085): every journal line they
# feed stays one bounded line. These pin that PROPERTY -- output bounded,
# elision marked, true size still reported -- not a private format string.
# The two constants are read from the module, so retuning them does not
# produce a false failure.
# ---------------------------------------------------------------------------

def test_bounded_repr_bounds_an_arbitrarily_large_value():
    """The point of the helper: census-state.json and a get_statuses payload
    are both arbitrarily large, and neither may dump itself into a journal
    line. A value small enough to print in full is left alone."""
    assert "repr truncated" not in ct.bounded_repr({"n": 1})

    value = "x" * 100_000
    result = ct.bounded_repr(value)

    # Bounded, marked as elided, and the TRUE size still stated -- that number
    # is how an operator tells "slightly over" from "a whole payload".
    assert len(result) < ct.MAX_REPR_CHARS + 100
    assert "repr truncated" in result
    assert str(len(repr(value))) in result


def test_bounded_keys_bounds_an_arbitrarily_wide_mapping():
    """Same property for the key list: a get_statuses payload over a big
    project has thousands of keys, and the shape is all diagnosis needs."""
    count = ct.MAX_REPORTED_KEYS + 5
    mapping = {f"k{i:02d}": i for i in range(count)}

    result = ct.bounded_keys(mapping)

    shown = sum(1 for i in range(count) if f"'k{i:02d}'" in result)
    assert shown == ct.MAX_REPORTED_KEYS
    # The suffix reports the TRUE key count, not the number shown.
    assert str(count) in result


def test_bounded_keys_survives_mixed_type_keys():
    """The invariant the `key=str` sort exists to protect. A bare
    `sorted({1: ..., "b": ..., None: ...})` raises TypeError comparing int to
    str -- i.e. the error path would fail while reporting a failure, which is
    the worst possible place for it. get_statuses payloads are server-supplied,
    so a mixed-key dict is exactly the malformed shape this reports on."""
    result = ct.bounded_keys({1: "a", "b": "c", None: "d"})

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


def _check(runner, project_root, *, run_id: str | None = RUN_ID, threshold_pct=70):
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


@pytest.mark.parametrize(
    "status, run_id, extra",
    [
        (ct.CompletionStatus.FIRE, None, {"share": 0.8}),
        (ct.CompletionStatus.NO_PREVIOUS_RUN, RUN_ID, {}),
        (ct.CompletionStatus.FILED_NOTHING, RUN_ID, {"share": 0.5}),
        (ct.CompletionStatus.ERROR, RUN_ID, {"vacuous": True, "detail": "exit 2, tool_error"}),
        (ct.CompletionStatus.ERROR, RUN_ID, {}),
        (ct.CompletionStatus.NOT_YET, RUN_ID, {"share": 0.5, "detail": "exit 75"}),
    ],
    ids=[
        "fire-without-run-id",
        "no-previous-run-with-run-id",
        "filed-nothing-with-share",
        "error-with-vacuous",
        "error-without-detail",
        "not-yet-with-detail",
    ],
)
def test_a_completion_refuses_a_field_its_status_does_not_use(status, run_id, extra):
    with pytest.raises(ValueError, match=str(status)):
        ct.Completion(status, run_id, 70, **extra)


@pytest.mark.parametrize(
    "completion",
    [
        _FIRING,
        _NOT_YET,
        _ERRORED,
        ct.Completion(ct.CompletionStatus.FIRE, RUN_ID, 70, vacuous=True),
        ct.Completion(ct.CompletionStatus.NO_PREVIOUS_RUN, None, 70),
        ct.Completion(ct.CompletionStatus.FILED_NOTHING, RUN_ID, 70),
    ],
    ids=lambda completion: completion.status,
)
def test_a_well_formed_completion_states_its_reason(completion):
    assert completion.reason.startswith("completion: ")


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


def test_run_completion_check_on_a_run_that_filed_nothing_is_filed_nothing(
    tmp_path, monkeypatch,
):
    """Pins the cross-process contract FILED_NOTHING hangs on: the real
    script's error code for a run no task carries, not a copy of it."""
    store = _RunStore([])

    with StubMcpServer(stateful=False, tools=store) as server:
        monkeypatch.setenv("FUSED_MEMORY_MCP_URL", server.url)
        completion = _check(ct.run_completion_check, tmp_path)

    assert completion.status is ct.CompletionStatus.FILED_NOTHING, completion.reason
    assert store.lookups


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
        config=Census(),
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
        "last_census_at": (NOW - timedelta(days=30)).date().isoformat(),
        "last_census_report": "plans/confusion-census-prior.md",
        "last_census_run_id": _PREVIOUS_RUN,
        "session_watermark": (NOW - timedelta(days=30)).isoformat(),
        **overrides,
    }


def test_decide_for_project_fires_when_the_previous_run_has_landed(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(0, _verdict_json("pass", 0.9))

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

    assert decision.fire is True
    [args] = runner.calls
    assert args[:2] == ["--run", _PREVIOUS_RUN]
    assert decision.completion is not None
    assert decision.completion.status is ct.CompletionStatus.FIRE


def test_decide_for_project_does_not_fire_while_the_previous_run_is_landing(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(75, _verdict_json("not_yet", 0.5))

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert f"completion: 0.5 of {_PREVIOUS_RUN} (threshold 0.70)" in decision.reasons


def test_decide_for_project_does_not_fire_when_the_check_fails(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_completion_state())
    runner = _ScriptedRunner(2, _verdict_json("error", None, error="store_unreachable"))

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert decision.completion is not None
    assert decision.completion.status is ct.CompletionStatus.NO_PREVIOUS_RUN
    assert runner.calls == []
    assert not any(r.startswith("max-interval") for r in decision.reasons)


def test_decide_for_project_a_never_censused_project_has_no_previous_run(tmp_path):
    _write_codebook(tmp_path, entries=[_entry_sighted("2026-06-01")])
    runner = _ScriptedRunner(0, _verdict_json("pass", 1.0))

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

    assert decision.completion is not None
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
    month_ago = datetime.now(UTC) - timedelta(days=30)
    return {
        "last_census_at": month_ago.date().isoformat(),
        "last_census_report": "plans/confusion-census-prior.md",
        "session_watermark": month_ago.isoformat(),
        **overrides,
    }


def test_cli_evaluate_without_a_previous_run_prints_a_completion_na_line(tmp_path, capsys):
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_real_now_state())

    _write_config(tmp_path)
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
    _write_config(tmp_path)

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


def _config_yaml(project_root, census_yaml=""):
    return (
        "project_id: p\n"
        f"project_root: {project_root}\n"
        "escalation_port: 1\n"
        f"cwd_prefixes: [{project_root}]\n"
        f"{census_yaml}\n"
    )


def _write_config(project_root, census_yaml=""):
    """A minimal valid docs/legibility/legibility.yaml for *project_root*."""
    legibility_dir = project_root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    path = legibility_dir / "legibility.yaml"
    path.write_text(_config_yaml(project_root, census_yaml), encoding="utf-8")
    return path


def _cfg(project_root, **census_overrides):
    return LegibilityConfig(
        project_id="p",
        project_root=str(project_root),
        escalation_port=1,
        cwd_prefixes=[str(project_root)],
        census=Census(**census_overrides),
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


def test_decide_for_project_row1_missing_state_fires_from_earliest_sighting(tmp_path):
    _write_spike_codebook(tmp_path)
    # No census-state.json written -- never censused.

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW)

    assert decision.fire is True
    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )
    assert (
        "floor: 35.5d since earliest codebook date (never censused) (floor 5d)"
        in decision.reasons
    )


def test_decide_for_project_legacy_state_floor_falls_back_to_last_census_at(tmp_path):
    _write_codebook(tmp_path)
    _write_census_state(
        tmp_path,
        last_census_at=(NOW - timedelta(days=3)).date().isoformat(),
        last_census_report="plans/confusion-census-prior.md",
    )

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW)

    assert decision.fire is False
    assert (
        "floor: 3.5d since last_census_at (no watermark yet) (floor 5d) -> BLOCKS all conditions"
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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW)

    assert decision.fire is False
    floor_lines = [r for r in decision.reasons if r.startswith("floor:")]
    assert len(floor_lines) == 1
    assert "since earliest codebook date (never censused)" in floor_lines[0]
    assert "BLOCKS" in floor_lines[0]


def _watermarked_state(*, last_census_days_ago, watermark_days_ago) -> dict[str, str | None]:
    return {
        "last_census_at": (NOW - timedelta(days=last_census_days_ago)).date().isoformat(),
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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=_passing_runner())

    assert decision.completion is not None
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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=_passing_runner())

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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=_passing_runner())

    assert decision.fire is True
    assert "floor: 12.5d since last_census_at (no watermark yet) (floor 5d)" in decision.reasons


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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW)

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

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=_landing_runner())

    assert decision.fire is True
    assert any(
        r.startswith("novelty-spike:") and r.endswith("-> FIRE") for r in decision.reasons
    )


def test_decide_for_project_row5_floor_blocks_novelty_spike_no_fire(tmp_path):
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=3)
    )

    decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=_landing_runner())

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
        decision = ct.decide_for_project(_cfg(tmp_path), now=NOW, completion_runner=runner)

    assert decision.fire is False
    assert decision.completion is None
    assert runner.calls == []
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_decide_for_project_reads_the_thresholds_it_is_handed_never_the_yaml_on_disk(
    tmp_path,
):
    """Task 3269 / PRD D8: the trickle pins the config_path it resolved, so
    the trigger must decide on the config it is handed, never re-read one."""
    _write_config(tmp_path, "census: {floor_days: 100}")
    _write_spike_codebook(tmp_path)
    _write_census_state(
        tmp_path, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=6)
    )

    decision = ct.decide_for_project(
        _cfg(tmp_path, floor_days=0), now=NOW, completion_runner=_landing_runner()
    )

    (floor_line,) = [r for r in decision.reasons if r.startswith("floor:")]
    assert "(floor 0d)" in floor_line
    assert decision.fire is True


def test_a_retired_key_leaves_the_trigger_output_unchanged(tmp_path, caplog):
    """PRD boundary row 13: a retired census key is ignored with one warning
    at load_config, and changes nothing the trigger decides."""
    project_root = tmp_path / "project"
    _write_spike_codebook(project_root)
    _write_census_state(
        project_root, **_watermarked_state(last_census_days_ago=6, watermark_days_ago=6)
    )
    plain_path = tmp_path / "plain.yaml"
    plain_path.write_text(_config_yaml(project_root, "census: {floor_days: 5}"), encoding="utf-8")
    retired_path = tmp_path / "retired.yaml"
    retired_path.write_text(
        _config_yaml(project_root, "census: {floor_days: 5, max_interval_days: 1}"),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        plain = load_config(plain_path)
        retired = load_config(retired_path)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "max_interval_days" in warnings[0]

    plain_decision = ct.decide_for_project(plain, now=NOW, completion_runner=_landing_runner())
    retired_decision = ct.decide_for_project(
        retired, now=NOW, completion_runner=_landing_runner()
    )
    assert retired_decision.reasons == plain_decision.reasons
    assert retired_decision.fire == plain_decision.fire


# ---------------------------------------------------------------------------
# step-19: RED — CLI `evaluate` subcommand: exits 1 only when the config
# does not load; an evaluation error is a NO-FIRE
# ---------------------------------------------------------------------------

def test_cli_evaluate_dark_factory_like_project_prints_no_fire_and_exits_0(tmp_path, capsys):
    # Mirrors the live dark_factory codebook today: candidates: [], no
    # dated sightings, no census-state.json.
    _write_codebook(tmp_path)

    _write_config(tmp_path)
    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION: NO-FIRE" in captured.out
    assert len(captured.out.strip().splitlines()) > 1  # decision line + >=1 reason line


def test_cli_evaluate_fire_inducing_fixture_prints_fire_and_exits_0(tmp_path, capsys):
    today = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
    _write_spike_codebook(tmp_path, today=today)

    _write_config(tmp_path)
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

    _write_config(tmp_path)
    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DECISION:" in captured.out


def _cli_floor_line(tmp_path, capsys):
    _write_config(tmp_path)
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



def test_cli_evaluate_prints_the_validated_thresholds(tmp_path, capsys):
    config_path = _write_config(tmp_path, "census: {floor_days: 7, completion_threshold_pct: 85}")
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_real_now_state())

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    (config_line,) = [line for line in out.splitlines() if line.startswith("config: ")]
    assert str(config_path) in config_line
    assert "floor_days=7" in config_line
    assert "completion_threshold_pct=85" in config_line
    (floor_line,) = [line for line in out.splitlines() if line.startswith("floor:")]
    assert "(floor 7d)" in floor_line


def test_cli_evaluate_rejects_a_malformed_census_value(tmp_path, capsys):
    config_path = _write_config(tmp_path, "census: {floor_days: -1}")
    _write_codebook(tmp_path)

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(config_path) in captured.err
    assert "floor_days" in captured.err
    assert "input_value=-1" in captured.err
    assert "DECISION" not in captured.out


def test_cli_evaluate_reads_the_config_that_config_names(tmp_path, capsys):
    project_root = tmp_path / "project"
    _write_codebook(project_root)
    _write_census_state(project_root, **_real_now_state())
    pinned = tmp_path / "pinned" / "legibility.yaml"
    pinned.parent.mkdir()
    pinned.write_text(_config_yaml(project_root, "census: {floor_days: 7}"), encoding="utf-8")

    exit_code = ct.main(
        ["evaluate", "--project-root", str(project_root), "--config", str(pinned)]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    (config_line,) = [line for line in out.splitlines() if line.startswith("config: ")]
    assert str(pinned) in config_line
    assert "floor_days=7" in config_line


def test_cli_evaluate_refuses_a_config_naming_another_project_root(tmp_path, capsys):
    """A worktree's legibility.yaml names the main checkout: evaluating it
    would read one tree's state against the other's config."""
    worktree = tmp_path / "worktree"
    main_checkout = tmp_path / "main"
    main_checkout.mkdir()
    _write_codebook(worktree)
    config_path = worktree / "docs" / "legibility" / "legibility.yaml"
    config_path.write_text(_config_yaml(main_checkout), encoding="utf-8")

    exit_code = ct.main(["evaluate", "--project-root", str(worktree)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(worktree) in captured.err
    assert str(main_checkout) in captured.err
    assert str(config_path) in captured.err
    assert "DECISION" not in captured.out


def test_cli_evaluate_defaults_the_project_root_to_the_cwd(tmp_path, capsys, monkeypatch):
    _write_config(tmp_path)
    _write_codebook(tmp_path)
    _write_census_state(tmp_path, **_real_now_state())
    monkeypatch.chdir(tmp_path)

    exit_code = ct.main(["evaluate"])

    assert exit_code == 0
    assert "DECISION:" in capsys.readouterr().out


def test_cli_evaluate_without_a_config_exits_1_naming_the_expected_path(tmp_path, capsys):
    _write_codebook(tmp_path)

    exit_code = ct.main(["evaluate", "--project-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(tmp_path / "docs" / "legibility" / "legibility.yaml") in captured.err
    assert "DECISION" not in captured.out
