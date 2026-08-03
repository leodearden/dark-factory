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
from pathlib import Path

import pytest
import yaml

from legibility import census_trigger as ct
from legibility.config import Census as LegibilityCensus

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
    # tuple[str, dict | None] — None only for the missing/malformed statuses.
    assert data is not None
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


def test_default_status_fetcher_raises_status_fetch_unavailable_when_unreachable(
    tmp_path, install_fake_httpx
):
    """An unreachable endpoint must surface as StatusFetchUnavailable, never
    as a raw transport exception.

    The transport failure is INJECTED rather than relied upon. This test
    originally passed on two ambient premises, both of which are now false:
    that httpx was not importable here (so the lazy-import ImportError branch
    fired), and that nothing listened on the default FUSED_MEMORY_MCP_URL
    (http://localhost:8002). httpx is installed today, and a real fused-memory
    MCP server listens on :8002 on any machine running the stack -- so the
    test flapped pass/fail with that live server's connection state instead of
    testing this module. Injecting a failing `httpx` module (via the shared
    `install_fake_httpx` fixture, as
    test_default_status_fetcher_sends_streamable_http_accept_headers below
    also does) makes the "unreachable" premise true by construction.
    """
    def _fake_post(url, **kwargs):
        raise OSError("[Errno 111] Connection refused")

    install_fake_httpx(_fake_post)

    fetcher = ct.default_status_fetcher(tmp_path)

    with pytest.raises(ct.StatusFetchUnavailable):
        fetcher()


class _FakeHttpxResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_default_status_fetcher_sends_streamable_http_accept_headers(
    tmp_path, install_fake_httpx
):
    """Task 2953: the streamable-HTTP MCP transport 406s
    ("Not Acceptable: Client must accept application/json") any tools/call
    POST whose Accept header doesn't include both application/json and
    text/event-stream -- verified live against a local MCP /mcp endpoint.
    default_status_fetcher's httpx import is lazy, but httpx is genuinely
    available here: a DIRECT dependency of `shared` (shared/pyproject.toml,
    `httpx>=0.27`, task 2965), not a transitive one. So the real POST would
    actually go out; the shared `install_fake_httpx` fixture substitutes a
    stub to capture the outbound call without a network, and without
    depending on whatever is listening on localhost:8002."""
    captured_kwargs = {}
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": {"statuses": {"1": "done"}}},
    }

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse(rpc_response)

    install_fake_httpx(_fake_post)

    fetcher = ct.default_status_fetcher(tmp_path)
    result = fetcher()

    assert result == {"statuses": {"1": "done"}}
    headers = captured_kwargs.get("headers") or {}
    assert "application/json" in headers.get("Accept", "")
    assert "text/event-stream" in headers.get("Accept", "")
    # Content-Type is part of the same transport contract -- pin it too so a
    # future edit dropping it can't pass on the Accept assertions alone.
    assert headers.get("Content-Type") == "application/json"
    # The JSON-RPC tools/call body must still ride along on the same call.
    envelope = captured_kwargs.get("json") or {}
    assert envelope.get("method") == "tools/call"
    assert envelope.get("params", {}).get("name") == "get_statuses"


def _capture_get_statuses_project_root(install_fake_httpx, project_root):
    """Drive `default_status_fetcher(project_root)` against a fake httpx and
    return the `project_root` argument it actually put on the wire. Reuses
    the `_FakeHttpxResponse` + shared `install_fake_httpx` harness above (the
    `import httpx` inside `_fetch` is lazy precisely so this works).

    Takes the fixture as a parameter rather than requesting it: this is a
    plain helper, not a test, so pytest will not inject fixtures into it."""
    captured_kwargs = {}

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse(
            {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": {"statuses": {}}}}
        )

    install_fake_httpx(_fake_post)

    ct.default_status_fetcher(project_root)()
    return captured_kwargs["json"]["params"]["arguments"]["project_root"]


def test_default_status_fetcher_sends_absolute_project_root_for_dot(
    tmp_path, monkeypatch, install_fake_httpx
):
    """Task 3291 root cause. fused-memory's `_normalize_project_root` hard-
    rejects ANY relative path -- verified live against localhost:8002, which
    answered `{"project_root": "."}` with
    `{"error": "project_root must be a non-empty absolute path, got: '.'",
      "error_type": "ValidationError"}`.

    In production that call ALWAYS carried a relative path: census.py's CLI
    defaults `--project-root` to `"."`, and `nightly._default_census_launcher`
    (nightly.py:521) launches census.py with no arguments at all. The MCP
    argument is resolved by the SERVER's cwd, not the client's, so a relative
    path is meaningless over the wire."""
    monkeypatch.chdir(tmp_path)

    sent = _capture_get_statuses_project_root(install_fake_httpx, ".")

    assert sent == str(tmp_path.resolve())
    assert Path(sent).is_absolute()


def test_default_status_fetcher_sends_absolute_project_root_for_relative_subdir(
    tmp_path, monkeypatch, install_fake_httpx
):
    """Second case so the fix cannot pass by special-casing `"."` alone."""
    (tmp_path / "sub" / "dir").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    sent = _capture_get_statuses_project_root(install_fake_httpx, "sub/dir")

    assert sent == str((tmp_path / "sub" / "dir").resolve())
    assert Path(sent).is_absolute()


def test_default_status_fetcher_leaves_absolute_project_root_unchanged(
    tmp_path, monkeypatch, install_fake_httpx
):
    """The other half of the contract: an ALREADY-absolute root must cross the
    wire byte-for-byte unchanged. Both tests above start from a relative path,
    so on their own they would not notice a "fix" that mangles absolute inputs.

    This matters concretely because fused-memory keys its `_MAIN_CHECKOUT_CACHE`
    on the path it is handed: if `resolve()` were ever to rewrite an operator's
    configured root (a symlinked home, say `/home/leo` -> elsewhere), the cache
    key would silently stop matching the configured one. Pinning pass-through
    for the already-absolute case is what keeps this fix a normalisation of
    relative paths rather than a rewrite of every path."""
    monkeypatch.chdir(tmp_path)
    absolute_root = str(tmp_path)

    sent = _capture_get_statuses_project_root(install_fake_httpx, absolute_root)

    assert sent == absolute_root


# ---------------------------------------------------------------------------
# amendment pass (review findings #1/#2): _extract_tool_result() unwraps the
# real MCP tools/call JSON-RPC envelope. default_status_fetcher's HTTP
# round-trip is never driven against a real endpoint here -- the tests above
# inject a fake `httpx` module instead, since anything else makes them depend
# on whether a live fused-memory MCP server happens to be listening on the
# default URL; these tests instead exercise the envelope parser
# directly against realistic tools/call response shapes, and then bridge its
# output into compute_tasks_landed to pin the exact contract between them.
# ---------------------------------------------------------------------------

def _done_statuses_envelope(count):
    return {str(i): "done" for i in range(count)}


def test_extract_tool_result_prefers_structured_content():
    # structuredContent and content disagree on purpose, so a returned value
    # equal to structuredContent's (not content's) proves precedence.
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"statuses": {"decoy": "done"}})}],
            "structuredContent": {"statuses": {"1": "done"}},
            "isError": False,
        },
    }
    assert ct._extract_tool_result(rpc_response) == {"statuses": {"1": "done"}}


def test_extract_tool_result_falls_back_to_content_text_when_no_structured_content():
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"statuses": {"1": "done"}})}],
            "isError": False,
        },
    }
    assert ct._extract_tool_result(rpc_response) == {"statuses": {"1": "done"}}


def test_extract_tool_result_raises_when_no_result_key():
    # e.g. a JSON-RPC error response, which carries "error" instead of "result".
    rpc_response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}}
    with pytest.raises(ct.StatusFetchUnavailable):
        ct._extract_tool_result(rpc_response)


def test_extract_tool_result_raises_when_content_text_unparseable():
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "not json"}]},
    }
    with pytest.raises(ct.StatusFetchUnavailable):
        ct._extract_tool_result(rpc_response)


def test_extract_tool_result_feeds_compute_tasks_landed_correct_delta():
    """Pins the exact shape contract between a realistic MCP tools/call
    envelope (as get_statuses actually returns it) and
    compute_tasks_landed -- closing the gap where only the hand-rolled
    _wrapped_fetcher test double (which bypasses envelope parsing entirely)
    was ever exercised. Before the fix, this delta silently came out
    negative (0 - 500) instead of 130."""
    statuses = _done_statuses_envelope(630)
    statuses.update({str(i): "pending" for i in range(630, 645)})
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"statuses": statuses})}],
            "structuredContent": {"statuses": statuses},
            "isError": False,
        },
    }
    state = {"last_census_done_count": 500}

    result = ct.compute_tasks_landed(
        state=state, status_fetcher=lambda: ct._extract_tool_result(rpc_response)
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
