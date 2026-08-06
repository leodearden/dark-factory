"""Tests for scripts/legibility/census.py — the periodic legibility census
runner (task eta, plans/confusion-reduction-prd.md §5.7, contract §7,
decisions 4/5/6/9, boundary test §8.7).

Import the target module as `import census as mod`, resolved via
scripts/tests/conftest.py's sys.path insertion (both scripts/ and
scripts/legibility/ are on sys.path; no package __init__ needed) — mirrors
test_legibility_coder.py's import style.

Every LLM / MCP / git side effect (invoke, verify_fn, synthesize_fn,
submit_fn, escalate_fn, status_fetcher, commit, batch_source) is ALWAYS a
fake/injected seam in this file — no test ever shells out to a real
`claude` process, hits a real MCP/escalation endpoint, or touches real git
state.
"""
from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import census as mod
import codebook
import coder
import digest as digest_mod
import inventory
import pytest
from legibility import census_trigger
from shared.cap_markers import BLOCKING_BANNER_MARKERS, REAL_CLI_CAP_MESSAGES

import config as config_mod

# ---------------------------------------------------------------------------
# Shared fixture helpers — synthetic transcript -> real digest text, mirrors
# test_legibility_coder.py's helper shapes (own copies, per this repo's
# convention of each test file owning its scaffolding, e.g. test_codebook.py's
# _minimal_v2()).
# ---------------------------------------------------------------------------

_SESSION_ID = "cafe1234-0000-4000-8000-000000000000"
_CWD = "/home/leo/src/dark-factory"
_TIMESTAMP = "2026-07-14T06:02:29.796Z"


def _user_text(s, *, cwd=_CWD, session_id=_SESSION_ID, timestamp=_TIMESTAMP):
    """Build a synthetic genuine (non-meta, non-sidechain) human user turn
    carrying real-transcript-shaped top-level cwd/sessionId/timestamp
    fields (observed on every non-queue-operation record in a real Claude
    Code transcript). Copied from test_legibility_coder.py:43-56."""
    return {
        "type": "user",
        "message": {"role": "user", "content": s},
        "isSidechain": False,
        "isMeta": False,
        "cwd": cwd,
        "sessionId": session_id,
        "timestamp": timestamp,
    }


def _write_jsonl(tmp_path, records, name="transcript.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")
    return path


def _build_digest_text(
    tmp_path, *, session_id=_SESSION_ID, agent_class="interactive", name="transcript.jsonl"
):
    """Build a real digest string via digest.build_digest on a minimal
    synthetic single-user-turn transcript — session/date/agent_class in
    the resulting frontmatter are deterministic from the inputs given
    here. Adapted from test_legibility_coder.py:68-75."""
    records = [_user_text("Please fix this, it is wrong.", session_id=session_id)]
    path = _write_jsonl(tmp_path, records, name=name)
    return digest_mod.build_digest(path, agent_class_override=agent_class)


def _hand_digest(session_id, body_marker, *, date="2026-07-14", agent_class="interactive"):
    """Hand-written minimal digest text: a leading frontmatter block with
    exactly the fields code_digest reads (session/date/agent_class) plus a
    one-line body. Used where a test's focus is not digest.py's rendering
    fidelity — that round trip is separately covered by _build_digest_text.
    Copied from test_legibility_coder.py:78-92."""
    return (
        "---\n"
        f'session: "{session_id}"\n'
        f'date: "{date}"\n'
        f'agent_class: "{agent_class}"\n'
        "---\n\n"
        f"## User Corrections\n- {body_marker}\n"
    )


def _minimal_v2_codebook() -> dict:
    """A minimal well-formed v2 codebook: one entry, no candidates. Mirrors
    test_codebook.py's _minimal_v2()."""
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": "high",
                "status": "open",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [],
            }
        ],
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# Fake seam factories — every LLM/MCP/git side effect census.py performs is
# an injected seam (invoke/verify_fn/synthesize_fn/submit_fn/escalate_fn/
# status_fetcher/commit/batch_source); these are the shared fakes the rest
# of this file's tests build on, so no test ever reaches a real subprocess,
# network call, or model.
# ---------------------------------------------------------------------------

def _make_fake_invoke(response_fn=None, *, default="{}"):
    """Fake `invoke(prompt, model) -> str` seam (the one-shot LLM-call
    primitive used for Sonnet mining via coder.code_digests(invoke=...)
    and the tiny headroom probe). Records every call as a dict in
    `.calls` (prompt/model) so a test can assert routing/content. When
    `response_fn(prompt, model)` is given it decides the reply per call;
    otherwise every call returns `default` verbatim."""
    calls = []

    def fake_invoke(prompt, model):
        calls.append({"prompt": prompt, "model": model})
        if response_fn is not None:
            return response_fn(prompt, model)
        return default

    fake_invoke.calls = calls
    return fake_invoke


def _make_fake_submit_fn(*, id_prefix="task"):
    """Fake curator-path `submit_fn(**kwargs) -> dict` seam. Records every
    call's kwargs in `.calls` and returns an incrementing fake task id."""
    calls = []

    def fake_submit_fn(**kwargs):
        calls.append(kwargs)
        return {"id": f"{id_prefix}-{len(calls)}"}

    fake_submit_fn.calls = calls
    return fake_submit_fn


def _make_fake_escalate_fn():
    """Fake info-escalation `escalate_fn(**kwargs) -> dict` seam. Records
    every call's kwargs in `.calls` and returns a fake escalation id."""
    calls = []

    def fake_escalate_fn(**kwargs):
        calls.append(kwargs)
        return {"id": f"escalation-{len(calls)}"}

    fake_escalate_fn.calls = calls
    return fake_escalate_fn


def _make_fake_status_fetcher(done_count):
    """Fake `status_fetcher() -> dict` seam, shaped like the real
    get_statuses envelope census_trigger.compute_tasks_landed already
    reads (`{"statuses": {id: status}}`, done count = count of "done"
    values) so census.py's own done-count baseline logic mirrors zeta's."""
    def fake_status_fetcher():
        return {"statuses": {f"t{i}": "done" for i in range(done_count)}}

    return fake_status_fetcher


class _TrackingBatchSource:
    """Iterable `batch_source` seam wrapper around a plain list of
    pre-built batches (each batch itself a list of digest-text strings).
    Records which batch indices were actually pulled in `.pulled`, so a
    test can assert that batches after a saturation/exhaustion stop point
    were never even consumed from the source — not merely unused by the
    caller. A plain generator can't carry this bookkeeping (generator
    objects accept no arbitrary attributes), hence the small class."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.pulled = []

    def __iter__(self):
        for i, batch in enumerate(self._batches):
            self.pulled.append(i)
            yield batch


def _poison(name):
    """A seam fake that raises if ever called -- used to prove a code path
    (e.g. run_census's DEFER branch) never reaches a given seam."""
    def _fn(*args, **kwargs):
        raise AssertionError(f"{name} must never be called on this path")

    return _fn


# ---------------------------------------------------------------------------
# step-1: RED — is_duplicate() / batch_dup_rate()
# ---------------------------------------------------------------------------

def _record(*, matches=None, candidates=None):
    return {"matches": matches or [], "candidates": candidates or []}


def test_is_duplicate_false_when_record_has_a_candidate():
    record = _record(candidates=[{"title": "a novel shape"}])
    assert mod.is_duplicate(record) is False


def test_is_duplicate_true_for_matches_only_record():
    record = _record(matches=[{"entry_id": "entry-a"}])
    assert mod.is_duplicate(record) is True


def test_is_duplicate_true_for_empty_record():
    record = _record()
    assert mod.is_duplicate(record) is True


def test_batch_dup_rate_nine_of_ten_duplicates():
    records = [_record(matches=[{"entry_id": "entry-a"}]) for _ in range(9)]
    records.append(_record(candidates=[{"title": "novel"}]))
    assert mod.batch_dup_rate(records) == pytest.approx(0.9)


def test_batch_dup_rate_empty_batch_is_zero_no_zero_division():
    assert mod.batch_dup_rate([]) == 0.0


# ---------------------------------------------------------------------------
# step-3: RED — mine_to_saturation() batch loop + saturation stop
# ---------------------------------------------------------------------------

def _batch_digests(n, prefix):
    return [_hand_digest(f"{prefix}-{i}", f"body {i}") for i in range(n)]


def _mining_response_fn(novel_sessions):
    """Fake coder-LLM reply chooser: any digest whose frontmatter session id
    is in `novel_sessions` gets a one-candidate (novel) judgment; every
    other digest gets a matches-only (duplicate) judgment. Mirrors
    test_legibility_coder.py's _make_batch_invoke session-substring idiom,
    adapted to decide dup-vs-novel instead of success-vs-failure."""
    def _fn(prompt, model):
        for session_id in novel_sessions:
            if session_id in prompt:
                return json.dumps(
                    {"matches": [], "candidates": [{"title": f"novel shape {session_id}"}]}
                )
        return json.dumps({"matches": [{"entry_id": "entry-a"}], "candidates": []})

    return _fn


def test_mine_to_saturation_stops_after_two_consecutive_saturated_batches():
    live_codebook = _minimal_v2_codebook()
    batch0 = _batch_digests(10, "b0")  # 5/10 novel -> dup_rate 0.5 (not saturated)
    batch1 = _batch_digests(10, "b1")  # 1/10 novel -> dup_rate 0.9 (saturated #1)
    batch2 = _batch_digests(10, "b2")  # 1/10 novel -> dup_rate 0.9 (saturated #2, stop)
    batch3 = _batch_digests(10, "b3")  # must never be pulled from the source
    novel_sessions = {f"b0-{i}" for i in range(5)} | {"b1-0"} | {"b2-0"}
    source = _TrackingBatchSource([batch0, batch1, batch2, batch3])
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(_mining_response_fn(novel_sessions))

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke,
    )

    assert result.stop_reason == "saturated"
    assert source.pulled == [0, 1, 2], "batch3 must never be consumed from the source"
    assert len(result.batch_stats) == 3
    assert result.batch_stats[0].dup_rate == pytest.approx(0.5)
    assert result.batch_stats[1].dup_rate == pytest.approx(0.9)
    assert result.batch_stats[2].dup_rate == pytest.approx(0.9)
    assert [s.total for s in result.batch_stats] == [10, 10, 10]
    assert [s.succeeded for s in result.batch_stats] == [10, 10, 10]
    assert [s.failed for s in result.batch_stats] == [0, 0, 0]
    assert all(s.status == "ok" for s in result.batch_stats), "no storms in this fixture"
    # every successfully-coded record across all 3 consumed batches accumulates
    assert len(result.records) == 30
    assert fake_invoke.calls, "the fake invoke must actually have been reached"
    assert all(call["model"] == "sonnet" for call in fake_invoke.calls)


def test_mine_to_saturation_exhausts_source_that_never_saturates():
    live_codebook = _minimal_v2_codebook()
    # Every digest is novel (dup_rate 0.0 always) so saturation is never hit.
    batches = [_batch_digests(4, f"e{i}") for i in range(3)]
    novel_sessions = {f"e{i}-{j}" for i in range(3) for j in range(4)}
    source = _TrackingBatchSource(batches)
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(_mining_response_fn(novel_sessions))

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke,
    )

    assert result.stop_reason == "exhausted"
    assert source.pulled == [0, 1, 2]
    assert len(result.batch_stats) == 3
    assert all(s.dup_rate == pytest.approx(0.0) for s in result.batch_stats)
    assert len(result.records) == 12


# ---------------------------------------------------------------------------
# amend: mine_to_saturation() must never let a storm batch (coder.RunResult
# .status == "failure", >50% coding failures, PRD §8.6) satisfy saturation —
# a short run of storms must not be able to trip the saturation stop on a
# handful of skewed, degraded-sample records (reviewer_comprehensive finding
# #1).
# ---------------------------------------------------------------------------

def _mining_response_fn_with_failures(novel_sessions=(), fail_sessions=()):
    """Like `_mining_response_fn`, but any digest whose session id is in
    `fail_sessions` gets a reply that fails to parse as JSON at all -- that
    digest's `code_digest` call comes back `ok=False` (a coding failure),
    so a batch with enough `fail_sessions` members trips coder.code_digests'
    own storm threshold (failed/total > 0.5)."""
    def _fn(prompt, model):
        for session_id in fail_sessions:
            if session_id in prompt:
                return "this is not JSON and will fail to parse"
        for session_id in novel_sessions:
            if session_id in prompt:
                return json.dumps(
                    {"matches": [], "candidates": [{"title": f"novel shape {session_id}"}]}
                )
        return json.dumps({"matches": [{"entry_id": "entry-a"}], "candidates": []})

    return _fn


def test_mine_to_saturation_storm_batch_never_counts_as_saturated():
    live_codebook = _minimal_v2_codebook()
    batch0 = _batch_digests(10, "b0")  # 1/10 novel -> dup_rate 0.9 (real saturation #1)
    # storm: 6/10 digests fail to parse (60% > 50% -> RunResult.status="failure");
    # of the 4 that DO succeed, every one is a duplicate -> dup_rate 1.0 on paper,
    # but this must NOT count as saturated, and must reset the counter to 0.
    batch1 = _batch_digests(10, "b1")
    batch2 = _batch_digests(10, "b2")  # 1/10 novel -> dup_rate 0.9 (saturated #1, post-reset)
    batch3 = _batch_digests(10, "b3")  # 1/10 novel -> dup_rate 0.9 (saturated #2, stop)
    batch4 = _batch_digests(10, "b4")  # must never be pulled from the source

    novel_sessions = {"b0-0", "b2-0", "b3-0"}
    fail_sessions = {f"b1-{i}" for i in range(6)}
    source = _TrackingBatchSource([batch0, batch1, batch2, batch3, batch4])
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(
        _mining_response_fn_with_failures(novel_sessions, fail_sessions)
    )

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke,
    )

    assert result.stop_reason == "saturated"
    assert source.pulled == [0, 1, 2, 3], "batch4 must never be consumed from the source"
    assert len(result.batch_stats) == 4

    assert result.batch_stats[0].saturated is True
    assert result.batch_stats[0].status == "ok"

    storm = result.batch_stats[1]
    assert storm.total == 10
    assert storm.succeeded == 4
    assert storm.failed == 6
    assert storm.status == "failure", "RunResult.status must be surfaced on BatchStats"
    assert storm.dup_rate == pytest.approx(1.0), "dup_rate is still recorded for visibility"
    assert storm.saturated is False, "a storm batch must never satisfy saturation"

    # post-storm-reset: TWO fresh consecutive real-saturated batches are needed to stop
    assert result.batch_stats[2].saturated is True
    assert result.batch_stats[2].status == "ok"
    assert result.batch_stats[3].saturated is True
    assert result.batch_stats[3].status == "ok"


# ---------------------------------------------------------------------------
# task 3280 step-1: RED — mine_to_saturation(max_batches=) bounds mining with
# a DISTINGUISHABLE stop reason. The operator batch cap is enforced inside the
# mining loop (not by islicing the source) precisely so "capped" can never be
# confused with "exhausted" — a source that genuinely ran dry and a run the
# operator bounded are different coverage claims, and conflating them is the
# silent-cap failure this flag exists to avoid.
# ---------------------------------------------------------------------------

def _never_saturating_source(n_batches, *, size=4):
    """A `_TrackingBatchSource` of *n_batches* all-novel batches (dup_rate
    0.0 every batch, so saturation is never reached) plus the matching
    `novel_sessions` set — the fixture for testing a stop that can only
    come from the cap or from exhaustion, never from saturation."""
    batches = [_batch_digests(size, f"c{i}") for i in range(n_batches)]
    novel_sessions = {f"c{i}-{j}" for i in range(n_batches) for j in range(size)}
    return _TrackingBatchSource(batches), novel_sessions


def test_mine_to_saturation_stops_at_operator_batch_cap():
    live_codebook = _minimal_v2_codebook()
    source, novel_sessions = _never_saturating_source(6)
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(_mining_response_fn(novel_sessions))

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke, max_batches=2,
    )

    assert result.stop_reason == "capped", "the cap must be distinguishable from exhaustion"
    assert len(result.batch_stats) == 2
    # The whole point of enforcing the cap INSIDE the loop: batches 2..5 were
    # never even pulled from the source, so no digest was rendered and no
    # coder.code_digests/LLM call was spent on them.
    assert source.pulled == [0, 1], "capped-away batches must never be consumed"
    assert len(result.records) == 8, "records from both mined batches accumulate"
    assert [s.index for s in result.batch_stats] == [0, 1]


def test_mine_to_saturation_cap_not_reached_leaves_stop_reason_unchanged():
    live_codebook = _minimal_v2_codebook()
    source, novel_sessions = _never_saturating_source(6)
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(_mining_response_fn(novel_sessions))

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke, max_batches=99,
    )

    assert result.stop_reason == "exhausted", "a cap never reached must not relabel the stop"
    assert source.pulled == [0, 1, 2, 3, 4, 5]
    assert len(result.batch_stats) == 6


def test_mine_to_saturation_saturation_at_the_cap_reports_saturated_not_capped():
    live_codebook = _minimal_v2_codebook()
    # batch0 saturates (#1), batch1 saturates (#2 -> stop) AND is exactly the
    # capped batch. Saturation is the stronger claim (novelty genuinely
    # exhausted, so coverage was sufficient regardless of the cap), so it is
    # checked first and must win.
    batch0 = _batch_digests(10, "b0")  # 1/10 novel -> dup_rate 0.9
    batch1 = _batch_digests(10, "b1")  # 1/10 novel -> dup_rate 0.9
    batch2 = _batch_digests(10, "b2")  # must never be pulled
    source = _TrackingBatchSource([batch0, batch1, batch2])
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)
    fake_invoke = _make_fake_invoke(_mining_response_fn({"b0-0", "b1-0"}))

    result = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=fake_invoke, max_batches=2,
    )

    assert result.stop_reason == "saturated", (
        "saturation on exactly the capped batch must report the stronger reason"
    )
    assert source.pulled == [0, 1]
    assert len(result.batch_stats) == 2


def test_mine_to_saturation_records_max_batches_on_result():
    live_codebook = _minimal_v2_codebook()
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)

    source, novel_sessions = _never_saturating_source(3)
    capped = mod.mine_to_saturation(
        source, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=_make_fake_invoke(_mining_response_fn(novel_sessions)),
        max_batches=2,
    )
    assert capped.max_batches == 2, "the cap must travel on the result for the report"

    source2, novel_sessions2 = _never_saturating_source(3)
    flagless = mod.mine_to_saturation(
        source2, live_codebook, project="dark_factory", model="sonnet",
        config=saturation, invoke=_make_fake_invoke(_mining_response_fn(novel_sessions2)),
    )
    assert flagless.max_batches is None, "no cap passed -> nothing to report"
    assert flagless.stop_reason == "exhausted"


@pytest.mark.parametrize("bad_cap", [0, -1, -50])
def test_mine_to_saturation_rejects_a_nonpositive_batch_cap(bad_cap):
    # A cap of 0 or less cannot be honored: the cap is checked AFTER a batch
    # is coded, so max_batches=0 would still spend one full coder.code_digests
    # call and then render "mined 1 batch(es); operator batch cap = 0". For a
    # flag whose whole contract is "no silent caps", a nonsense value must
    # fail loud rather than half-apply.
    live_codebook = _minimal_v2_codebook()
    source, _ = _never_saturating_source(3)
    saturation = config_mod.Saturation(dup_rate=0.9, consecutive_batches=2)

    with pytest.raises(ValueError, match="max_batches"):
        mod.mine_to_saturation(
            source, live_codebook, project="dark_factory", model="sonnet",
            config=saturation, invoke=_poison("invoke"), max_batches=bad_cap,
        )

    assert source.pulled == [], "the guard must fire before any batch is consumed"


# ---------------------------------------------------------------------------
# step-5: RED — compute_matrix() / render_matrix() origin x manifestation
# ---------------------------------------------------------------------------

def test_compute_matrix_tallies_origin_manifestation_pairs():
    sightings = [
        {"origin_phase": "implement", "manifested_phase": "merge"},
        {"origin_phase": "implement", "manifested_phase": "merge"},
        {"origin_phase": "implement", "manifested_phase": "verify"},
        {"origin_phase": "review", "manifested_phase": "merge"},
    ]
    matrix = mod.compute_matrix(sightings)
    assert matrix["implement"]["merge"] == 2
    assert matrix["implement"]["verify"] == 1
    assert matrix["review"]["merge"] == 1
    assert "unknown" not in matrix, "no unknown sighting present -> no unknown bucket"


def test_compute_matrix_unknown_origin_or_manifested_lands_in_explicit_unknown_bucket():
    sightings = [
        {"origin_phase": "unknown", "manifested_phase": "merge"},
        {"origin_phase": "implement", "manifested_phase": "unknown"},
        {"origin_phase": None, "manifested_phase": None},  # absent entirely -> unknown/unknown
    ]
    matrix = mod.compute_matrix(sightings)
    assert matrix["unknown"]["merge"] == 1
    assert matrix["implement"]["unknown"] == 1
    assert matrix["unknown"]["unknown"] == 1
    # never inferred to a concrete phase -- the only rows are the ones actually seen
    assert set(matrix.keys()) == {"unknown", "implement"}


def test_compute_matrix_empty_sightings_is_empty_matrix():
    assert mod.compute_matrix([]) == {}


def test_render_matrix_deterministic_table_with_unknown_row_col_when_present():
    sightings = [
        {"origin_phase": "implement", "manifested_phase": "merge"},
        {"origin_phase": "implement", "manifested_phase": "merge"},
        {"origin_phase": "unknown", "manifested_phase": "merge"},
    ]
    matrix = mod.compute_matrix(sightings)
    rendered = mod.render_matrix(matrix)

    lines = rendered.rstrip("\n").split("\n")
    assert lines[0] == "| origin \\ manifested | merge |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| implement | 2 |"
    assert lines[3] == "| unknown | 1 |"
    assert len(lines) == 4
    # deterministic: same input renders byte-identical output every time
    assert rendered == mod.render_matrix(mod.compute_matrix(sightings))


def test_render_matrix_omits_unknown_when_no_unknown_sightings():
    sightings = [{"origin_phase": "implement", "manifested_phase": "merge"}]
    matrix = mod.compute_matrix(sightings)
    rendered = mod.render_matrix(matrix)
    assert "unknown" not in rendered


def test_render_matrix_empty_matrix_is_deterministic_placeholder():
    assert mod.render_matrix({}) == "_No sightings recorded._\n"


# ---------------------------------------------------------------------------
# step-7: RED — preflight_headroom() tiny probe
# ---------------------------------------------------------------------------

def test_preflight_headroom_ok_on_normal_reply():
    fake_invoke = _make_fake_invoke(default="pong")
    result = mod.preflight_headroom(fake_invoke, model="sonnet")
    assert result.ok is True
    assert fake_invoke.calls[0]["model"] == "sonnet"


@pytest.mark.parametrize(
    "banner",
    [
        "You have reached your usage limit for this period.",
        "Rate limit exceeded, please try again later.",
        "Please run /login to authenticate.",
        "Invalid API key provided.",
    ],
)
def test_preflight_headroom_defers_on_known_banner(banner):
    fake_invoke = _make_fake_invoke(default=banner)
    result = mod.preflight_headroom(fake_invoke, model="sonnet")
    assert result.ok is False
    assert result.reason


def test_preflight_headroom_banner_match_is_case_insensitive():
    fake_invoke = _make_fake_invoke(default="USAGE LIMIT REACHED, TRY AGAIN LATER")
    result = mod.preflight_headroom(fake_invoke, model="sonnet")
    assert result.ok is False


def test_preflight_headroom_invocation_error_defers_fail_safe():
    def raising_invoke(prompt, model):
        raise coder.CoderInvocationError(
            "claude CLI exited 1 (model='sonnet'): simulated backend outage"
        )

    result = mod.preflight_headroom(raising_invoke, model="sonnet")
    assert result.ok is False
    assert result.reason


# ---------------------------------------------------------------------------
# task 3645 / DEFECT 1: the probe's marker list must cover the cap text the
# CLI ACTUALLY emits, not just the four strings this module happened to
# start with.
#
# The corpus is IMPORTED from shared.cap_markers rather than restated here.
# That is the whole point: the marker list is a CONTRACT checked against real
# CLI transcripts at both consuming sites, so a future cap rewording is one
# corpus edit that turns this suite AND shared/tests/test_cap_markers.py red
# until the markers cover it — instead of the situation this task found, where
# the same blind spot had to be discovered independently at two sites.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", REAL_CLI_CAP_MESSAGES)
def test_preflight_headroom_defers_on_every_real_cli_cap_message(message):
    """Every verbatim real-CLI cap message defers, naming the matched marker.

    Fails today on the weekly-limit, "You've used all available credits" and
    "You're out of extra usage" messages: none of them contains any of the
    four original markers, so a weekly cap sailed through the preflight and
    every subsequent verify call was fail-closed rejected as an ordinary
    verdict.
    """
    fake_invoke = _make_fake_invoke(default=message)
    result = mod.preflight_headroom(fake_invoke, model="sonnet")
    assert result.ok is False, f"real CLI cap message passed the probe: {message!r}"
    assert result.reason
    # The reason must quote the marker that fired, so an operator reading the
    # log can tell WHICH signal tripped rather than just that something did.
    assert "banner marker" in result.reason


def test_preflight_headroom_defers_on_ticket_weekly_limit_phrasing():
    """The exact phrasing quoted in task 3645, with a hyphen separator.

    The corpus carries the middot spelling (the first-hand transcript); the
    ticket quotes the hyphen one.  Both are pinned because the separator is
    incidental punctuation the CLI has already varied, and a guard that
    depends on it is one release away from silence.
    """
    fake_invoke = _make_fake_invoke(
        default="You've hit your weekly limit - resets Aug 5, 11am"
    )
    result = mod.preflight_headroom(fake_invoke, model="sonnet")
    assert result.ok is False
    assert result.reason


# ---------------------------------------------------------------------------
# step-9: RED — build_task_payloads() curator-path submit_task payloads
# ---------------------------------------------------------------------------

def _verified_cluster(**overrides):
    cluster = {
        "title": "Silent no-op subagent contract",
        "summary": "Agents were given contracts their runtime could not honor.",
        "evidence": ["quote from a transcript"],
        "severity": "high",
        "sightings": [
            {"origin_phase": "implement", "manifested_phase": "verify", "session": "sess-1"},
        ],
    }
    cluster.update(overrides)
    return cluster


def test_build_task_payloads_one_payload_per_cluster_curator_path():
    clusters = [_verified_cluster(title="Cluster A"), _verified_cluster(title="Cluster B")]
    payloads = mod.build_task_payloads(
        clusters, project_root="/home/leo/src/some-project", project_id="some_project",
    )

    assert len(payloads) == 2
    for payload in payloads:
        assert payload.get("task_kind", "normal") == "normal"
        assert "planning_mode" not in payload, "curator dedup path -- never planning_mode"
        assert payload["project_root"] == "/home/leo/src/some-project"
        assert payload["title"]
        assert payload["description"]


def test_build_task_payloads_title_and_description_sourced_from_cluster():
    cluster = _verified_cluster(
        title="A specific novel cause", summary="A precise factual summary of what happened.",
    )
    payloads = mod.build_task_payloads([cluster], project_root="/root", project_id="proj")
    assert "A specific novel cause" in payloads[0]["title"]
    assert "A precise factual summary of what happened." in payloads[0]["description"]


def test_build_task_payloads_description_has_no_prose_routing_intent():
    clusters = [_verified_cluster()]
    payloads = mod.build_task_payloads(clusters, project_root="/root", project_id="proj")
    description = payloads[0]["description"].lower()
    for banned_phrase in ("file this into", "route to", "planning_mode"):
        assert banned_phrase not in description


def test_build_task_payloads_harness_rooted_cause_can_target_dark_factory():
    clusters = [
        _verified_cluster(
            title="Orchestrator confusion",
            target_project_root="/home/leo/src/dark-factory",
            target_project_id="dark_factory",
        )
    ]
    payloads = mod.build_task_payloads(
        clusters, project_root="/home/leo/src/hosted-project", project_id="hosted_project",
    )
    assert payloads[0]["project_root"] == "/home/leo/src/dark-factory"


# ---------------------------------------------------------------------------
# amend round 2: target_project_root/target_project_id move together, never
# independently (reviewer_comprehensive finding #2).
# ---------------------------------------------------------------------------

def test_build_task_payloads_partial_target_override_falls_back_to_own_project():
    only_root = [
        _verified_cluster(
            title="Only root override", target_project_root="/home/leo/src/dark-factory",
        )
    ]
    payloads = mod.build_task_payloads(
        only_root, project_root="/home/leo/src/hosted-project", project_id="hosted_project",
    )
    assert payloads[0]["project_root"] == "/home/leo/src/hosted-project", (
        "a lone target_project_root override must be ignored, never mixed with the "
        "census's own project_id"
    )
    assert "hosted_project" in payloads[0]["description"]

    only_id = [_verified_cluster(title="Only id override", target_project_id="dark_factory")]
    payloads = mod.build_task_payloads(
        only_id, project_root="/home/leo/src/hosted-project", project_id="hosted_project",
    )
    assert payloads[0]["project_root"] == "/home/leo/src/hosted-project", (
        "a lone target_project_id override must be ignored, never mixed with the "
        "census's own project_root"
    )
    assert "hosted_project" in payloads[0]["description"]


# ---------------------------------------------------------------------------
# amend: _novel_clusters() dedup-by-title + titleless-candidate skip.
# codebook.apply_coding_record groups new candidates BY TITLE (codebook.py:
# 494), so verification must operate on the same set of resolvable titles --
# a second cluster for an already-seen title can never resolve to a second
# "pending" candidate id in _find_pending_candidate_id, wasting a Sonnet
# verify call and diverging the matrix from the codebook
# (reviewer_comprehensive finding #3).
# ---------------------------------------------------------------------------

def _novel_record(session, *, candidates):
    return {"session": session, "matches": [], "candidates": candidates}


def test_novel_clusters_deduplicates_candidates_sharing_a_title():
    records = [
        _novel_record(
            "sess-1",
            candidates=[
                {
                    "title": "Silent no-op subagent contract",
                    "cause": "first sighting",
                    "origin_phase": "implement",
                    "manifested_phase": "verify",
                }
            ],
        ),
        _novel_record(
            "sess-2",
            candidates=[
                {
                    "title": "Silent no-op subagent contract",
                    "cause": "second sighting, same title",
                    "origin_phase": "implement",
                    "manifested_phase": "verify",
                }
            ],
        ),
    ]

    clusters = mod._novel_clusters(records)

    assert len(clusters) == 1, (
        "a second candidate sharing an already-seen title must not spend a "
        "second verify slot -- apply_coding_record would collapse it into "
        "the same pending codebook candidate as the first"
    )
    assert clusters[0]["title"] == "Silent no-op subagent contract"
    assert clusters[0]["sightings"][0]["session"] == "sess-1", "first occurrence wins"


def test_novel_clusters_skips_titleless_candidates():
    records = [
        _novel_record("sess-1", candidates=[{"title": "", "cause": "empty title"}]),
        _novel_record("sess-2", candidates=[{"cause": "title key entirely absent"}]),
    ]

    clusters = mod._novel_clusters(records)

    assert clusters == [], (
        "apply_coding_record requires a title to key its merge on -- a "
        "titleless candidate must be skipped, mirroring that requirement"
    )


def test_novel_clusters_distinct_titles_each_get_their_own_cluster():
    records = [
        _novel_record("sess-1", candidates=[{"title": "Cause A"}]),
        _novel_record("sess-2", candidates=[{"title": "Cause B"}]),
    ]

    clusters = mod._novel_clusters(records)

    assert {c["title"] for c in clusters} == {"Cause A", "Cause B"}


# ---------------------------------------------------------------------------
# step-11: RED — promote_candidate() / reject_candidate() / retire_entry()
# in-place codebook lifecycle transforms (never-delete, PRD decision 3)
# ---------------------------------------------------------------------------

def _codebook_with_candidate():
    """Mirrors test_codebook.py's _minimal_v2() base shape, plus one
    pending candidate carrying a real sighting -- the fixture promote/
    reject tests build on."""
    cb = _minimal_v2_codebook()
    cb["candidates"] = [
        {
            "id": "cand-20260714-1",
            "title": "A novel confusion shape",
            "first_seen": "2026-07-14",
            "disposition": "pending",
            "sightings": [
                {
                    "date": "2026-07-14",
                    "project": "dark_factory",
                    "session": "sess-1",
                    "origin_phase": "implement",
                    "manifested_phase": "verify",
                },
            ],
        }
    ]
    return cb


def test_promote_candidate_appends_entry_and_stamps_disposition():
    before = _codebook_with_candidate()
    entry_fields = {
        "id": "new-entry-a",
        "title": "A novel confusion shape",
        "severity": "medium",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "verify",
    }

    result = mod.promote_candidate(before, "cand-20260714-1", entry_fields)

    new_entry = next(e for e in result["entries"] if e["id"] == "new-entry-a")
    assert new_entry["sightings"] == before["candidates"][0]["sightings"]

    candidate = next(c for c in result["candidates"] if c["id"] == "cand-20260714-1")
    assert candidate["disposition"] == "promoted"
    assert candidate.get("promoted_to") == "new-entry-a", "must name the new entry id"
    assert len(result["candidates"]) == len(before["candidates"]), "candidate is RETAINED"

    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)  # must not raise

    # never mutates the input codebook
    assert before["candidates"][0]["disposition"] == "pending"
    assert len(before["entries"]) == 1


def test_reject_candidate_stamps_disposition_retained():
    before = _codebook_with_candidate()

    result = mod.reject_candidate(before, "cand-20260714-1")

    candidate = next(c for c in result["candidates"] if c["id"] == "cand-20260714-1")
    assert candidate["disposition"] == "rejected"
    assert len(result["candidates"]) == len(before["candidates"]), "candidate is RETAINED"

    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)

    assert before["candidates"][0]["disposition"] == "pending"


def test_retire_entry_sets_status_retained():
    before = _minimal_v2_codebook()

    result = mod.retire_entry(before, "entry-a")

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["status"] == "retired"
    assert len(result["entries"]) == len(before["entries"]), "entry is RETAINED"

    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)

    assert before["entries"][0]["status"] == "open"


# ---------------------------------------------------------------------------
# step-13: RED — advance_census_state() (zeta/2579 MUST-persist contract)
# ---------------------------------------------------------------------------

def test_advance_census_state_writes_all_three_fields(tmp_path):
    path = tmp_path / "census-state.json"

    mod.advance_census_state(
        path,
        now_iso="2026-07-14T12:00:00+00:00",
        report_path="plans/confusion-census-2026-07-14.md",
        done_count=42,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "last_census_at": "2026-07-14T12:00:00+00:00",
        "last_census_report": "plans/confusion-census-2026-07-14.md",
        "last_census_done_count": 42,
    }


def test_advance_census_state_done_count_zero_is_written_as_integer_zero(tmp_path):
    path = tmp_path / "census-state.json"

    mod.advance_census_state(
        path, now_iso="2026-07-14T12:00:00+00:00", report_path="plans/x.md", done_count=0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "last_census_done_count" in data, "must never be dropped as falsy"
    assert data["last_census_done_count"] == 0
    assert isinstance(data["last_census_done_count"], int)


def test_advance_census_state_writes_none_done_count_as_json_null(tmp_path):
    """Task 3291: when the done-count could not be OBSERVED, the honest
    baseline is `null` -- not a fabricated 0 (that is the defect being
    fixed), not a carried-forward stale value (a quieter guess), and not an
    omitted key (which would violate the 2579/eta MUST-persist contract).
    `null` honours that contract literally -- the key is still always
    present -- while being truthful about the state being unknown."""
    path = tmp_path / "census-state.json"

    mod.advance_census_state(
        path, now_iso="2026-07-31T12:00:00+00:00", report_path="plans/x.md", done_count=None,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "last_census_done_count" in data, "MUST-persist contract: key is never dropped"
    assert data["last_census_done_count"] is None
    assert '"last_census_done_count": null' in path.read_text(encoding="utf-8")


def test_null_done_count_baseline_makes_condition_b_fail_safe(tmp_path, caplog):
    """End-to-end proof that an UNKNOWN baseline disarms condition (b)
    instead of arming it. A poisoned `0` baseline makes compute_tasks_landed
    return `current_done - 0` -- every done task ever, ~24x over the 120
    threshold -- as soon as the get_statuses fetch works (task 3291; see
    census_trigger's module docstring for the replayed measurements). A
    `null` baseline instead routes into the existing absent-baseline branch:
    one WARNING, `None`, no fire."""
    path = tmp_path / "census-state.json"
    mod.advance_census_state(
        path, now_iso="2026-07-31T12:00:00+00:00", report_path="plans/x.md", done_count=None,
    )

    # A null baseline is UNKNOWN, not malformed -- the state file still loads.
    status, data = census_trigger.load_census_state(path)
    assert status == "ok"
    # load_census_state returns tuple[str, dict | None] — the None is real for
    # the "missing"/"malformed" statuses, so narrow it explicitly rather than
    # subscripting an Optional.
    assert data is not None
    assert data["last_census_done_count"] is None

    with caplog.at_level(logging.WARNING):
        landed = census_trigger.compute_tasks_landed(
            state=data,
            # A perfectly VALID fetcher: the fail-safe here comes from the
            # unknown baseline, not from any fetch problem.
            status_fetcher=lambda: {"statuses": {f"t{i}": "done" for i in range(2870)}},
        )

    assert landed is None
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1

    # ... and with tasks_landed=None, condition (b) cannot fire even when
    # days_since is well past tasks_landed_min_days. Condition (a)
    # (max_interval_days) remains the unconditional backstop.
    config = census_trigger.CensusConfig()
    now = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)
    decision = census_trigger.evaluate(
        now=now,
        last_census_at=now - timedelta(days=config.tasks_landed_min_days + 1),
        never_censused=False,
        tasks_landed=None,
        candidate_first_seens=[],
        config=config,
    )

    assert decision.fire is False


def test_advance_census_state_round_trips_through_census_trigger_load(tmp_path):
    path = tmp_path / "census-state.json"

    mod.advance_census_state(
        path,
        now_iso="2026-07-14T12:00:00+00:00",
        report_path="plans/confusion-census-2026-07-14.md",
        done_count=7,
    )

    status, data = census_trigger.load_census_state(path)
    assert status == "ok"
    assert data is not None  # tuple[str, dict | None]; None only for missing/malformed
    assert data["last_census_at"] == "2026-07-14T12:00:00+00:00"
    assert data["last_census_report"] == "plans/confusion-census-2026-07-14.md"
    assert data["last_census_done_count"] == 7


def test_advance_census_state_atomic_replace_no_partial_left_behind(tmp_path):
    path = tmp_path / "census-state.json"
    path.write_text(json.dumps({"stale": "data"}), encoding="utf-8")

    mod.advance_census_state(
        path, now_iso="2026-07-14T12:00:00+00:00", report_path="plans/x.md", done_count=3,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["last_census_done_count"] == 3
    assert "stale" not in data, "pre-existing state must be FULLY replaced"

    leftovers = [p for p in tmp_path.iterdir() if p.name != "census-state.json"]
    assert leftovers == [], "no partial/temp file left behind"


# ---------------------------------------------------------------------------
# step-15: RED — render_report() dated markdown assembly
# ---------------------------------------------------------------------------

def _sample_mining_result():
    return mod.MiningResult(
        records=[],
        batch_stats=[
            mod.BatchStats(index=0, total=10, succeeded=10, failed=0, dup_rate=0.5, saturated=False),
            mod.BatchStats(index=1, total=10, succeeded=10, failed=0, dup_rate=0.9, saturated=True),
        ],
        stop_reason="saturated",
    )


def test_render_report_contains_dated_header_and_all_sections():
    report = mod.render_report(
        date="2026-07-14",
        project_id="dark_factory",
        force=False,
        matrix_md="| origin \\ manifested | merge |\n| --- | --- |\n| implement | 2 |\n",
        mining_result=_sample_mining_result(),
        synthesis_md="Fable synthesis prose goes here.",
        filed_task_ids=["1234", "1235"],
        cost_note="~$3.42 across 20 Sonnet calls + 1 Fable call.",
    )

    assert "# confusion census 2026-07-14" in report
    assert "dark_factory" in report
    # matrix embedded verbatim
    assert "| origin \\ manifested | merge |\n| --- | --- |\n| implement | 2 |\n" in report
    # saturation-stats section: batch count, per-batch dup rates, stop_reason
    assert "saturated" in report
    assert "2" in report  # batch count
    assert "0.5" in report
    assert "0.9" in report
    # filed task ids
    assert "1234" in report
    assert "1235" in report
    # cost note
    assert "~$3.42 across 20 Sonnet calls + 1 Fable call." in report
    # synthesis prose
    assert "Fable synthesis prose goes here." in report
    # no force marker on a non-forced run
    assert "--force" not in report


def test_render_report_force_marker_present_when_forced():
    report = mod.render_report(
        date="2026-07-14",
        project_id="dark_factory",
        force=True,
        matrix_md="matrix",
        mining_result=_sample_mining_result(),
        synthesis_md="prose",
        filed_task_ids=[],
        cost_note="cost",
    )
    assert "--force" in report
    assert "operator-initiated" in report.lower()


def test_render_report_is_deterministic_no_clock():
    kwargs: dict[str, Any] = dict(
        date="2026-07-14",
        project_id="dark_factory",
        force=False,
        matrix_md="matrix",
        mining_result=_sample_mining_result(),
        synthesis_md="prose",
        filed_task_ids=["1"],
        cost_note="cost",
    )
    assert mod.render_report(**kwargs) == mod.render_report(**kwargs)


_GOLDEN_FLAGLESS_REPORT = """\
# confusion census 2026-07-14

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.50 (total=10, succeeded=10, failed=0, saturated=False)
  - batch 1: dup_rate=0.90 (total=10, succeeded=10, failed=0, saturated=True)

## Origin x Manifestation Matrix

matrix
## Synthesis

prose

## Filed Tasks

- 1

## Cost

cost
"""
"""Byte-for-byte `render_report` output for a FLAGLESS run, captured
verbatim from the module BEFORE task 3280 added the operator cost-control
flags (`--max-batches`, `--max-verify-clusters`, `--dry-run-filing`).

This is a LOCK, not a spec under development: every new report line those
flags introduce must be gated on a non-None flag value, so a run that
passes none of them renders exactly this. Do NOT regenerate this constant
to make a failing run pass -- a diff here means a cost-control rendering
leaked into the unflagged path (and therefore into the nightly trickle,
which launches census.py with no extra argv)."""


def _capped_mining_result(*, stop_reason, max_batches, batches=2):
    return mod.MiningResult(
        records=[],
        batch_stats=[
            mod.BatchStats(
                index=i, total=10, succeeded=10, failed=0, dup_rate=0.1, saturated=False,
            )
            for i in range(batches)
        ],
        stop_reason=stop_reason,
        max_batches=max_batches,
    )


def _render(**overrides):
    # Annotated for the same reason as _run_census_kwargs: a heterogeneous
    # dict whose inferred value union would otherwise be re-reported once per
    # union member per render_report parameter.
    kwargs: dict[str, Any] = dict(
        date="2026-07-14",
        project_id="dark_factory",
        force=False,
        matrix_md="matrix",
        mining_result=_sample_mining_result(),
        synthesis_md="prose",
        filed_task_ids=["1"],
        cost_note="cost",
    )
    kwargs.update(overrides)
    return mod.render_report(**kwargs)


def test_render_report_capped_run_names_cap_and_partial_coverage():
    report = _render(
        mining_result=_capped_mining_result(stop_reason="capped", max_batches=2),
    )

    saturation_section = report.split("## Saturation", 1)[1].split("##", 1)[0]
    lowered = saturation_section.lower()
    assert "20" in saturation_section, "sessions actually mined (2 batches x 10) must be stated"
    # Assert the substantive substring, not a bare "2": the section always
    # carries "- batches: 2" and digit-bearing per-batch bullets, so a bare
    # digit check passes even if the cap VALUE stopped being rendered --
    # exactly the regression this line claims to guard.
    assert "operator batch cap = 2" in saturation_section
    assert "cap" in lowered, "the operator cap must be named, never applied silently"
    # A capped report must be unreadable as full coverage.
    assert "partial" in lowered
    assert "not mined" in lowered


def test_render_report_capped_run_says_the_skipped_sessions_are_not_re_mined():
    # PARTIAL coverage must not read as "the rest gets picked up next time".
    # run_census always advances last_census_at and _census_window_dates
    # anchors the NEXT window there, so the capped-away sessions fall outside
    # every future window -- the same dead-recovery-path hazard the dry-run
    # WARNING is written to avoid.
    report = _render(
        mining_result=_capped_mining_result(stop_reason="capped", max_batches=2),
    )

    saturation_section = report.split("## Saturation", 1)[1].split("##", 1)[0]
    lowered = saturation_section.lower()
    assert "last_census_at" in lowered, "the re-anchoring mechanism must be named"
    assert "next census window starts here" in lowered
    assert "never re-enumerated" in lowered, (
        "the report must say the capped-away sessions are not swept later"
    )
    # ...and must not leave a re-run reading as the recovery path.
    assert "census-state.json" in lowered, "the one real recovery lever is named"


def test_render_report_cap_not_reached_makes_no_re_anchor_claim():
    # The re-anchor disclosure belongs to the CAPPED branch only: a cap that
    # was set but never reached mined exactly what an uncapped run would.
    report = _render(
        mining_result=_capped_mining_result(stop_reason="saturated", max_batches=99),
    )

    saturation_section = report.split("## Saturation", 1)[1].split("##", 1)[0]
    lowered = saturation_section.lower()
    assert "last_census_at" not in lowered
    assert "never re-enumerated" not in lowered


def test_render_report_cap_set_but_not_reached_is_reported_distinctly():
    report = _render(
        mining_result=_capped_mining_result(stop_reason="saturated", max_batches=99),
    )

    saturation_section = report.split("## Saturation", 1)[1].split("##", 1)[0]
    lowered = saturation_section.lower()
    assert "99" in saturation_section, "the cap is still named for the operator's record"
    assert "cap" in lowered
    assert "not reached" in lowered
    # No partial-coverage claim: the run stopped on its own terms.
    assert "partial" not in lowered
    assert "not mined" not in lowered


def test_render_report_no_cap_renders_no_coverage_line():
    report = _render(
        mining_result=_capped_mining_result(stop_reason="exhausted", max_batches=None),
    )

    saturation_section = report.split("## Saturation", 1)[1].split("##", 1)[0]
    lowered = saturation_section.lower()
    assert "cap" not in lowered, "an uncapped run must render no cap text at all"
    assert "coverage" not in lowered
    assert "partial" not in lowered


def test_render_report_verify_cap_states_verified_of_novel_and_deferred():
    report = _render(
        verify_coverage=mod.VerifyCoverage(novel=812, verified=150, cap=150),
    )

    assert "## Verification" in report
    section = report.split("## Verification", 1)[1].split("##", 1)[0]
    assert "150" in section
    assert "812" in section
    assert "662" in section, "the deferred remainder must be stated, not left to arithmetic"
    lowered = section.lower()
    assert "verified 150 of 812" in lowered
    # Deferred means "not yet adjudicated", never "dropped": the deferred
    # clusters still merged into the codebook as pending candidates.
    assert "pending candidate" in lowered
    assert "deferred" in lowered
    assert "dropped" not in lowered
    # "a later census picks it up" is CONDITIONAL: this window's sightings are
    # never re-mined, so a deferred cluster is re-adjudicated only on a
    # recurrence. Say so, exactly as the batch-cap and dry-run paths do.
    assert "recurs" in lowered, "the later-census pickup must be stated as conditional"
    assert "not re-mined" in lowered or "never re-mined" in lowered

    # Placement: between Saturation and the matrix.
    assert report.index("## Saturation") < report.index("## Verification")
    assert report.index("## Verification") < report.index("## Origin x Manifestation Matrix")


def test_render_report_verify_cap_set_but_not_reached_claims_no_deferral():
    # Mirror of the batch cap's "not reached" branch. With novel == verified
    # nothing was deferred and nothing went unverified, so the deferral clause
    # would be a false statement about this run.
    report = _render(verify_coverage=mod.VerifyCoverage(novel=3, verified=3, cap=5))

    assert "## Verification" in report
    section = report.split("## Verification", 1)[1].split("##", 1)[0]
    lowered = section.lower()
    assert "verified all 3 novel cluster" in lowered
    assert "5" in section, "the cap is still named for the operator's record"
    assert "not reached" in lowered
    # Nothing was deferred -- the deferral wording must be absent entirely.
    assert "deferred" not in lowered
    assert "0 deferred" not in lowered
    assert "pending candidate" not in lowered


def test_render_report_without_verify_coverage_renders_no_verification_section():
    assert "## Verification" not in _render()
    assert "## Verification" not in _render(verify_coverage=None)


_PAYLOADS_PATH = "/p/plans/confusion-census-2026-07-30-payloads.json"


def test_render_report_dry_run_filing_section_names_count_and_path():
    report = _render(
        filed_task_ids=[],
        dry_run=mod.DryRunFiling(path=_PAYLOADS_PATH, payload_count=12),
    )

    section = report.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert "dry-run: 12 payload" in section
    assert _PAYLOADS_PATH in section
    assert "_none filed._" not in section


def test_render_report_dry_run_takes_precedence_over_empty_filed_ids():
    # An empty filed list plus a dry run must never read as "a normal run
    # that happened to file nothing" -- the dry-run wording wins.
    report = _render(
        filed_task_ids=[],
        dry_run=mod.DryRunFiling(path=_PAYLOADS_PATH, payload_count=3),
    )

    section = report.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert "_none filed._" not in section
    assert "dry-run" in section.lower()
    assert "nothing filed" in section.lower()


def test_render_report_without_dry_run_filed_tasks_section_unchanged():
    filed = _render(filed_task_ids=["1234", "1235"])
    filed_section = filed.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert "- 1234" in filed_section
    assert "- 1235" in filed_section
    assert "dry-run" not in filed_section.lower()

    none_filed = _render(filed_task_ids=[])
    none_section = none_filed.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert "_none filed._" in none_section
    assert "dry-run" not in none_section.lower()


def test_render_report_flagless_output_is_byte_identical_golden():
    report = mod.render_report(
        date="2026-07-14",
        project_id="dark_factory",
        force=False,
        matrix_md="matrix",
        mining_result=_sample_mining_result(),
        synthesis_md="prose",
        filed_task_ids=["1"],
        cost_note="cost",
    )
    assert report == _GOLDEN_FLAGLESS_REPORT


# ---------------------------------------------------------------------------
# step-17: RED — run_census() DEFER path (headroom banner)
# ---------------------------------------------------------------------------

def _run_census_kwargs(tmp_path, **overrides) -> dict[str, Any]:
    # The return annotation is load-bearing for the type gate, not decoration.
    # This dict is deliberately heterogeneous — injected callables, a
    # LegibilityConfig, Paths, strs, bools, None — so without it pyright infers
    # the value type as a wide union and then re-reports that union once per
    # member per use: at each `mod.run_census(**kwargs)` call site (one error
    # per parameter) and at each `kwargs[...]` attribute access. Measured: this
    # single annotation cleared 337 of the 350 errors this file carried.
    kwargs: dict[str, Any] = dict(
        batch_source=None,
        invoke=_make_fake_invoke(default="pong"),
        verify_fn=_poison("verify_fn"),
        synthesize_fn=_poison("synthesize_fn"),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_make_fake_escalate_fn(),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_poison("commit"),
        codebook_dict=_minimal_v2_codebook(),
        config=config_mod.LegibilityConfig(
            project_id="dark_factory",
            project_root=str(tmp_path),
            escalation_port=8103,
            cwd_prefixes=[str(tmp_path)],
        ),
        project_root=str(tmp_path),
        project_id="dark_factory",
        codebook_path=tmp_path / "confusion-codebook.yaml",
        census_state_path=tmp_path / "census-state.json",
        report_path=tmp_path / "confusion-census-2026-07-14.md",
        date="2026-07-14",
        force=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_census_defers_on_headroom_banner(tmp_path, caplog):
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(default="You have reached your usage limit for this period."),
        batch_source=_poison("batch_source"),
    )
    fake_submit_fn = kwargs["submit_fn"]
    fake_escalate_fn = kwargs["escalate_fn"]

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "deferred"
    assert outcome.reason
    # The two defer sites must be distinguishable by FIELD, not by parsing
    # prose out of a shared reason string (task 3645).
    assert outcome.deferred_stage == "preflight"

    assert len(fake_escalate_fn.calls) == 1
    call = fake_escalate_fn.calls[0]
    assert call.get("category") in ("infra_issue", "risk_identified")
    assert call.get("summary"), "a loud summary naming the deferral"

    assert sum(1 for r in caplog.records if r.levelno >= logging.WARNING) >= 1

    assert fake_submit_fn.calls == []
    assert not kwargs["codebook_path"].exists()
    assert not kwargs["census_state_path"].exists()
    assert not kwargs["report_path"].exists()


# ---------------------------------------------------------------------------
# task 3645 / DEFECT 2: the stage-boundary re-probe.
#
# Headroom was probed exactly ONCE, at preflight. A cap arriving during the
# long, expensive gap between preflight and verify was never noticed -- and
# because the default verifier fails CLOSED per cluster, it then presented as
# an ordinary run in which every cluster happened to be rejected.
# ---------------------------------------------------------------------------

def _capped_after_preflight_invoke(cap_message=REAL_CLI_CAP_MESSAGES[0]):
    """`invoke` that passes the FIRST headroom probe and fails every later one.

    Keys on the probe prompt specifically rather than on call ordinal, because
    mining shares this same seam: mining prompts must keep returning real
    judgments (otherwise there would be no novel cluster to verify, and the
    gate under test would be skipped for the wrong reason).
    """
    probes = []

    def response_fn(prompt, model):
        if prompt == mod._HEADROOM_PROBE_PROMPT:
            probes.append(prompt)
            return "pong" if len(probes) == 1 else cap_message
        return _happy_invoke_response(prompt, model)

    return response_fn


def test_run_census_defers_at_verify_boundary_when_cap_arrives_after_preflight(
    tmp_path, caplog
):
    """A cap arriving after preflight aborts BEFORE verify, persisting nothing.

    The load-bearing assertions are the three `not ... .exists()` ones. They
    prove no matrix was rendered from a truncated verified list, no
    reject_candidate burned a cluster that was never actually adjudicated, and
    -- most importantly -- last_census_at never advanced past a window that
    was not adjudicated, so these sightings ARE re-mined on the next run. The
    mining spend is sunk; the state stays honest.
    """
    batch = [_hand_digest("novel-verified", "a genuinely new confusion shape")]
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_capped_after_preflight_invoke()),
        batch_source=[batch],
        verify_fn=_poison("verify_fn"),
        synthesize_fn=_poison("synthesize_fn"),
        commit=_poison("commit"),
    )
    fake_submit_fn = kwargs["submit_fn"]
    fake_escalate_fn = kwargs["escalate_fn"]

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "deferred"
    assert outcome.deferred_stage == "verify"
    assert outcome.unverified_clusters == 1
    assert outcome.reason
    assert "verify" in outcome.reason.lower(), "the reason names the stage that was skipped"

    # verify_fn is a _poison: reaching it at all would have raised.
    assert fake_submit_fn.calls == []
    assert not kwargs["report_path"].exists()
    assert not kwargs["codebook_path"].exists()
    assert not kwargs["census_state_path"].exists()

    assert len(fake_escalate_fn.calls) == 1
    call = fake_escalate_fn.calls[0]
    assert call.get("category") == "infra_issue"
    assert call.get("severity") == "info"
    summary = call.get("summary") or ""
    detail = call.get("detail") or ""
    assert "verify" in (summary + detail).lower(), "the escalation names the stage"
    assert "1" in (summary + detail), "the escalation names the unverified count"

    assert sum(1 for r in caplog.records if r.levelno >= logging.WARNING) >= 1


# ---------------------------------------------------------------------------
# step-19: RED — run_census() HAPPY path, full seam wiring + static routing
# ---------------------------------------------------------------------------

def _make_fake_verify_fn(*, verified_titles=(), rejected_titles=(), fixed_entry_ids=()):
    """Fake Sonnet `verify_fn(clusters, *, model)` seam. Splits the input
    clusters by title into verified/rejected per the given title sets;
    `fixed_entry_ids` is returned verbatim so a test can exercise the
    retire_entry path independently of any particular cluster. Records
    every call's (clusters, model) in `.calls`."""
    calls = []

    def fake_verify_fn(clusters, *, model):
        calls.append({"clusters": clusters, "model": model})
        return {
            "verified": [c for c in clusters if c.get("title") in verified_titles],
            "rejected": [c for c in clusters if c.get("title") in rejected_titles],
            "fixed": list(fixed_entry_ids),
        }

    fake_verify_fn.calls = calls
    return fake_verify_fn


def _make_fake_synthesize_fn(text="Fable synthesis prose."):
    """Fake Fable `synthesize_fn(verified, *, model)` seam. Always returns
    `text`; records every call's (verified, model) in `.calls`."""
    calls = []

    def fake_synthesize_fn(verified, *, model):
        calls.append({"verified": verified, "model": model})
        return text

    fake_synthesize_fn.calls = calls
    return fake_synthesize_fn


def _make_fake_commit():
    """Fake best-effort git `commit(paths=, message=)` seam. Records every
    call's kwargs in `.calls`."""
    calls = []

    def fake_commit(**kwargs):
        calls.append(kwargs)
        return None

    fake_commit.calls = calls
    return fake_commit


def _happy_invoke_response(prompt, model):
    """Fake coder-LLM reply chooser for the run_census happy-path test: the
    "novel-verified" digest gets a one-candidate judgment whose cluster
    `verify_fn` will mark verified (-> promoted); "novel-rejected" gets a
    one-candidate judgment `verify_fn` will mark rejected; every other
    prompt (including the tiny headroom probe, which never mentions either
    marker) gets a matches-only (duplicate) judgment carrying no banner
    marker, so the preflight passes."""
    if "novel-verified" in prompt:
        return json.dumps(
            {
                "matches": [],
                "candidates": [
                    {
                        "title": "Silent no-op subagent contract",
                        "cause": "Agents were given contracts their runtime could not honor.",
                        "area": "orchestrator",
                        "origin_phase": "implement",
                        "manifested_phase": "verify",
                        "evidence_quote": "the subagent silently no-oped",
                    }
                ],
            }
        )
    if "novel-rejected" in prompt:
        return json.dumps(
            {
                "matches": [],
                "candidates": [
                    {
                        "title": "Spurious pattern",
                        "origin_phase": "review",
                        "manifested_phase": "merge",
                    }
                ],
            }
        )
    return json.dumps({"matches": [{"entry_id": "entry-a"}], "candidates": []})


def test_run_census_happy_path_full_seam_wiring(tmp_path):
    batch = [
        _hand_digest("dup-1", "nothing new here"),
        _hand_digest("novel-verified", "a genuinely new confusion shape"),
        _hand_digest("novel-rejected", "a spurious one-off"),
    ]
    fake_invoke = _make_fake_invoke(_happy_invoke_response)
    fake_verify_fn = _make_fake_verify_fn(
        verified_titles={"Silent no-op subagent contract"},
        rejected_titles={"Spurious pattern"},
    )
    fake_synthesize_fn = _make_fake_synthesize_fn()
    fake_submit_fn = _make_fake_submit_fn()
    fake_commit = _make_fake_commit()

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=fake_invoke,
        batch_source=[batch],
        verify_fn=fake_verify_fn,
        synthesize_fn=fake_synthesize_fn,
        submit_fn=fake_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(3),
        commit=fake_commit,
    )

    outcome = mod.run_census(**kwargs)

    # --- static model routing (ratified policy, PRD §5/§12) ---
    assert any(c["model"] == "sonnet" for c in fake_invoke.calls), "miners routed to census_miner"
    assert len(fake_verify_fn.calls) == 1
    assert fake_verify_fn.calls[0]["model"] == "sonnet", "verify routed to census_verify"
    assert len(fake_synthesize_fn.calls) == 1
    assert fake_synthesize_fn.calls[0]["model"] == "fable", "synthesis routed to census_synthesis"
    verified_titles_seen = {c["title"] for c in fake_synthesize_fn.calls[0]["verified"]}
    assert verified_titles_seen == {"Silent no-op subagent contract"}

    # --- remediation filing: curator path, never planning_mode ---
    assert len(fake_submit_fn.calls) == 1, "only the verified cluster is filed"
    filed_kwargs = fake_submit_fn.calls[0]
    assert "planning_mode" not in filed_kwargs
    assert "Silent no-op subagent contract" in filed_kwargs["title"]

    # --- codebook persisted: merge + promote + reject, never-delete ---
    assert kwargs["codebook_path"].exists()
    persisted = codebook.load(kwargs["codebook_path"])
    assert codebook.validate(persisted) == []
    promoted_entry = next(
        (e for e in persisted["entries"] if e["title"] == "Silent no-op subagent contract"), None
    )
    assert promoted_entry is not None, "verified candidate must be promoted to an entry"
    rejected_candidate = next(
        c for c in persisted["candidates"] if c["title"] == "Spurious pattern"
    )
    assert rejected_candidate["disposition"] == "rejected", "rejected candidate RETAINED, marked"

    # --- census state advanced with status_fetcher's done-count baseline ---
    assert kwargs["census_state_path"].exists()
    state = json.loads(kwargs["census_state_path"].read_text(encoding="utf-8"))
    assert state["last_census_done_count"] == 3

    # --- report written to the plans/ path ---
    assert kwargs["report_path"].exists()
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    assert "# confusion census 2026-07-14" in report_text
    assert "Fable synthesis prose." in report_text

    # --- best-effort commit of report + codebook + state ---
    assert len(fake_commit.calls) == 1

    # --- outcome: filed task ids + report path + saturation stop_reason ---
    assert outcome.status == "done"
    assert outcome.report_path == str(kwargs["report_path"])
    assert outcome.filed_task_ids == ["task-1"]
    assert outcome.stop_reason == "exhausted"


# ---------------------------------------------------------------------------
# task 3291: run_census() must never persist a FABRICATED done-count baseline.
#
# This is the test that would have caught the 2026-07-24 regression on the day
# it happened. census.py's CLI defaults --project-root to "." and
# nightly._default_census_launcher launches it with no arguments, so the
# get_statuses call went out with a relative path; fused-memory rejected it
# with a {"error", "error_type"} envelope on an isError:false response; and
# the old `(status.get("statuses") or {})` idiom silently read that as a
# done-count of 0 and persisted it as a real baseline.
# ---------------------------------------------------------------------------

def _make_error_envelope_status_fetcher():
    """Fake `status_fetcher() -> dict` returning fused-memory's tool-error
    envelope verbatim as observed live against localhost:8002. It is a
    perfectly well-formed dict -- that is precisely why the old idiom
    swallowed it -- so nothing short of a shape check can distinguish it
    from a real status snapshot."""
    def fake_status_fetcher():
        return {
            "error": "project_root must be a non-empty absolute path, got: '.'",
            "error_type": "ValidationError",
        }

    return fake_status_fetcher


def _make_raising_status_fetcher():
    def fake_status_fetcher():
        raise census_trigger.StatusFetchUnavailable("get_statuses unreachable at localhost:8002")

    return fake_status_fetcher


@pytest.mark.parametrize(
    "make_fetcher",
    [_make_error_envelope_status_fetcher, _make_raising_status_fetcher],
    ids=["tool-error-envelope", "raising-fetcher"],
)
def test_run_census_unobservable_done_count_persists_null_not_zero(
    tmp_path, caplog, make_fetcher
):
    """An unobservable done-count degrades the BASELINE only -- it must not
    abandon a run whose mining has already been paid for and whose dated
    report is already on disk.

    Before task 3291 the envelope case silently persisted 0, and the raising
    case crashed run_census AFTER the report write, leaving the codebook and
    census state unadvanced."""
    batch = [
        _hand_digest("dup-1", "nothing new here"),
        _hand_digest("novel-verified", "a genuinely new confusion shape"),
    ]
    fake_commit = _make_fake_commit()

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[batch],
        verify_fn=_make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"}),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=make_fetcher(),
        commit=fake_commit,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    # --- the census still COMPLETES; only the baseline degrades ---
    assert outcome.status == "done"
    assert kwargs["report_path"].exists(), "the paid-for report must survive"
    assert kwargs["codebook_path"].exists(), "codebook must still be dumped"
    assert len(fake_commit.calls) == 1, "the run must still commit"

    # --- the baseline is an honest null, NOT a fabricated 0 ---
    state = json.loads(kwargs["census_state_path"].read_text(encoding="utf-8"))
    assert "last_census_done_count" in state, "MUST-persist contract still holds"
    assert state["last_census_done_count"] is None
    assert state["last_census_done_count"] != 0, "a fabricated 0 is the defect being fixed"

    # --- the degradation is LOUD: exactly one warning naming the failure ---
    fetch_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "done-count" in r.getMessage()
    ]
    assert len(fetch_warnings) == 1, "degrading silently is what caused the incident"

    # --- round-trip: condition (b) is fail-SAFE, not always-armed ---
    status, data = census_trigger.load_census_state(kwargs["census_state_path"])
    assert status == "ok"
    assert census_trigger.compute_tasks_landed(
        state=data,
        status_fetcher=lambda: {"statuses": {f"t{i}": "done" for i in range(2870)}},
    ) is None


# ---------------------------------------------------------------------------
# amend: run_census() must treat submit_fn as best-effort, per payload — a
# raised exception or a non-dict result must never abort the run after
# codebook.dump() has already persisted (reviewer_comprehensive finding #2).
# ---------------------------------------------------------------------------

def test_run_census_submit_fn_raising_is_best_effort_not_fatal(tmp_path, caplog):
    batch = [
        _hand_digest("dup-1", "nothing new here"),
        _hand_digest("novel-verified", "a genuinely new confusion shape"),
    ]
    fake_invoke = _make_fake_invoke(_happy_invoke_response)
    fake_verify_fn = _make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"})
    fake_commit = _make_fake_commit()

    def raising_submit_fn(**kwargs):
        raise RuntimeError("simulated transient MCP failure")

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=fake_invoke,
        batch_source=[batch],
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=raising_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=fake_commit,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "done", "one filing failure must not abort the pipeline"
    assert outcome.filed_task_ids == []
    assert any("submit_fn" in r.message for r in caplog.records), "must log loudly, not swallow silently"

    # the codebook was already persisted before filing was attempted, and the
    # rest of the pipeline (report + state + commit) must still complete
    assert kwargs["codebook_path"].exists()
    assert kwargs["census_state_path"].exists()
    assert kwargs["report_path"].exists()
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    assert "_none filed._" in report_text
    assert len(fake_commit.calls) == 1


# ---------------------------------------------------------------------------
# amend round 2: an id-less submit_fn result must not be counted as a
# genuinely-filed task, nor leak a "- None" bullet into the human-facing
# report or inflate the filed-task count (reviewer_comprehensive finding #1).
# ---------------------------------------------------------------------------

def test_run_census_submit_fn_non_dict_result_is_not_counted_as_filed(tmp_path, caplog):
    batch = [
        _hand_digest("dup-1", "nothing new here"),
        _hand_digest("novel-verified", "a genuinely new confusion shape"),
    ]
    fake_invoke = _make_fake_invoke(_happy_invoke_response)
    fake_verify_fn = _make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"})

    def none_returning_submit_fn(**kwargs):
        return None

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=fake_invoke,
        batch_source=[batch],
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=none_returning_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert outcome.filed_task_ids == [], (
        "a non-dict submit_fn result must not crash .get('id'), and must not "
        "be counted as a genuinely-filed task"
    )
    assert any("no usable id" in r.message for r in caplog.records), "must log loudly, not swallow silently"

    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    filed_section = report_text.split("## Filed Tasks")[1].split("## Cost")[0]
    assert "None" not in filed_section, "an id-less result must never render as a '- None' bullet"
    assert "_none filed._" in filed_section


def test_run_census_promote_clamps_out_of_enum_severity_to_medium(tmp_path):
    """A custom verify_fn may enrich a cluster with a severity outside
    codebook.py's {high, medium, low} entry enum (e.g. an escalation-style
    'critical', valid elsewhere in this codebase but not in the codebook's
    own schema). That must be clamped, not persisted verbatim into a
    codebook.dump() that would otherwise raise deep into the pipeline
    (reviewer_comprehensive finding #3)."""
    batch = [
        _hand_digest("dup-1", "nothing new here"),
        _hand_digest("novel-verified", "a genuinely new confusion shape"),
    ]
    fake_invoke = _make_fake_invoke(_happy_invoke_response)

    def bad_severity_verify_fn(clusters, *, model):
        verified = [
            {**c, "severity": "critical"}
            for c in clusters
            if c.get("title") == "Silent no-op subagent contract"
        ]
        return {"verified": verified, "rejected": [], "fixed": []}

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=fake_invoke,
        batch_source=[batch],
        verify_fn=bad_severity_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done", "an out-of-enum severity from verify_fn must not crash the census"
    persisted = codebook.load(kwargs["codebook_path"])
    assert codebook.validate(persisted) == []
    promoted_entry = next(
        e for e in persisted["entries"] if e["title"] == "Silent no-op subagent contract"
    )
    assert promoted_entry["severity"] == "medium", "an out-of-enum severity is clamped, never persisted verbatim"


def test_run_census_storm_batch_is_logged_and_noted_in_report(tmp_path, caplog):
    # 2/3 digests fail to parse (>50%) -> coder.RunResult.status="failure" for
    # this batch (a storm, PRD §8.6); run_census must surface it rather than
    # silently reporting on the degraded dup-rate signal.
    batch = [
        _hand_digest("bad-1", "x"),
        _hand_digest("bad-2", "y"),
        _hand_digest("dup-1", "z"),
    ]

    def storm_invoke(prompt, model):
        if "bad-1" in prompt or "bad-2" in prompt:
            return "not valid JSON -- this coding call fails to parse"
        return json.dumps({"matches": [{"entry_id": "entry-a"}], "candidates": []})

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(storm_invoke),
        batch_source=[batch],
        verify_fn=_make_fake_verify_fn(),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "done", "a storm degrades data, it does not defer/abort the census"
    assert any("storm" in r.message.lower() for r in caplog.records), "must log loudly"
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    assert "storm" in report_text.lower(), "the report's cost note must call out degraded data"


# ---------------------------------------------------------------------------
# task 3280 step-9: RED — run_census(max_batches=) threads the operator batch
# cap end to end: mining is bounded, the capped-away batches are never pulled
# from the source, the written report says coverage was PARTIAL, and the rest
# of the pipeline still completes normally.
# ---------------------------------------------------------------------------

def _happy_batch(prefix):
    """One run_census-shaped batch for the `_happy_invoke_response` fixture:
    a duplicate, a novel-verified and a novel-rejected digest. *prefix*
    keeps each batch's session ids distinct while preserving the
    substrings `_happy_invoke_response` keys on."""
    return [
        _hand_digest(f"{prefix}-dup-1", "nothing new here"),
        _hand_digest(f"{prefix}-novel-verified", "a genuinely new confusion shape"),
        _hand_digest(f"{prefix}-novel-rejected", "a spurious one-off"),
    ]


def test_run_census_max_batches_caps_mining_and_reports_it(tmp_path):
    source = _TrackingBatchSource([_happy_batch(f"b{i}") for i in range(4)])
    fake_verify_fn = _make_fake_verify_fn(
        verified_titles={"Silent no-op subagent contract"},
        rejected_titles={"Spurious pattern"},
    )
    fake_submit_fn = _make_fake_submit_fn()
    fake_commit = _make_fake_commit()

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=source,
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=fake_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(3),
        commit=fake_commit,
        max_batches=1,
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.stop_reason == "capped"
    assert source.pulled == [0], "batches 1..3 must never be consumed from the source"

    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    lowered = report_text.lower()
    assert "operator batch cap = 1" in lowered
    assert "partial" in lowered, "a capped run must never read as full coverage"
    assert "not mined" in lowered
    # ...and PARTIAL must not read as "the remainder comes next run": this
    # very run advanced census-state, so the next window starts here.
    assert "last_census_at" in lowered
    assert "never re-enumerated" in lowered
    assert kwargs["census_state_path"].exists(), (
        "the report's re-anchor claim is only honest because state really advanced"
    )

    # The rest of the pipeline still ran to completion on the mined batch.
    assert outcome.status == "done"
    assert kwargs["codebook_path"].exists()
    assert kwargs["census_state_path"].exists()
    assert len(fake_verify_fn.calls) == 1
    assert len(fake_submit_fn.calls) == 1, "the verified cluster is still filed"
    assert len(fake_commit.calls) == 1


def test_run_census_without_max_batches_is_unchanged(tmp_path):
    source = _TrackingBatchSource([_happy_batch(f"b{i}") for i in range(2)])
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=source,
        verify_fn=_make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"}),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.stop_reason in {"saturated", "exhausted"}
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    lowered = report_text.lower()
    assert "batch cap" not in lowered, "an uncapped run must render no cap text"
    assert "coverage" not in lowered


# ---------------------------------------------------------------------------
# task 3280 step-11: RED — run_census(max_verify_clusters=) bounds the
# per-cluster verify spend WITHOUT losing data: the deferred remainder still
# merges into the codebook as pending candidates, so "deferred" means "not yet
# adjudicated", never "dropped".
# ---------------------------------------------------------------------------

_THREE_NOVEL_TITLES = ("novel one", "novel two", "novel three")


def _three_novel_invoke(prompt, model):
    """Fake coder-LLM reply chooser yielding THREE distinct novel candidate
    titles, one per `nov-N` digest, so `_novel_clusters` produces exactly
    three clusters in a deterministic mining order. Every other prompt
    (including the headroom probe) is a matches-only duplicate carrying no
    banner marker."""
    for i, title in enumerate(_THREE_NOVEL_TITLES):
        if f"nov-{i}" in prompt:
            return json.dumps(
                {
                    "matches": [],
                    "candidates": [
                        {
                            "title": title,
                            "cause": f"cause for {title}",
                            "area": "orchestrator",
                            "origin_phase": "implement",
                            "manifested_phase": "verify",
                            "evidence_quote": f"quote for {title}",
                        }
                    ],
                }
            )
    return json.dumps({"matches": [{"entry_id": "entry-a"}], "candidates": []})


def _three_novel_batch():
    return [_hand_digest(f"nov-{i}", f"novel body {i}") for i in range(3)]


def test_run_census_max_verify_clusters_defers_the_rest_as_pending_candidates(tmp_path, caplog):
    fake_verify_fn = _make_fake_verify_fn(verified_titles={_THREE_NOVEL_TITLES[0]})
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
        max_verify_clusters=1,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    # (a) exactly one cluster verified, and it is the FIRST in mining order
    assert len(fake_verify_fn.calls) == 1
    verified_clusters = fake_verify_fn.calls[0]["clusters"]
    assert len(verified_clusters) == 1
    assert verified_clusters[0]["title"] == _THREE_NOVEL_TITLES[0], "first-N in mining order"

    # (b) the DEFERRED titles are still in the dumped codebook, still pending
    persisted = codebook.load(kwargs["codebook_path"])
    assert codebook.validate(persisted) == []
    by_title = {c["title"]: c for c in persisted["candidates"]}
    for deferred_title in _THREE_NOVEL_TITLES[1:]:
        assert deferred_title in by_title, "a deferred cluster must never be DROPPED"
        assert by_title[deferred_title]["disposition"] == "pending", (
            "deferred means unverified/not-yet-adjudicated, findable by a later census"
        )

    # (c) the report states the split
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    lowered = report_text.lower()
    assert "verified 1 of 3 novel clusters" in lowered
    assert "2 deferred" in lowered
    assert "pending candidate" in lowered

    # (c2) the Cost section must agree with it. verify is ONE call PER CLUSTER
    # (_build_default_verify_fn), so a hardcoded "verify=1" would render a
    # report claiming 150 clusters verified at a cost of one call -- and this
    # line is exactly what an operator reads to check the cap did anything.
    cost_section = report_text.split("## Cost", 1)[1]
    assert "verify=1" in cost_section, "one verify call per VERIFIED cluster, capped at 1"
    assert "verify=3" not in cost_section, "the cap must be visible in the cost line"

    # loud, never silent
    assert any("defer" in r.message.lower() for r in caplog.records)

    # (d) the rest of the pipeline still persisted
    assert outcome.status == "done"
    assert kwargs["census_state_path"].exists()


def test_run_census_without_verify_cap_passes_every_novel_cluster(tmp_path):
    fake_verify_fn = _make_fake_verify_fn(verified_titles={_THREE_NOVEL_TITLES[0]})
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    mod.run_census(**kwargs)

    assert len(fake_verify_fn.calls[0]["clusters"]) == 3, "flagless -> every novel cluster"
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    assert "## Verification" not in report_text
    # An uncapped run pays one verify call per cluster; the cost line says 3,
    # not a hardcoded 1.
    assert "verify=3" in report_text.split("## Cost", 1)[1]


def test_run_census_warns_and_escalates_when_every_cluster_is_rejected(tmp_path, caplog):
    """Clusters offered, none verified, is the observable signature of a
    SYSTEMIC verifier failure — say it out loud rather than reporting an
    empty census in the same voice as an unremarkable one.

    Scoping the subprocess cwd (2026-08-03) removed one CAUSE of that
    silence, not the class: _build_default_verify_fn fails CLOSED per
    cluster, so a model that goes unreachable mid-run, a different
    permission denial, or persistently unparseable verdicts all still land
    as ordinary rejections with nothing surfacing the pattern.
    """
    fake_escalate_fn = _make_fake_escalate_fn()
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=_make_fake_verify_fn(verified_titles=set()),  # every cluster rejects
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=fake_escalate_fn,
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    warnings = " ".join(r.message.lower() for r in caplog.records)
    assert "systemic" in warnings, "the pattern must be NAMED, not left to be inferred"
    assert "3" in warnings, "how many clusters were offered"

    assert len(fake_escalate_fn.calls) == 1, fake_escalate_fn.calls
    escalation = fake_escalate_fn.calls[0]
    assert escalation["severity"] == "info"
    assert "reject" in escalation["summary"].lower()

    # A suspicious run is still a COMPLETED run: mining is already paid for,
    # and an all-rejected census is legitimately possible. Detect, never fail.
    assert outcome.status == "done"
    assert kwargs["census_state_path"].exists()


def test_run_census_does_not_warn_when_some_cluster_verifies(tmp_path, caplog):
    """The detector must not fire on an ordinary run — a partial rejection
    is the normal case and must stay quiet, or the warning is noise that
    gets filtered out before the real incident."""
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=_make_fake_verify_fn(verified_titles={_THREE_NOVEL_TITLES[0]}),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),  # 2 of 3 rejected -> nothing to escalate
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    with caplog.at_level(logging.WARNING):
        mod.run_census(**kwargs)

    assert not any("systemic" in r.message.lower() for r in caplog.records)


def test_run_census_mass_rejection_escalation_failure_is_not_fatal(tmp_path, caplog):
    """A raising escalate_fn must not discard a run whose mining is already
    paid for. Unlike the defer-path escalation (which returns immediately
    afterwards), this one sits between the mining spend and the output
    writes."""
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=_make_fake_verify_fn(verified_titles=set()),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),  # raises on the detector's call
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert kwargs["report_path"].exists()
    assert kwargs["census_state_path"].exists()
    # Swallowed, but never silently: both the finding and the failed post log.
    warnings = " ".join(r.message.lower() for r in caplog.records)
    assert "systemic" in warnings
    assert "escalation failed" in warnings


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"max_batches": 0},
        {"max_batches": -1},
        {"max_verify_clusters": 0},
        {"max_verify_clusters": -1},
    ],
)
def test_run_census_rejects_a_nonpositive_cost_cap_before_spending_anything(tmp_path, bad_kwargs):
    # Both caps are public seams (callable from tests and other scripts), so
    # the guard lives here too, not only at the CLI boundary. It fires BEFORE
    # preflight_headroom, so a nonsense cap costs zero invoke calls -- and
    # max_verify_clusters=-1 in particular would otherwise slice
    # novel_clusters[:-1], silently verifying all but the LAST cluster while
    # reporting cap=-1 as if it had been honored.
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_poison("invoke"),
        batch_source=_poison("batch_source"),
        escalate_fn=_poison("escalate_fn"),
        **bad_kwargs,
    )

    with pytest.raises(ValueError, match="max_batches|max_verify_clusters"):
        mod.run_census(**kwargs)

    assert not kwargs["report_path"].exists()
    assert not kwargs["codebook_path"].exists()
    assert not kwargs["census_state_path"].exists()


# ---------------------------------------------------------------------------
# task 3280 step-13: RED — run_census(dry_run_payloads_path=) stubs ONLY the
# external task filing. Every other side effect (codebook dump, promotions,
# report, census-state advance, best-effort commit) proceeds normally, so a
# dry run is a faithful preview of what a real run would file, not a
# half-executed census.
# ---------------------------------------------------------------------------

def test_run_census_dry_run_filing_writes_payloads_and_files_nothing(tmp_path, caplog):
    payloads_path = tmp_path / "confusion-census-2026-07-14-payloads.json"
    fake_submit_fn = _make_fake_submit_fn()
    fake_commit = _make_fake_commit()
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[_happy_batch("b0")],
        verify_fn=_make_fake_verify_fn(
            verified_titles={"Silent no-op subagent contract"},
            rejected_titles={"Spurious pattern"},
        ),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=fake_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(3),
        commit=fake_commit,
        dry_run_payloads_path=payloads_path,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    # (a) NOTHING filed -- the real submit_fn was never reached
    assert fake_submit_fn.calls == []

    # (b) every would-be payload written, in build_task_payloads' own shape
    assert payloads_path.exists()
    payloads = json.loads(payloads_path.read_text(encoding="utf-8"))
    assert isinstance(payloads, list) and len(payloads) == 1
    payload = payloads[0]
    assert payload["project_root"] == str(tmp_path)
    assert payload["title"].startswith("[legibility census]")
    assert "Silent no-op subagent contract" in payload["title"]
    assert payload["description"]
    assert payload["task_kind"] == "normal"
    assert payload["priority"]
    assert payload["metadata"]["source"] == "legibility_census"

    # (c) the report says a dry run happened, naming count and path
    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    section = report_text.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert "dry-run: 1 payload" in section
    assert str(payloads_path) in section
    assert "_none filed._" not in section

    # (d) everything else still happened, and the payload file is committed
    # alongside the artifacts it was produced from
    assert kwargs["codebook_path"].exists()
    assert kwargs["census_state_path"].exists()
    assert len(fake_commit.calls) == 1
    committed = fake_commit.calls[0]["paths"]
    assert str(payloads_path) in committed
    assert str(kwargs["report_path"]) in committed
    assert str(kwargs["codebook_path"]) in committed
    assert str(kwargs["census_state_path"]) in committed

    # (e) the outcome names the review file instead of a bare filed_tasks=0
    assert outcome.status == "done"
    assert outcome.filed_task_ids == []
    assert outcome.dry_run is not None
    assert outcome.dry_run.path == str(payloads_path)
    assert outcome.dry_run.payload_count == 1

    assert any("dry" in r.message.lower() for r in caplog.records), "loud, never silent"


# ---------------------------------------------------------------------------
# task 3280 step-18: RED — the dry-run WARNING must describe the state the run
# ACTUALLY left behind. A dry run is NOT a no-op that can be replayed: the
# codebook merge, the promotions, codebook.dump and advance_census_state all
# really happened, so re-running the census files NOTHING (the same confusions
# now code as `matches` against the advanced codebook, `_novel_clusters` comes
# back empty, build_task_payloads returns [], and _census_window_dates has
# re-anchored at this run's last_census_at so the earlier window is never
# enumerated again). Advertising a re-run as the recovery path sends the
# operator down a road that silently drops the remediation work.
# ---------------------------------------------------------------------------

def test_run_census_dry_run_warning_states_advanced_state_and_no_rerun_recovery(
    tmp_path, caplog,
):
    payloads_path = tmp_path / "confusion-census-2026-07-14-payloads.json"
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[_happy_batch("b0")],
        verify_fn=_make_fake_verify_fn(
            verified_titles={"Silent no-op subagent contract"},
            rejected_titles={"Spurious pattern"},
        ),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(3),
        commit=_make_fake_commit(),
        dry_run_payloads_path=payloads_path,
    )

    with caplog.at_level(logging.WARNING):
        mod.run_census(**kwargs)

    records = [r for r in caplog.records if "dry-run-filing" in r.message]
    assert len(records) == 1, "exactly one dry-run WARNING, loud and singular"
    msg = records[0].message.lower()

    # (a) still names the payload count and path -- the operator's handle on
    # the work
    assert "1 task payload" in msg
    assert str(payloads_path).lower() in msg

    # (b) states that the codebook AND census-state have already advanced --
    # this run mutated persistent state, it was not a rehearsal
    assert "codebook" in msg
    assert "census-state" in msg or "census state" in msg
    assert any(marker in msg for marker in ("advanced", "already")), (
        "the WARNING must say the state was ALREADY advanced, not that it might be"
    )

    # (c) names hand-filing the payload file as the remaining path
    assert "hand" in msg

    # (d) THE FINDING: never advertise a re-run as recovery. A second census
    # cannot re-file these payloads, so pointing the operator at one loses
    # the remediation work.
    for dead_end in ("re-run", "rerun", "run again"):
        assert dead_end not in msg, (
            f"WARNING advertises the dead recovery path {dead_end!r}: a repeat "
            "census codes these confusions as matches and files nothing"
        )


# ---------------------------------------------------------------------------
# task 3280 step-20: RED — the payload file is a human-review deliverable and
# the ONLY handle on a dry run's remediation work (the codebook and
# census-state have already advanced, so nothing can regenerate it). A second
# dry run on the same date must therefore never overwrite the first: the
# earlier artifact is left untouched and this run's payloads go to a numbered
# sibling, loudly.
# ---------------------------------------------------------------------------

def _dry_run_kwargs(tmp_path, payloads_path, *, verify_fn, submit_fn, commit):
    return _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[_happy_batch("b0")],
        verify_fn=verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(3),
        commit=commit,
        dry_run_payloads_path=payloads_path,
    )


def test_run_census_dry_run_does_not_clobber_an_existing_payload_file(tmp_path, caplog):
    collide_path = tmp_path / "confusion-census-2026-07-14-payloads.json"
    collide_path.write_text('["SENTINEL"]', encoding="utf-8")
    sibling_path = tmp_path / "confusion-census-2026-07-14-payloads-2.json"

    fake_verify_fn = _make_fake_verify_fn(
        verified_titles={"Silent no-op subagent contract"},
        rejected_titles={"Spurious pattern"},
    )
    fake_commit = _make_fake_commit()
    kwargs = _dry_run_kwargs(
        tmp_path, collide_path,
        verify_fn=fake_verify_fn, submit_fn=_make_fake_submit_fn(), commit=fake_commit,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    # (a) the earlier review artifact survives verbatim -- overwriting it
    # would destroy work that no re-run can reproduce
    assert collide_path.read_text(encoding="utf-8") == '["SENTINEL"]'

    # (b) this run's payloads landed on the numbered sibling, in
    # build_task_payloads' own shape
    assert sibling_path.exists()
    verified = [
        c for c in fake_verify_fn.calls[0]["clusters"]
        if c.get("title") == "Silent no-op subagent contract"
    ]
    expected = mod.build_task_payloads(
        verified, project_root=str(tmp_path), project_id="dark_factory",
    )
    assert expected, "guard: the scenario must actually produce a payload"
    assert json.loads(sibling_path.read_text(encoding="utf-8")) == expected

    # (c)/(d)/(e) every downstream consumer names the path actually written
    assert outcome.dry_run is not None
    assert outcome.dry_run.path == str(sibling_path)

    report_text = kwargs["report_path"].read_text(encoding="utf-8")
    section = report_text.split("## Filed Tasks", 1)[1].split("##", 1)[0]
    assert str(sibling_path) in section
    assert str(collide_path) not in section, "the report must not point at the file it did NOT write"

    assert str(sibling_path) in fake_commit.calls[0]["paths"]
    assert str(collide_path) not in fake_commit.calls[0]["paths"]

    # (f) never silent -- one WARNING names BOTH paths
    collision_records = [
        r for r in caplog.records
        if str(collide_path) in r.message and str(sibling_path) in r.message
    ]
    assert len(collision_records) == 1, "the collision must be reported, naming both paths"


def test_run_census_dry_run_uses_the_given_path_when_free(tmp_path, caplog):
    payloads_path = tmp_path / "confusion-census-2026-07-14-payloads.json"
    fake_commit = _make_fake_commit()
    kwargs = _dry_run_kwargs(
        tmp_path, payloads_path,
        verify_fn=_make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"}),
        submit_fn=_make_fake_submit_fn(),
        commit=fake_commit,
    )

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    # no gratuitous renaming: the free path is used verbatim (step-13 contract)
    assert payloads_path.exists()
    assert outcome.dry_run is not None  # DryRunFiling | None on the outcome
    assert outcome.dry_run.path == str(payloads_path)
    assert list(tmp_path.glob("*-payloads-*.json")) == []
    assert not any("already exists" in r.message for r in caplog.records)


def test_run_census_without_dry_run_files_normally_and_writes_no_payload_file(tmp_path):
    fake_submit_fn = _make_fake_submit_fn()
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[_happy_batch("b0")],
        verify_fn=_make_fake_verify_fn(verified_titles={"Silent no-op subagent contract"}),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=fake_submit_fn,
        escalate_fn=_poison("escalate_fn"),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    outcome = mod.run_census(**kwargs)

    assert len(fake_submit_fn.calls) == 1, "the flagless path still files per payload"
    assert outcome.filed_task_ids == ["task-1"]
    assert outcome.dry_run is None
    assert list(tmp_path.glob("*-payloads.json")) == []


# ---------------------------------------------------------------------------
# step-21: RED — main(argv) CLI
# ---------------------------------------------------------------------------

def _write_legibility_yaml(config_path, *, project_id="dark_factory", project_root=None,
                            escalation_port=8103, cwd_prefixes=None,
                            agent_transcript_roots=None):
    """Write a minimal valid legibility.yaml to *config_path* (any path —
    the caller decides whether it lives at the default
    <project-root>/docs/legibility/legibility.yaml location or elsewhere,
    to exercise --config's override). Plain-text lines, not a yaml.safe_dump
    round trip — mirrors test_legibility_nightly.py's _write_config, kept
    independent of the module under test's own YAML writer.

    When *agent_transcript_roots* is given, an ``agent_transcript_roots:``
    block is appended so the loaded cfg opts into archive-root enumeration."""
    project_root = project_root if project_root is not None else config_path.parent
    cwd_prefixes = cwd_prefixes if cwd_prefixes is not None else [str(project_root)]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"project_id: {project_id}",
        f"project_root: {project_root}",
        f"escalation_port: {escalation_port}",
        "cwd_prefixes:",
    ]
    lines += [f"  - {prefix}" for prefix in cwd_prefixes]
    if agent_transcript_roots is not None:
        lines.append("agent_transcript_roots:")
        lines += [f"  - {r}" for r in agent_transcript_roots]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _default_config_path(project_root):
    return project_root / "docs" / "legibility" / "legibility.yaml"


def _make_fake_main_run_census(outcome=None):
    """Fake `run_census(**kwargs) -> CensusOutcome` seam for main()-level
    tests — records every call's kwargs in `.calls`, never touches a real
    seam. Defaults to a "done" outcome so a happy-path main() call has
    something sensible to print/return."""
    calls = []

    def fake_run_census(**kwargs):
        calls.append(kwargs)
        return outcome or mod.CensusOutcome(
            status="done", report_path="plans/confusion-census-2026-01-02.md",
            filed_task_ids=["1234"], stop_reason="exhausted",
        )

    fake_run_census.calls = calls
    return fake_run_census


def test_main_force_bypasses_gate_and_calls_run_census(tmp_path, monkeypatch):
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    # proves --force never even reaches the gate
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code == 0
    assert len(fake_run_census.calls) == 1
    assert fake_run_census.calls[0]["force"] is True


def test_main_without_force_no_fire_noops_with_exit_zero(tmp_path, monkeypatch, capsys):
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=False, reasons=["max-interval: not yet due"])

    monkeypatch.setattr(census_trigger, "decide_for_project", fake_decide)

    exit_code = mod.main(["--project-root", str(tmp_path)])

    assert exit_code == 0
    assert fake_run_census.calls == [], "a no-fire decision must never call run_census"
    out = capsys.readouterr().out
    assert "no-fire" in out.lower(), "an explanatory stdout line naming the no-fire decision"
    assert "max-interval: not yet due" in out


# ---------------------------------------------------------------------------
# step-13: main() must configure logging -- census has the SAME omission as
# nightly. Nothing under scripts/legibility/ called logging.basicConfig, so
# root sat at its WARNING default and every INFO line this module emits --
# notably "census: no codebook at ... yet, starting from an empty v2
# document" -- was discarded before it reached the journal. Mirrors
# test_legibility_nightly.py's step-11 trio.
#
# The _isolated_root_logging helper below is DUPLICATED verbatim from that
# file. Its proper home is scripts/tests/conftest.py -- which task 3270 holds
# no lock on, so the move is deferred to the next review cycle rather than
# smuggled in. Not a norm, a known wart: keep the two copies in step.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _isolated_root_logging():
    """Yield the ROOT logger with its handlers emptied, restoring it after.

    ``config.configure_logging`` goes through ``logging.basicConfig``, which
    is a NO-OP when root already has handlers -- the very property that makes
    it safe under pytest -- so its effect is only observable with root
    cleared first.

    Restoring the level AND the exact handler list is load-bearing here: this
    file runs ~10 ``caplog.at_level(logging.WARNING)`` blocks (lines 1318,
    1520, 1650, 1817, 1938, 2018, ...) that leaked root state would silently
    perturb.

    Keep in step with the copy in test_legibility_nightly.py until one of
    them moves to conftest.py -- see the section comment above.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = root.handlers[:]
    root.handlers[:] = []
    root.setLevel(logging.WARNING)  # the un-configured default this fix exists to beat
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_main_configures_logging_so_info_lines_reach_the_journal(tmp_path, monkeypatch):
    """Driven down the cheap NO-FIRE path -- no LLM, no subprocess, exit 0 --
    because logging must be configured before the gate, not only on runs that
    fire."""
    # The DEFAULT-level case, so the ambient env has to be cleared: this same
    # change teaches the CLIs to honour LEGIBILITY_LOG_LEVEL, and a developer
    # debugging the trickle (or a host whose unit env is sourced) exporting
    # LEGIBILITY_LOG_LEVEL=WARNING would otherwise turn a working fix red.
    monkeypatch.delenv("LEGIBILITY_LOG_LEVEL", raising=False)
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=False, reasons=["max-interval: not yet due"])

    monkeypatch.setattr(census_trigger, "decide_for_project", fake_decide)

    with _isolated_root_logging() as root:
        exit_code = mod.main(["--project-root", str(tmp_path)])
        # Sample inside the block, assert outside, so a failing assertion is
        # reported by pytest with root logging already restored.
        effective_level = root.getEffectiveLevel()
        handler_count = len(root.handlers)

    assert exit_code == 0
    assert fake_run_census.calls == [], "the NO-FIRE path must stay cheap -- no run_census"
    assert effective_level <= logging.INFO, (
        "census.main() must lower root to INFO -- otherwise the empty-codebook line "
        "and every other census INFO line is dropped before it reaches the journal"
    )
    assert handler_count >= 1, "root needs a handler, or INFO records go nowhere"


def test_main_without_force_fire_decision_runs_pipeline(tmp_path, monkeypatch):
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=["max-interval: overdue -> FIRE"])

    monkeypatch.setattr(census_trigger, "decide_for_project", fake_decide)

    exit_code = mod.main(["--project-root", str(tmp_path)])

    assert exit_code == 0
    assert len(fake_run_census.calls) == 1
    assert fake_run_census.calls[0]["force"] is False


def test_main_config_flag_overrides_default_path_and_date_flag_threads_through(tmp_path, monkeypatch):
    # deliberately NOT at the default <project-root>/docs/legibility/legibility.yaml
    # location, so this only passes if --config is actually honored.
    alt_config = _write_legibility_yaml(tmp_path / "alt-legibility.yaml", project_root=tmp_path)
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main([
        "--project-root", str(tmp_path),
        "--config", str(alt_config),
        "--force",
        "--date", "2026-01-02",
    ])

    assert exit_code == 0
    kwargs = fake_run_census.calls[0]
    assert kwargs["date"] == "2026-01-02"
    assert kwargs["project_id"] == "dark_factory"
    assert kwargs["project_root"] == str(tmp_path)


def test_main_cost_control_flags_thread_into_run_census(tmp_path, monkeypatch):
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main([
        "--project-root", str(tmp_path),
        "--force",
        "--date", "2026-07-30",
        "--max-batches", "50",
        "--max-verify-clusters", "150",
        "--dry-run-filing",
    ])

    assert exit_code == 0
    kwargs = fake_run_census.calls[0]
    assert kwargs["max_batches"] == 50
    assert kwargs["max_verify_clusters"] == 150
    # the dated payload file sits alongside the dated report
    assert str(kwargs["dry_run_payloads_path"]) == str(
        tmp_path / "plans" / "confusion-census-2026-07-30-payloads.json"
    )
    assert str(kwargs["report_path"]) == str(
        tmp_path / "plans" / "confusion-census-2026-07-30.md"
    )


def test_main_without_cost_control_flags_passes_defaults(tmp_path, monkeypatch):
    # The nightly launcher (nightly.py) runs census.py with NO extra argv, so
    # this is the shape that must stay behaviorally byte-identical.
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code == 0
    kwargs = fake_run_census.calls[0]
    assert kwargs["max_batches"] is None
    assert kwargs["max_verify_clusters"] is None
    assert kwargs["dry_run_payloads_path"] is None


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-batches", "0"),
        ("--max-batches", "-1"),
        ("--max-verify-clusters", "0"),
        ("--max-verify-clusters", "-5"),
    ],
)
def test_main_rejects_a_nonpositive_cost_cap_at_the_cli_boundary(
    tmp_path, monkeypatch, capsys, flag, value,
):
    # A nonsense cap on a flag whose entire purpose is to be an explicit,
    # legible bound must exit non-zero with a message, not degenerate into a
    # half-applied cap. argparse raises SystemExit(2) for a type= rejection.
    _write_legibility_yaml(_default_config_path(tmp_path))
    monkeypatch.setattr(mod, "run_census", _poison("run_census"))
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--project-root", str(tmp_path), "--force", flag, value])

    assert excinfo.value.code != 0
    err = capsys.readouterr().err.lower()
    assert flag in err, "the rejected flag must be named"
    assert "1 or greater" in err, "the message must say what a valid cap looks like"


def test_main_accepts_a_cap_of_one(tmp_path, monkeypatch):
    # The boundary itself is valid: 1 is the smallest cap that can be honored.
    _write_legibility_yaml(_default_config_path(tmp_path))
    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main([
        "--project-root", str(tmp_path), "--force",
        "--max-batches", "1", "--max-verify-clusters", "1",
    ])

    assert exit_code == 0
    assert fake_run_census.calls[0]["max_batches"] == 1
    assert fake_run_census.calls[0]["max_verify_clusters"] == 1


def test_main_dry_run_summary_line_names_payload_file(tmp_path, monkeypatch, capsys):
    _write_legibility_yaml(_default_config_path(tmp_path))
    payloads_path = "/p/plans/confusion-census-2026-07-30-payloads.json"
    fake_run_census = _make_fake_main_run_census(
        outcome=mod.CensusOutcome(
            status="done",
            report_path="plans/confusion-census-2026-07-30.md",
            filed_task_ids=[],
            stop_reason="capped",
            dry_run=mod.DryRunFiling(path=payloads_path, payload_count=7),
        )
    )
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main([
        "--project-root", str(tmp_path), "--force", "--dry-run-filing",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert payloads_path in out
    assert "7 payload" in out
    assert "nothing filed" in out.lower()
    # a bare filed_tasks=0 would read as "a normal run that filed nothing"
    assert "filed_tasks=0" not in out


def test_main_missing_config_returns_nonzero(tmp_path, monkeypatch):
    # no legibility.yaml written anywhere -- config.load_config must fail
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))
    monkeypatch.setattr(mod, "run_census", _poison("run_census"))

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code != 0


def test_main_returns_nonzero_on_fail_loud_error(tmp_path, monkeypatch):
    _write_legibility_yaml(_default_config_path(tmp_path))

    def raising_run_census(**kwargs):
        raise RuntimeError("codebook merge produced an invalid codebook")

    monkeypatch.setattr(mod, "run_census", raising_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))
    # Once main() routes fail-loud errors through escalate_fn, the failure path
    # POSTs escalate_info; stub the single MCP boundary so this test never
    # attempts a real localhost POST (conftest: no test hits a real endpoint).
    monkeypatch.setattr(mod, "_post_mcp_tool_call", lambda *a, **k: {})

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code != 0


def test_main_failure_files_escalation(tmp_path, monkeypatch):
    """main()'s fail-loud catch-all must file an escalate_info via the reused
    escalate_fn closure (PRD decision 8: degradation never silent) -- a hard
    census failure exits non-zero AND leaves an operator signal, rather than
    dying with only a stderr line (the silent-census incident this fixes)."""
    _write_legibility_yaml(_default_config_path(tmp_path))

    def raising_run_census(**kwargs):
        raise RuntimeError("codebook merge produced an invalid codebook")

    monkeypatch.setattr(mod, "run_census", raising_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    posts = []

    def rec(url, tool_name, arguments):
        posts.append((url, tool_name, arguments))
        return {}

    monkeypatch.setattr(mod, "_post_mcp_tool_call", rec)

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code == 1
    escalations = [args for (_url, tool_name, args) in posts if tool_name == "escalate_info"]
    assert len(escalations) == 1, "exactly one escalate_info POST for the hard failure"
    arguments = escalations[0]
    assert arguments["task_id"] == "legibility-census-dark_factory"
    assert arguments["category"] == "infra_issue"
    assert arguments["summary"] and "dark_factory" in arguments["summary"]
    assert "codebook merge produced an invalid codebook" in arguments["detail"]


def test_main_failure_escalation_is_best_effort_when_poster_raises(tmp_path, monkeypatch, caplog):
    """The failure escalation is best-effort: if the escalation POST itself
    raises, the closure swallows it (logging a WARNING) and main() STILL
    returns 1 -- the escalation never masks the authoritative exit code."""
    _write_legibility_yaml(_default_config_path(tmp_path))

    def raising_run_census(**kwargs):
        raise RuntimeError("codebook merge produced an invalid codebook")

    monkeypatch.setattr(mod, "run_census", raising_run_census)
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    def raising_post(url, tool_name, arguments):
        raise RuntimeError("escalation server down")

    monkeypatch.setattr(mod, "_post_mcp_tool_call", raising_post)

    with caplog.at_level(logging.WARNING, logger="legibility.census"):
        exit_code = mod.main(["--project-root", str(tmp_path), "--force"])

    assert exit_code == 1
    assert any(
        r.levelno >= logging.WARNING for r in caplog.records
    ), "a raising escalation poster must be logged, never propagated"


# ---------------------------------------------------------------------------
# default_batch_source — the whole census window is enumerated in ONE walk via
# enumerate_sessions_in_range (O(files), not O(window_days × files)), and the
# shipped agent_transcript_roots is threaded into that single call with NO
# operator flip (resolved against cfg.project_root). Patches
# inventory.enumerate_sessions_in_range (the module object census.py itself
# references via `import inventory`) to capture its window + kwargs.
# ---------------------------------------------------------------------------

def test_default_batch_source_passes_resolved_archive_roots_to_enumerate(tmp_path, monkeypatch):
    config_path = _write_legibility_yaml(
        _default_config_path(tmp_path), project_root=tmp_path,
        agent_transcript_roots=["data/orchestrator/agent-transcripts"],
    )
    cfg = config_mod.load_config(config_path)

    captured = []

    def fake_enumerate_sessions_in_range(
        projects_root, cwd_prefixes, start_date, end_date, **kwargs
    ):
        captured.append((start_date, end_date, kwargs))
        return []

    monkeypatch.setattr(
        inventory, "enumerate_sessions_in_range", fake_enumerate_sessions_in_range
    )

    now = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    # Consume the generator: nothing enumerates until it is iterated.
    list(mod.default_batch_source(cfg, projects_root=tmp_path / "projects", now=now))

    # With no census-state.json under cfg.project_root the window is the
    # multi-date default lookback (>1 date) — the old per-date loop would have
    # called the enumerator once per date.
    window = mod._census_window_dates(cfg.project_root, now=now)
    assert len(window) > 1, "sanity: this test needs a genuinely multi-date window"

    # (1) ONE range-enumerate call regardless of window length — the
    # O(files)-not-O(window_days × files) census-level signal.
    assert len(captured) == 1, "enumerate_sessions_in_range must be called exactly once"

    start_date, end_date, kwargs = captured[0]
    # (2) resolved archive roots threaded with no operator flip.
    expected_roots = inventory.resolve_agent_transcript_roots(
        cfg.project_root, cfg.agent_transcript_roots
    )
    assert expected_roots == [tmp_path / "data" / "orchestrator" / "agent-transcripts"]
    assert kwargs["agent_transcript_roots"] == expected_roots
    # (3) the [start, end] window equals _census_window_dates' first/last.
    assert (start_date, end_date) == (window[0], window[-1])


# ---------------------------------------------------------------------------
# task 2953: _post_mcp_tool_call() streamable-HTTP Accept headers
# ---------------------------------------------------------------------------

class _FakeHttpxResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_post_mcp_tool_call_sends_streamable_http_accept_headers(install_fake_httpx):
    """Task 2953: the streamable-HTTP MCP transport 406s any tools/call POST
    lacking an Accept header covering both application/json and
    text/event-stream (verified live against a local MCP /mcp endpoint --
    shared by census.py's submit_task and escalate_info posters via this one
    function). httpx is imported lazily, but it IS importable here -- a
    direct dependency of `shared` (shared/pyproject.toml, `httpx>=0.27`,
    task 2965) -- so an un-faked call would really hit the network. The
    shared `install_fake_httpx` fixture substitutes a stub so the outbound
    request shape is assertable without a live listener on :8002."""
    captured_kwargs = {}
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": {"ok": True}},
    }

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse(rpc_response)

    install_fake_httpx(_fake_post)

    result = mod._post_mcp_tool_call("http://localhost:8002/mcp", "submit_task", {"a": 1})

    assert result == {"ok": True}
    headers = captured_kwargs.get("headers") or {}
    assert "application/json" in headers.get("Accept", "")
    assert "text/event-stream" in headers.get("Accept", "")
    # Content-Type is part of the same transport contract -- pin it too so a
    # future edit dropping it can't pass on the Accept assertions alone.
    assert headers.get("Content-Type") == "application/json"
    # The JSON-RPC tools/call body (name + arguments) must still ride along.
    envelope = captured_kwargs.get("json") or {}
    assert envelope.get("method") == "tools/call"
    assert envelope.get("params", {}).get("name") == "submit_task"
    assert envelope.get("params", {}).get("arguments") == {"a": 1}


# ---------------------------------------------------------------------------
# _build_stage_invokes — each census stage gets its OWN claude-CLI subprocess
# timeout, threaded through the invoke(prompt, model) seam via
# functools.partial(coder._invoke_cli, timeout=...). The shared 120s coder
# default is fine for mining/headroom but fatal for the per-cluster Sonnet
# verify-vs-main and the one large Fable synthesis; this is where that split
# is bound.
# ---------------------------------------------------------------------------

def _config_with_timeouts(mining, verify, synthesis):
    """A minimal valid LegibilityConfig carrying explicit stage timeouts."""
    return config_mod.LegibilityConfig(
        project_id="dark_factory",
        project_root="/home/leo/src/dark-factory",
        escalation_port=8103,
        cwd_prefixes=["/home/leo/src/dark-factory"],
        timeouts=config_mod.Timeouts(
            census_mining_secs=mining,
            census_verify_secs=verify,
            census_synthesis_secs=synthesis,
        ),
    )


def test_build_stage_invokes_threads_each_stage_timeout(monkeypatch):
    # Record the timeout every stage invoke threads to coder._invoke_cli.
    recorded = []

    def fake_invoke_cli(prompt, model, *, claude_bin=None, timeout=None, cwd=None):
        recorded.append(timeout)
        return "dummy"

    monkeypatch.setattr(coder, "_invoke_cli", fake_invoke_cli)

    cfg = _config_with_timeouts(111, 222, 333)
    mining, verify, synth = mod._build_stage_invokes(cfg, project_root="/home/leo/src/dark-factory")

    # Drive each partial exactly as its census seam does: two positional
    # args, no kwargs (invoke(prompt, model)).
    mining("p", "haiku")
    verify("p", "sonnet")
    synth("p", "fable")

    # Each stage's own timeout threads through, in [mining, verify, synth]
    # order — proving every stage carries its distinct budget.
    assert recorded == [111, 222, 333]


def test_main_wires_per_stage_timeouts_into_run_census(tmp_path, monkeypatch):
    # REGRESSION for the exact bug: main() once handed the raw
    # coder._invoke_cli (120s module default) to EVERY stage, so every
    # per-cluster Sonnet verify-vs-main call and the large Fable synthesis
    # call died at 120s and census-state.json never advanced. Pin that each
    # stage now receives its own budget.
    #
    # DEFAULT config — NO timeouts block — so cfg.timeouts falls back to the
    # schema defaults (120/900/1800). This is exactly the shape of a
    # pre-existing legibility.yaml.
    _write_legibility_yaml(_default_config_path(tmp_path))

    recorded = []

    def fake_invoke_cli(prompt, model, *, claude_bin=None, timeout=None, cwd=None):
        recorded.append({"model": model, "timeout": timeout})
        return '{"verified": true}'

    monkeypatch.setattr(coder, "_invoke_cli", fake_invoke_cli)

    fake_run_census = _make_fake_main_run_census()
    monkeypatch.setattr(mod, "run_census", fake_run_census)
    # --force must never reach the gate.
    monkeypatch.setattr(census_trigger, "decide_for_project", _poison("decide_for_project"))

    exit_code = mod.main(["--project-root", str(tmp_path), "--force"])
    assert exit_code == 0

    kwargs = fake_run_census.calls[0]

    # (1) mining/headroom invoke -> mining budget 120 (unchanged; short calls).
    kwargs["invoke"]("ping", "haiku")
    assert recorded[-1] == {"model": "haiku", "timeout": 120}

    # (2) verify_fn -> per-cluster Sonnet call carries 900 (was the fatal 120).
    kwargs["verify_fn"]([{"title": "x"}], model="sonnet")
    assert recorded[-1] == {"model": "sonnet", "timeout": 900}

    # (3) synthesize_fn -> the one large Fable call carries 1800.
    kwargs["synthesize_fn"]([{"title": "x"}], model="fable")
    assert recorded[-1] == {"model": "fable", "timeout": 1800}


# ---------------------------------------------------------------------------
# task 3645 / DEFECT 3: a cap arriving DURING verification must never be
# laundered into a verdict.
#
# _build_default_verify_fn fails CLOSED per cluster: any invocation error or
# parse failure appends the cluster to `rejected` and continues. That is the
# right default for a genuinely unverifiable CLAIM, and exactly wrong for an
# infra failure -- a cap mid-verify silently mass-rejects the remaining
# population, and the run exits 0 looking unremarkable.
#
# Three detectors, ordered cheapest-first. These two fire at the FIRST
# corrupted cluster, which is what a periodic counter alone cannot do.
# ---------------------------------------------------------------------------

def _verdict(verified=True):
    return json.dumps({"verified": verified, "reason": "because"})


def _clusters(n):
    return [{"title": f"cluster-{i}"} for i in range(n)]


def _make_recording_probe(*results):
    """Fake `headroom_probe() -> HeadroomResult` seam, recording its calls.

    Returns the i-th result for the i-th call, repeating the last one after
    that, so a test can say "healthy, then capped" without counting calls.
    """
    calls = []

    def probe():
        calls.append(len(calls))
        return results[min(len(calls) - 1, len(results) - 1)]

    probe.calls = calls
    return probe


def test_default_verify_fn_raises_when_the_raw_reply_carries_a_banner():
    """Detector (a): the cap text arrives as the verify reply itself.

    This is the common case -- the CLI prints its cap banner to stdout, which
    parse_coder_output would otherwise turn into a parse failure and thus a
    rejection. The banner-matching CONTENT is only the TRIGGER: a probe
    CONFIRMS the cap before the run is aborted, so the detector can never
    decide from content alone that the account is capped.
    """
    cap = REAL_CLI_CAP_MESSAGES[0]
    replies = [_verdict(), _verdict(), cap, _verdict(), _verdict()]
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        return replies[len(calls) - 1]

    probe = _make_recording_probe(
        mod.HeadroomResult(ok=False, reason="probe reply carries a banner marker: 'weekly limit'")
    )
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(mod.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(5), model="sonnet")

    exc = excinfo.value
    assert exc.stage == "verify"
    assert exc.reason and "banner" in exc.reason.lower()
    assert exc.verified == 2, "the two clusters adjudicated before the cap"
    assert exc.unverified == 3, "the hitting cluster plus the two never attempted"
    assert len(probe.calls) == 1, "exactly one confirming probe, on a path that already failed"


# ---------------------------------------------------------------------------
# task 3645 / REVIEW FIX, part 1/2: detector (a) must never re-read a
# WELL-FORMED verdict as a cap banner.
#
# The verify reply is a model-authored {"verified":…, "reason":…} whose reason
# legitimately QUOTES the cluster under adjudication -- and this repo's
# codebook is dominated by clusters ABOUT usage/weekly limits
# (docs/legibility/confusion-codebook.yaml carries ~15 such sightings at lines
# 569, 575, 925, 985, 1048-1163). A loose OR-substring scan over the whole raw
# reply therefore fires on ordinary healthy output.
#
# The consequence is worse than merely over-deferring: _defer returns before
# every write AND before advance_census_state, so last_census_at is untouched,
# the next run re-mines the same window, re-hits the same cluster and aborts
# again -- a permanent census stall plus a stream of false infra_issue
# escalations claiming the account is capped.
# ---------------------------------------------------------------------------

def _cap_themed_verdict(marker, *, verified=True):
    """A WELL-FORMED verdict whose `reason` quotes a cap-themed cluster.

    This is ordinary healthy verifier output, not a banner: the model is
    adjudicating a confusion cluster that happens to be ABOUT usage limits.
    """
    return json.dumps(
        {
            "verified": verified,
            "reason": (
                "Confirmed against main: the watcher rotation was interrupted "
                f"by a {marker}; the 'continue where you left off' resume "
                "prompt appears at turns 17/27."
            ),
        }
    )


@pytest.mark.parametrize("marker", BLOCKING_BANNER_MARKERS)
def test_default_verify_fn_does_not_read_a_cap_themed_verdict_as_a_banner(marker):
    """A reply that PARSES into a verdict IS a verdict, whatever it quotes.

    Parametrized over every marker in the shared contract, so the guarantee is
    "no marker can be smuggled through a reason string", not "the two the test
    author happened to think of".
    """
    def invoke(prompt, model):
        return _cap_themed_verdict(marker)

    probe = _make_recording_probe(
        mod.HeadroomResult(ok=False, reason="must never be consulted on a parsed verdict")
    )
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(3), model="sonnet")

    assert [c["title"] for c in result["verified"]] == ["cluster-0", "cluster-1", "cluster-2"]
    assert result["rejected"] == []
    assert probe.calls == [], "a parsed verdict is never a cap, so nothing to confirm"


@pytest.mark.parametrize("marker", ["usage limit", "weekly limit"])
def test_default_verify_fn_reads_a_cap_themed_refutation_as_an_ordinary_rejection(marker):
    """`verified: false` about a cap-themed cluster is an ordinary verdict too.

    The refutation half of the same class: a well-formed rejection must reject,
    not abort the census.
    """
    def invoke(prompt, model):
        return _cap_themed_verdict(marker, verified=False)

    probe = _make_recording_probe(mod.HeadroomResult(ok=False, reason="must never be consulted"))
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(2), model="sonnet")

    assert result["verified"] == []
    assert [c["title"] for c in result["rejected"]] == ["cluster-0", "cluster-1"]
    assert probe.calls == []


def test_run_census_completes_when_the_real_verifier_returns_cap_themed_verdicts(tmp_path):
    """END-TO-END: a cap-themed verdict must not stall the census.

    Uses the REAL `_build_default_verify_fn`, not a fake, so the whole
    detector -> _defer -> abort-before-persistence chain is in the loop.

    The load-bearing assertions are that report_path / codebook_path /
    census_state_path all EXIST -- i.e. advance_census_state ran and
    last_census_at MOVED. That is what pins this defect as non-self-
    perpetuating: with the pre-parse scan in place the run aborts before every
    write, last_census_at is untouched, and the next run re-mines the same
    window, re-hits the same cluster and aborts again, forever.
    """
    verdict = _cap_themed_verdict("usage limit")

    def response_fn(prompt, model):
        if prompt == mod._HEADROOM_PROBE_PROMPT:
            return "pong"
        if prompt.startswith("You are the periodic-census verifier"):
            return verdict
        return _happy_invoke_response(prompt, model)

    fake_invoke = _make_fake_invoke(response_fn)
    fake_submit_fn = _make_fake_submit_fn()
    fake_escalate_fn = _make_fake_escalate_fn()

    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=fake_invoke,
        batch_source=[[_hand_digest("novel-verified", "a genuinely new confusion shape")]],
        verify_fn=mod._build_default_verify_fn(str(tmp_path), fake_invoke),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=fake_submit_fn,
        escalate_fn=fake_escalate_fn,
        commit=_make_fake_commit(),
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status != "deferred", (
        "a cap-themed verdict is healthy output, not a cap: "
        f"deferred_stage={outcome.deferred_stage!r} reason={outcome.reason!r}"
    )
    assert outcome.deferred_stage is None
    assert kwargs["report_path"].exists(), "the report was written"
    assert kwargs["codebook_path"].exists(), "the codebook merge was persisted"
    assert kwargs["census_state_path"].exists(), (
        "advance_census_state ran: last_census_at MOVED, so the next run does "
        "not re-mine this window and re-hit the same cluster forever"
    )
    assert len(fake_submit_fn.calls) == 1, "the verified cluster was filed"
    assert not any(
        c.get("category") == "infra_issue" for c in fake_escalate_fn.calls
    ), "no false 'the account is capped' escalation"


# ---------------------------------------------------------------------------
# task 3645 / REVIEW FIX, part 2/2: the RESIDUAL half of the same false-
# positive class.
#
# Confining the scan to parse FAILURES stops a well-formed verdict being
# re-read as a banner, but an unparseable reply is still arbitrary model
# output about cap-related subject matter: if the model ignores the prompt's
# "STRICT JSON ONLY (no prose, no markdown fences)" instruction and answers in
# prose about a usage-limit cluster, the marker still matches and the run
# still aborts with nothing written -- the identical self-perpetuating stall,
# just reached less often.
#
# The principle these pin: banner-matching CONTENT may only TRIGGER a probe,
# never itself decide that the account is capped.
# ---------------------------------------------------------------------------

_UNPARSEABLE_CAP_THEMED_PROSE = (
    "Verified. The rotation really was interrupted by a usage limit at turns "
    "17 and 27, per the digest."
)


def _prose_at_cluster_two_invoke():
    """`invoke` returning STRICT-JSON-violating prose for cluster #2 of 4.

    The prose is about a cap-themed cluster, so it matches a marker -- and it
    is unparseable, so it reaches the scan. This is the residual case that
    survives the parse-success split.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            return _UNPARSEABLE_CAP_THEMED_PROSE
        return _verdict()

    return invoke


def test_default_verify_fn_rejects_unparseable_cap_themed_prose_when_the_probe_says_healthy():
    """A healthy probe means the marker matched CONTENT, not a real cap.

    The fail-closed default for an unparseable verdict is preserved: the
    cluster rejects and the loop runs to completion, exactly as it would for
    any other malformed reply.
    """
    probe = _make_recording_probe(mod.HeadroomResult(ok=True))
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", _prose_at_cluster_two_invoke(), headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(probe.calls) == 1, "one confirming probe, only on the matched-marker path"
    assert len(result["verified"]) == 3, "the loop ran to completion"
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]


def test_default_verify_fn_raises_when_an_unparseable_banner_reply_is_probe_confirmed():
    """A probe that reports no capacity turns the same content into an abort."""
    probe = _make_recording_probe(
        mod.HeadroomResult(ok=False, reason="probe reply carries a banner marker: 'weekly limit'")
    )
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", _prose_at_cluster_two_invoke(), headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(mod.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage == "verify"
    assert exc.verified == 1
    assert exc.unverified == 3
    reason = (exc.reason or "").lower()
    assert "usage limit" in reason, "the reason names the matched marker"
    assert "probe" in reason, "and says a probe confirmed it, not that content decided it"


def test_default_verify_fn_never_infers_a_cap_from_content_without_a_probe():
    """With no probe there is no way to confirm, so a cap is never inferred.

    This is the whole principle in one assertion: content triggers, the probe
    decides. Absent a probe, banner-matching content is just another
    unparseable verdict.
    """
    verify_fn = mod._build_default_verify_fn("/tmp/root", _prose_at_cluster_two_invoke())

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(result["verified"]) == 3
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]


def test_default_verify_fn_raises_when_an_invocation_error_reprobes_capped():
    """Detector (b): a per-cluster invocation error, then a failing re-probe.

    The cluster that hit the cap must NOT be recorded as rejected. Recording
    it would be the defect in miniature: an infra failure written into the
    codebook as a verdict about the claim.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise coder.CoderInvocationError("claude CLI exited 1: simulated cap")
        return _verdict()

    probe = _make_recording_probe(
        mod.HeadroomResult(ok=False, reason="probe reply carries a banner marker")
    )
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(mod.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage == "verify"
    assert exc.verified == 1
    assert exc.unverified == 3
    assert len(probe.calls) == 1, "exactly one re-probe, on a path that already failed"


def test_default_verify_fn_still_rejects_when_the_reprobe_says_healthy():
    """Detector (b), negative: a healthy re-probe preserves the fail-closed contract.

    An invocation error with capacity still available means the claim really
    could not be verified. That must keep rejecting exactly as it does today
    -- the guard narrows the fail-closed default to non-cap causes, it does
    not remove it.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise coder.CoderInvocationError("claude CLI exited 1: unrelated outage")
        return _verdict()

    probe = _make_recording_probe(mod.HeadroomResult(ok=True))
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(probe.calls) == 1
    assert len(result["verified"]) == 3, "the loop ran to completion"
    rejected_titles = {c["title"] for c in result["rejected"]}
    assert rejected_titles == {"cluster-1"}, "the failing cluster rejects, as today"


def test_default_verify_fn_without_probe_args_behaves_exactly_as_before():
    """The pre-existing signature keeps working, unguarded and fail-closed.

    Every existing call site and test passes no probe; they must be
    completely unaffected by this work.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise coder.CoderInvocationError("claude CLI exited 1: simulated outage")
        return _verdict()

    verify_fn = mod._build_default_verify_fn("/tmp/root", invoke)
    result = verify_fn(_clusters(3), model="sonnet")

    assert len(result["verified"]) == 2
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]
    assert result["fixed"] == []


def test_run_census_aborts_cleanly_on_mid_verify_headroom_exhaustion(tmp_path, caplog):
    """A CensusHeadroomExhausted raised INSIDE verify_fn aborts before any write.

    `invoke` answers every headroom probe with "pong", so the stage-boundary
    gate passes and this test isolates the in-verify path.

    The three `not ... .exists()` assertions are the load-bearing ones: they
    prove no matrix was rendered from the truncated `verified` list, no
    reject_candidate burned a cluster that was never actually adjudicated, and
    last_census_at never advanced past a window that was not adjudicated.
    """
    def raising_verify_fn(clusters, *, model):
        raise mod.CensusHeadroomExhausted(
            stage="verify",
            reason="verify reply carries a banner marker: 'weekly limit'",
            verified=1,
            unverified=4,
        )

    fake_escalate_fn = _make_fake_escalate_fn()
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=raising_verify_fn,
        synthesize_fn=_poison("synthesize_fn"),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=fake_escalate_fn,
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_poison("commit"),
    )
    fake_submit_fn = kwargs["submit_fn"]

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**kwargs)

    assert outcome.status == "deferred"
    assert outcome.deferred_stage == "verify"
    assert outcome.unverified_clusters == 4

    # synthesize_fn and commit are _poison: reaching either would have raised.
    assert fake_submit_fn.calls == []
    assert not kwargs["report_path"].exists()
    assert not kwargs["codebook_path"].exists()
    assert not kwargs["census_state_path"].exists()

    # Exactly one escalation -- the mass-rejection detector must NOT also fire,
    # because the abort returns before it.
    assert len(fake_escalate_fn.calls) == 1, fake_escalate_fn.calls
    call = fake_escalate_fn.calls[0]
    assert call["category"] == "infra_issue"
    assert call["severity"] == "info"
    blob = (call.get("summary") or "") + (call.get("detail") or "")
    assert "verify" in blob.lower()
    assert "4" in blob, "the escalation names the unverified count"


def test_run_census_calls_verify_fn_once_with_the_full_cluster_list(tmp_path):
    """No-regression: the verify_fn(clusters, *, model) seam is UNCHANGED.

    The obvious reading of "re-probe between verify batches" would have
    run_census chunk the list and call verify_fn once per chunk — changing the
    contract of an injected seam for every fake in this file and any custom
    verifier. In-verify probing lives inside the DEFAULT verifier instead, so
    this stays one call with the whole list.
    """
    fake_verify_fn = _make_fake_verify_fn(verified_titles=set(_THREE_NOVEL_TITLES))
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=fake_verify_fn,
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_make_fake_escalate_fn(),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    assert len(fake_verify_fn.calls) == 1, "one call, not one per batch"
    assert len(fake_verify_fn.calls[0]["clusters"]) == 3, "the FULL cluster list"
    assert fake_verify_fn.calls[0]["model"] == "sonnet"


def test_default_verify_fn_probes_at_batch_boundaries_only():
    """Detector (c): the periodic backstop probes between batches, not per cluster.

    `invoke` returns a clean verdict every time, so neither the banner
    detector nor the exception detector can fire — this isolates the backstop.
    12 clusters at probe_every=5 means boundaries after clusters 5 and 10:
    twice, not twelve times. The probe is cheap but not free, and verification
    is already one agentic call per cluster.
    """
    probe = _make_recording_probe(mod.HeadroomResult(ok=True))
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    result = verify_fn(_clusters(12), model="sonnet")

    assert len(result["verified"]) == 12
    assert len(probe.calls) == 2, "boundaries after clusters 5 and 10 only"


def test_default_verify_fn_backstop_raises_at_the_first_failing_boundary():
    """The backstop aborts at its boundary, leaving the rest untouched."""
    probe = _make_recording_probe(
        mod.HeadroomResult(ok=False, reason="probe reply carries a banner marker")
    )
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    with pytest.raises(mod.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(12), model="sonnet")

    exc = excinfo.value
    assert exc.stage == "verify"
    assert exc.verified == 5
    assert exc.unverified == 7, "clusters 6-12, none of them adjudicated"
    assert len(probe.calls) == 1


def test_default_verify_fn_backstop_does_not_probe_after_the_last_cluster():
    """No probe when nothing remains -- it would guard a stage already over."""
    probe = _make_recording_probe(mod.HeadroomResult(ok=True))
    verify_fn = mod._build_default_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    verify_fn(_clusters(5), model="sonnet")

    assert probe.calls == [], "cluster 5 is the last one; there is nothing left to guard"


def test_report_cost_note_counts_real_headroom_probes(tmp_path):
    """The cost note reports the probes this run ACTUALLY made, not a literal 1.

    A legibility tool that under-reports its own spend, in the very artifact
    an operator reads to check --max-verify-clusters, is the same defect class
    this pipeline exists to find.

    A run with clusters to verify pays the preflight probe AND the
    stage-boundary gate, so the honest count is at least 2.
    """
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_three_novel_invoke),
        batch_source=[_three_novel_batch()],
        verify_fn=_make_fake_verify_fn(verified_titles=set(_THREE_NOVEL_TITLES)),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_make_fake_escalate_fn(),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    mod.run_census(**kwargs)

    cost_line = kwargs["report_path"].read_text(encoding="utf-8").split("## Cost", 1)[1]
    assert "headroom-probe=1" not in cost_line, "the hardcoded literal must be gone"
    assert "headroom-probe=2" in cost_line, "preflight + the pre-verify stage gate"


def test_report_cost_note_reports_one_probe_when_nothing_is_verified(tmp_path):
    """No clusters to verify -> the stage gate is skipped -> genuinely 1 probe.

    The count must track what happened, in both directions: reading 2 here
    would be over-reporting, which is the same dishonesty as under-reporting.
    """
    kwargs = _run_census_kwargs(
        tmp_path,
        invoke=_make_fake_invoke(_happy_invoke_response),
        batch_source=[[_hand_digest("dup-1", "nothing new here")]],
        verify_fn=_make_fake_verify_fn(),
        synthesize_fn=_make_fake_synthesize_fn(),
        submit_fn=_make_fake_submit_fn(),
        escalate_fn=_make_fake_escalate_fn(),
        status_fetcher=_make_fake_status_fetcher(0),
        commit=_make_fake_commit(),
    )

    mod.run_census(**kwargs)

    cost_line = kwargs["report_path"].read_text(encoding="utf-8").split("## Cost", 1)[1]
    assert "verify=0" in cost_line
    assert "headroom-probe=1" in cost_line
