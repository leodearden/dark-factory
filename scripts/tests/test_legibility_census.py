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

import json
import logging
from datetime import datetime, timezone

import pytest

import census as mod
import codebook
import coder
import config as config_mod
import digest as digest_mod
import inventory
from legibility import census_trigger

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
    kwargs = dict(
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
    kwargs = dict(
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
    assert "2" in saturation_section, "batches mined and the cap value must be stated"
    assert "cap" in lowered, "the operator cap must be named, never applied silently"
    # A capped report must be unreadable as full coverage.
    assert "partial" in lowered
    assert "not mined" in lowered


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

def _run_census_kwargs(tmp_path, **overrides):
    kwargs = dict(
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

    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
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


def test_post_mcp_tool_call_sends_streamable_http_accept_headers(monkeypatch):
    """Task 2953: the streamable-HTTP MCP transport 406s any tools/call POST
    lacking an Accept header covering both application/json and
    text/event-stream (verified live against a local MCP /mcp endpoint --
    shared by census.py's submit_task and escalate_info posters via this one
    function). httpx is imported lazily and is not importable in this test
    env, so a fake `httpx` module is injected via sys.modules."""
    import sys

    captured_kwargs = {}
    rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": {"ok": True}},
    }

    fake_httpx = type(sys)("httpx")

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse(rpc_response)

    fake_httpx.post = _fake_post
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

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

    def fake_invoke_cli(prompt, model, *, claude_bin=None, timeout=None):
        recorded.append(timeout)
        return "dummy"

    monkeypatch.setattr(coder, "_invoke_cli", fake_invoke_cli)

    cfg = _config_with_timeouts(111, 222, 333)
    mining, verify, synth = mod._build_stage_invokes(cfg)

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

    def fake_invoke_cli(prompt, model, *, claude_bin=None, timeout=None):
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
