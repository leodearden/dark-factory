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
from pathlib import Path

import pytest

import census as mod
import codebook
import coder
import config as config_mod
import digest as digest_mod

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
