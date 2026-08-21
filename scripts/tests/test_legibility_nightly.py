"""Tests for scripts/legibility/nightly.py — the nightly trickle pipeline
assembly (PRD task epsilon: plans/confusion-reduction-prd.md §5.5, decisions
7/8, boundary test §8.8).

Every stage is exercised behind a dependency-injection seam (invoke for the
LLM, status_fetcher for the census, poster for escalation, committer for
git) plus pytest-monkeypatched module functions — no test here ever touches
a real LLM, the machine-operated main git checkout, a live escalation
server, or real systemd. The one true end-to-end test
(test_run_nightly_happy_path_end_to_end) uses a real temp `git init` repo so
the commit path is genuinely exercised without risking main.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from legibility import (
    census_trigger,
    codebook,
    digest,
    nightly,
    trickle_state,
)
from legibility import (
    config as config_mod,
)
from legibility.config import load_config

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_trickle_state(tmp_path, monkeypatch):
    """Point XDG_STATE_HOME at tmp_path for EVERY test in this module.

    ``run_nightly`` records run state through ``trickle_state.record_run``
    on every exit path (task 3340), so without this an ordinary test run
    would write real state files for the fixture's synthetic
    ``project_id=testproj`` into the operator's real
    ``~/.local/state/dark-factory/legibility/``. Module-scoped rather than
    scoped to the recorder tests because EVERY run_nightly test now
    reaches the recorder — including the ones that assert on this module's
    WARNING records, which a failed real-home write would otherwise
    pollute.
    """
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'xdg-state'))


def _write_config(
    root: Path, *, project_id: str, escalation_port: int = 8199, cwd_prefixes=None,
    agent_transcript_roots=None, max_daily_digest_bytes: int | None = None,
) -> Path:
    """Write a minimal valid docs/legibility/legibility.yaml under *root*.

    When *agent_transcript_roots* is given, an ``agent_transcript_roots:``
    block is appended so the loaded cfg opts into archive-root enumeration
    (resolved against *root* by ``inventory.resolve_agent_transcript_roots``).

    When *max_daily_digest_bytes* is given, a ``budgets:`` block is appended
    so the loaded cfg carries a NON-stock daily byte budget. Omitted by
    default, so every existing caller keeps the stock 300_000 (the
    ``budgets`` block's own pydantic default) and its assertions stay
    valid. Squeezing this is how the totally-budget-suppressed night of
    2026-07-16..29 is replayed as sampler STATE (``selected == []`` with
    ``budget_skipped > 0``) through the supported config seam, rather than
    by resurrecting task 3268's already-fixed raw-transcript-bytes cost
    basis.
    """
    cwd_prefixes = cwd_prefixes if cwd_prefixes is not None else [str(root / "work")]
    legibility_dir = root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / "legibility.yaml"
    lines = [
        f"project_id: {project_id}",
        f"project_root: {root}",
        f"escalation_port: {escalation_port}",
        "cwd_prefixes:",
    ]
    lines += [f"  - {prefix}" for prefix in cwd_prefixes]
    if agent_transcript_roots is not None:
        lines.append("agent_transcript_roots:")
        lines += [f"  - {r}" for r in agent_transcript_roots]
    if max_daily_digest_bytes is not None:
        lines.append("budgets:")
        lines.append(f"  max_daily_digest_bytes: {max_daily_digest_bytes}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _encode_cwd(cwd: str) -> str:
    """Mirror inventory.encode_cwd (kept independent -- this is fixture code,
    not a reuse of the module under test).

    '/', '.' and '_' all map to '-'. "Independent" means SEPARATELY WRITTEN,
    not unchecked: test_legibility_inventory.py's TestEncoderLockstep pins
    this copy to the canonical (orchestrator.session_registry.encode_cwd) and
    to real on-disk dir names (task 3272, which found this fixture and three
    production copies all missing the '_' rule at once). Deliberately does
    NOT import the canonical -- a fixture that calls the code under test
    moves in lockstep with its bugs, which is exactly how that divergence
    survived a green suite."""
    return cwd.replace('/', '-').replace('.', '-').replace('_', '-')


def _write_transcript(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-1',
    user_text: str = 'please fix this confusing bug',
) -> None:
    """Write a minimal fixture session transcript JSONL with a genuine user
    turn plus a structural tool-error (nonzero confusion signal): a user
    turn, an assistant tool_use, and a user tool_result carrying
    ``is_error: true`` with 'No such file or directory' content (matches
    sampling._NOT_FOUND_PATTERNS too, for a score > 0 on more than one
    signal class).

    *user_text* overrides the first user turn. It exists so a test can write
    TWO transcripts that both survive to the digest phase: the sampler's
    ``dedupe_shapes`` fingerprints on (stratum, signal-shape,
    ``_normalize_first_turn(first_turn_text)``), so two transcripts differing
    only in ``session_id`` collapse to one and a multi-record night silently
    becomes a single-record one. Defaulted, so every existing caller is
    unchanged."""
    lines = [
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {'role': 'user', 'content': user_text},
        },
        {
            'type': 'assistant',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'assistant',
                'content': [
                    {'type': 'tool_use', 'id': 'tool-1', 'name': 'Bash',
                     'input': {'command': 'cat missing-file'}},
                ],
            },
        },
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool-1', 'is_error': True,
                     'content': 'No such file or directory'},
                ],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')


def _write_quiet_transcript(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-quiet',
) -> None:
    """A sibling of :func:`_write_transcript` that carries NO confusion signal.

    Same JSONL shape — a user turn, an assistant ``tool_use``, a user
    ``tool_result`` — but the exchange SUCCEEDS: no ``is_error``, no
    ``sampling._NOT_FOUND_PATTERNS`` text ('no such file' / 'not found' /
    'does not exist' / 'command not found'), no
    ``_SELF_CORRECT_PATTERNS`` in the assistant's text blocks, no
    ``_DF_GUARD_TEXT_PATTERNS`` / ``_DF_GUARD_TOOL_NAMES``, and no
    ``_INTERRUPT_PATTERNS``. So the record scores 0 and the sampler drops it
    as zero-signal BEFORE the budget phase — a genuinely quiet night, never
    a budget-skipped candidate. The score-0 claim is asserted (not assumed)
    in ``test_run_nightly_quiet_night_is_not_reported_as_suppressed``.
    """
    lines = [
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {'role': 'user', 'content': 'please add a docstring to the helper'},
        },
        {
            'type': 'assistant',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'Adding the docstring now.'},
                    {'type': 'tool_use', 'id': 'tool-1', 'name': 'Read',
                     'input': {'file_path': '/tmp/helper.py'}},
                ],
            },
        },
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool-1',
                     'content': 'def helper():\n    return 1\n'},
                ],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# step-3/4: select_scored_records (+ the stratified sampler)
# ---------------------------------------------------------------------------

def test_select_scored_records_includes_matching_session(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    scored = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    assert len(scored) == 1
    record = scored[0]
    assert record.session.path == session_path
    assert record.score > 0
    assert record.stratum


def test_select_scored_records_reads_archive_root(tmp_path):
    # End-to-end proof that the shipped agent_transcript_roots is LIVE with no
    # operator flip: resolve (cfg.project_root) -> enumerate (walk the archive)
    # -> read (iter_json_lines) -> classify (encoded worktree parent dir).
    # An archived role transcript at the production nested layout
    # <archive>/<task_id>/<enc>/<sid>.jsonl is enumerated ALONGSIDE an
    # (empty) ~/.claude/projects tree and classified 'orchestrated-task'.
    main_cwd = '/home/leo/src/dark-factory'
    worktree_cwd = '/home/leo/src/dark-factory/.worktrees/2573'
    encoded = '-home-leo-src-dark-factory--worktrees-2573'
    cfg = load_config(_write_config(
        tmp_path, project_id='dark_factory', cwd_prefixes=[main_cwd],
        agent_transcript_roots=['archive'],
    ))
    archived = tmp_path / 'archive' / '2573' / encoded / 'sess-x.jsonl'
    archived.parent.mkdir(parents=True, exist_ok=True)
    _write_transcript(
        archived, cwd=worktree_cwd, timestamp='2026-07-13T10:00:00Z', session_id='sess-x',
    )
    empty_projects_root = tmp_path / 'projects'  # never created — projects tree absent

    scored = nightly.select_scored_records(cfg, empty_projects_root, date(2026, 7, 13))

    assert len(scored) == 1
    assert scored[0].session.path == archived
    assert scored[0].stratum == 'orchestrated-task'


def test_select_scored_records_excludes_out_of_scope_sessions(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'

    # Wrong cwd -- outside cfg.cwd_prefixes.
    other_cwd = str(tmp_path / 'elsewhere')
    _write_transcript(
        projects_root / _encode_cwd(other_cwd) / 'session-2.jsonl',
        cwd=other_cwd, timestamp='2026-07-13T10:00:00Z',
    )

    # Right cwd, wrong date.
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-3.jsonl',
        cwd=work_cwd, timestamp='2026-07-01T10:00:00Z',
    )

    scored = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    assert scored == []


# ---------------------------------------------------------------------------
# task 3268: select_digest_sessions charges DIGEST bytes, not raw transcript size
# ---------------------------------------------------------------------------

def _write_multi_mb_transcript(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-big',
    error_lines: int = 3000,
) -> None:
    """A :func:`_write_transcript`-shaped session padded to a REAL on-disk
    size larger than the stock 300_000-byte daily budget.

    This is the shape reify sessions actually have — thousands of
    ``is_error`` tool_results — at ~400-900KB, the low end of the measured
    0.5-6.5MB range."""
    lines = [
        {
            'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'sessionId': session_id,
            'message': {'role': 'user', 'content': 'please fix this confusing bug'},
        },
    ]
    for i in range(error_lines):
        lines.append({
            'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'sessionId': session_id,
            'message': {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': f'tool-{i}', 'is_error': True,
                     'content': f'No such file or directory: /tmp/missing-{i} ' + 'x' * 180},
                ],
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')


def test_nightly_cap_is_an_alias_of_the_sampler_cap_not_a_second_literal():
    """``DEFAULT_MAX_DIGEST_BYTES`` and ``sampling.DEFAULT_DIGEST_MAX_BYTES``
    are a word-order swap apart. If they were independent literals, a drift
    between the cost basis and the render basis would be invisible to every
    test — so nightly's name BINDS to the sampler's rather than re-declaring
    it (reviewer_comprehensive, task 3268 amendment pass)."""
    assert nightly.DEFAULT_MAX_DIGEST_BYTES is nightly.sampling.DEFAULT_DIGEST_MAX_BYTES


def test_select_digest_sessions_charges_real_digest_bytes(tmp_path, monkeypatch):
    """The cost basis handed to the sampler must be the REAL rendered digest
    size -- what makes ``max_daily_digest_bytes`` mean what it says.

    This half only. The OTHER half of the guarantee -- that the charge uses
    the same ``max_bytes`` ``build_digests`` renders with -- cannot be
    detected here: ``_write_transcript``'s digest measures 446 bytes at both
    ``max_bytes=15360`` and ``max_bytes=999999``, so the cap never binds and
    any cap above ~446 (including ``build_digest``'s own
    default, had ``select_digest_sessions`` omitted the kwarg entirely) would
    satisfy this assertion (reviewer_comprehensive, task 3268 amendment
    pass). ``test_select_digest_sessions_costs_at_the_max_bytes_build_digests
    _renders_with`` asserts the cap DIRECTLY instead of inferring it from
    output size.
    """
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    captured = {}

    def spy_stratified_sample(records, config, *, cost_fn=None):
        captured['cost_fn'] = cost_fn
        captured['records'] = list(records)
        return nightly.sampling.SampleResult(
            selected=list(records), per_stratum_counts={}, zero_signal_dropped=0,
            bytes_used=0,
        )

    monkeypatch.setattr(nightly.sampling, 'stratified_sample', spy_stratified_sample)

    nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))

    cost_fn = captured['cost_fn']
    assert cost_fn is not None, 'select_digest_sessions passed no cost_fn'

    record = captured['records'][0]
    expected = len(
        digest.build_digest(
            record.path,
            agent_class_override=record.stratum,
            max_bytes=nightly.DEFAULT_MAX_DIGEST_BYTES,
        ).encode('utf-8')
    )
    assert cost_fn(record) == expected
    assert expected != record.size_bytes


def test_select_digest_sessions_costs_at_the_max_bytes_build_digests_renders_with(
    tmp_path, monkeypatch,
):
    """The cap the sampler CHARGES at and the cap ``build_digests`` RENDERS
    at must be the same number, or ``bytes_used`` describes a digest nobody
    produced.

    Asserted DIRECTLY off the recorded ``max_bytes`` kwarg rather than
    inferred from the rendered size -- the sibling test above cannot see this
    at all, because its fixture's digest (446 bytes) is far under the cap, so
    the cap never binds and a regression to any larger value would go
    undetected (reviewer_comprehensive, task 3268 amendment pass).
    """
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    caps = []

    def recording_build(path, **kwargs):
        caps.append(kwargs.get('max_bytes'))
        return 'a digest'

    # COST side: select_digest_sessions -> digest_byte_cost_fn, which
    # resolves digest.build_digest at call time.
    monkeypatch.setattr(nightly.sampling.digest, 'build_digest', recording_build)
    sample = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))
    assert caps == [nightly.DEFAULT_MAX_DIGEST_BYTES], (
        'the sampler must charge at the nightly cap, not at build_digest\'s own default'
    )

    # RENDER side: build_digests, called by run_nightly with no max_bytes
    # override, must reach the renderer with that SAME cap.
    caps.clear()
    nightly.build_digests(sample.selected, build=recording_build)
    assert caps == [nightly.DEFAULT_MAX_DIGEST_BYTES]


def test_select_digest_sessions_selects_multi_mb_session_under_stock_budget(tmp_path):
    """A session whose RAW transcript dwarfs the whole nightly budget must
    still be selected — the live defect, end to end.

    Measured basis: an 879,254-byte transcript of this shape renders to a
    15,123-byte digest, so it costs ~5% of the 300_000-byte budget rather
    than 293% of it. Charged at raw transcript size the single reserve
    group here is skipped whole, the greedy leftover fill has nothing to
    add, and select_digest_sessions selects nothing — which is exactly what
    the live nightly run did every night from 2026-07-16 to 2026-07-29.
    """
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    assert cfg.budgets.max_daily_digest_bytes == 300000, 'fixture must use the STOCK budget'

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-big.jsonl'
    _write_multi_mb_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z',
    )
    raw_size = session_path.stat().st_size
    assert raw_size > 300000, f'fixture must exceed the whole budget; got {raw_size}'

    sample = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))

    assert [r.path for r in sample.selected] == [session_path]


def test_select_digest_sessions_returns_the_whole_sample_result(tmp_path):
    """The accounting must reach the caller, not be dropped on the floor.

    Task 2573 computed ``budget_skipped`` correctly and task 2581's consumer
    (this function) discarded it by returning only ``.selected`` — so for 14
    nights (2026-07-16..29) a run that found real signal and threw ALL of it
    away on the byte budget reported exactly what a genuine no-change night
    reports. The whole ``SampleResult`` is the return value now.
    """
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    result = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))

    assert isinstance(result, nightly.sampling.SampleResult)
    # The four accounting fields the operator line and the suppression
    # predicate need, all reachable off the return value.
    assert result.total_records == 1
    assert result.zero_signal_dropped == 0
    assert result.budget_skipped == 0
    assert result.bytes_used > 0
    assert [r.path for r in result.selected] == [session_path]


# ---------------------------------------------------------------------------
# step-5/6: build_digests
# ---------------------------------------------------------------------------

def test_build_digests_happy_path(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')
    selected = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    digests, extractor_failures = nightly.build_digests(selected)

    assert extractor_failures == []
    assert len(digests) == 1
    assert digests[0].startswith('---\n')


def test_build_digests_reuses_the_costing_render(tmp_path):
    """A session rendered to CHARGE it against the byte budget must not be
    rendered a second time to emit it.

    Every selected session used to be rendered twice per nightly run, and
    ``digest._truncate_sections`` is super-linear in signal-item count (it
    pops ONE item and re-renders the whole digest per iteration): measured on
    the ``_write_multi_mb_transcript`` shape at 0.80s / 2.57s / 14.96s per
    render for 1000 / 2000 / 4000 error lines, i.e. roughly a 2x on the
    dominant cost of the job (reviewer_comprehensive, task 3268 amendment
    pass). Measured after the fix: 2 renders -> 1, 4.18s -> 2.23s, with a
    byte-identical digest.
    """
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    renders = []
    real_build = digest.build_digest

    def counting_build(path, **kwargs):
        renders.append(path)
        return real_build(path, **kwargs)

    # select_digest_sessions has no build seam of its own -- the costing
    # renderer is digest.build_digest, resolved at call time.
    digest.build_digest = counting_build
    try:
        rendered: dict = {}
        sample = nightly.select_digest_sessions(
            cfg, projects_root, date(2026, 7, 13), rendered=rendered,
        )
        assert renders == [session_path], 'costing must render exactly once'

        digests, extractor_failures = nightly.build_digests(
            sample.selected, build=counting_build, rendered=rendered,
        )
    finally:
        digest.build_digest = real_build

    # Still exactly one render in total: build_digests served the cache.
    assert renders == [session_path]
    assert extractor_failures == []
    assert digests == [
        digest.build_digest(
            session_path, agent_class_override=sample.selected[0].stratum,
            max_bytes=nightly.DEFAULT_MAX_DIGEST_BYTES,
        )
    ], 'the cached digest must be byte-identical to a fresh render'


def test_build_digests_re_renders_when_no_cache_is_shared(tmp_path):
    """The cache is opt-in: a caller that shares none still gets a correct
    digest, just at the cost of a second render."""
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    sample = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))
    digests, extractor_failures = nightly.build_digests(sample.selected)

    assert extractor_failures == []
    assert len(digests) == 1
    assert digests[0].startswith('---\n')


def test_run_nightly_shares_one_render_cache_across_both_stages(tmp_path, monkeypatch):
    """``run_nightly`` is the only production wiring of the two stages, so
    the reuse only pays off if IT threads one dict through both."""
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    seen = {}
    real_select = nightly.select_digest_sessions
    real_build = nightly.build_digests

    def spy_select(cfg, roots, target, *, rendered=None):
        seen['select'] = rendered
        return real_select(cfg, roots, target, rendered=rendered)

    def spy_build(selected, **kwargs):
        seen['build'] = kwargs.get('rendered')
        return real_build(selected, **kwargs)

    monkeypatch.setattr(nightly, 'select_digest_sessions', spy_select)
    monkeypatch.setattr(nightly, 'build_digests', spy_build)

    nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=date(2026, 7, 13),
        now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
        invoke=lambda prompt, model: '{"proposals": []}',
        status_fetcher=None,
        poster=lambda url, envelope: None,
    )

    assert seen['select'] is not None, 'run_nightly passed no render cache to costing'
    assert seen['build'] is seen['select'], 'the two stages must share ONE dict'
    assert seen['select'], 'the shared cache must actually hold the costing render'


def test_build_digests_isolates_extractor_crash(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')
    selected = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    def _crashing_build(path, *, agent_class_override=None, max_bytes=15360):
        raise ValueError('boom: corrupt transcript')

    digests, extractor_failures = nightly.build_digests(selected, build=_crashing_build)

    assert digests == []
    assert len(extractor_failures) == 1
    session_name, reason = extractor_failures[0]
    assert session_name == session_path.name
    assert 'boom' in reason


# ---------------------------------------------------------------------------
# step-7/8: git helpers (_git_status_changed, _git_commit_docs_only) --
# against a REAL tmp `git init` repo, plus a fake-runner ref-lock retry test.
# ---------------------------------------------------------------------------

_CODEBOOK_RELPATH = Path('docs') / 'legibility' / 'confusion-codebook.yaml'


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
    docs = repo / 'docs' / 'legibility'
    docs.mkdir(parents=True)
    (docs / 'confusion-codebook.yaml').write_text('version: 2\nentries: []\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'initial'], cwd=repo, check=True)
    return repo


def test_git_status_changed_detects_dirty_codebook(tmp_path):
    repo = _init_repo(tmp_path)

    assert nightly._git_status_changed(repo, _CODEBOOK_RELPATH) is False

    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )

    assert nightly._git_status_changed(repo, _CODEBOOK_RELPATH) is True


def test_git_commit_docs_only_commits_only_that_path(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )
    (repo / 'scratch.txt').write_text('unrelated wip\n', encoding='utf-8')

    result = nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'legibility: nightly sightings')

    assert result.ok is True
    assert result.sha

    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'scratch.txt' in status
    assert 'confusion-codebook.yaml' not in status

    shown = subprocess.run(
        ['git', 'show', '--stat', '--format=', 'HEAD'],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'confusion-codebook.yaml' in shown
    assert 'scratch.txt' not in shown


def test_git_commit_docs_only_noop_returns_not_ok(tmp_path):
    repo = _init_repo(tmp_path)

    result = nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'legibility: no-op')

    assert result.ok is False


def test_git_commit_docs_only_retries_on_ref_lock_then_succeeds(tmp_path):
    calls = {'commit_attempts': 0}

    def fake_runner(args, **kwargs):
        if 'commit' in args:
            calls['commit_attempts'] += 1
            if calls['commit_attempts'] < 3:
                return subprocess.CompletedProcess(args, 1, stdout='', stderr='fatal: cannot lock ref ...')
            return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
        if 'rev-parse' in args:
            return subprocess.CompletedProcess(args, 0, stdout='deadbeef\n', stderr='')
        raise AssertionError(f'unexpected git invocation: {args}')

    result = nightly._git_commit_docs_only(
        tmp_path, [_CODEBOOK_RELPATH], 'msg', runner=fake_runner, retries=5, backoff=0,
    )

    assert result.ok is True
    assert result.sha == 'deadbeef'
    assert calls['commit_attempts'] == 3


def test_git_commit_docs_only_gives_up_on_persistent_lock_error(tmp_path):
    def fake_runner(args, **kwargs):
        if 'commit' in args:
            return subprocess.CompletedProcess(args, 1, stdout='', stderr='fatal: cannot lock ref ...')
        raise AssertionError(f'unexpected git invocation: {args}')

    result = nightly._git_commit_docs_only(
        tmp_path, [_CODEBOOK_RELPATH], 'msg', runner=fake_runner, retries=3, backoff=0,
    )

    assert result.ok is False


def test_git_commit_docs_only_never_invokes_stash(tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(nightly.subprocess, 'run', spy_run)

    repo = _init_repo(tmp_path)
    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )

    nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'msg')

    assert not any('stash' in call for call in calls)


# ---------------------------------------------------------------------------
# step-9/10: escalation (_build_escalation_arguments, post_escalation)
# ---------------------------------------------------------------------------

def test_build_escalation_arguments_shape(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    arguments = nightly._build_escalation_arguments(cfg, 'summary text', 'detail text')

    assert arguments == {
        'task_id': 'legibility-trickle-proj_a',
        'agent_role': 'legibility-trickle',
        'category': 'infra_issue',
        'severity': 'info',
        'summary': 'summary text',
        'detail': 'detail text',
    }


def test_post_escalation_calls_poster_with_mcp_envelope(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', escalation_port=8199))
    calls = []

    def fake_poster(url, envelope):
        calls.append((url, envelope))

    ok = nightly.post_escalation(cfg, 'summary text', 'detail text', poster=fake_poster)

    assert ok is True
    assert len(calls) == 1
    url, envelope = calls[0]
    assert url == 'http://localhost:8199/mcp'
    assert envelope['method'] == 'tools/call'
    assert envelope['params']['name'] == 'escalate_info'
    assert envelope['params']['arguments'] == {
        'task_id': 'legibility-trickle-proj_a',
        'agent_role': 'legibility-trickle',
        'category': 'infra_issue',
        'severity': 'info',
        'summary': 'summary text',
        'detail': 'detail text',
    }


def test_post_escalation_is_best_effort_on_poster_failure(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def raising_poster(url, envelope):
        raise RuntimeError('escalation server unreachable')

    ok = nightly.post_escalation(cfg, 'summary text', 'detail text', poster=raising_poster)

    assert ok is False
# ---------------------------------------------------------------------------
# task 4511 step-1/2: post_escalation journals the summary/detail pair ITSELF,
# before the POST.
#
# The 2026-08-18 incident (esc-legibility-trickle-reify-3): the only copy of
# 'coder storm: 6/6 digests failed ... [Errno 2] No such file or directory:
# claude' lived inside the archived escalation JSON, and
# `journalctl --user -u legibility-trickle@reify.service` showed nothing but
# an unexplained benign 400 and `status=1/FAILURE`. Logging from inside
# `post_escalation` -- the only path to the POST -- makes the reason survive
# both a future branch that forgets to log and an escalation server that is
# down.
# ---------------------------------------------------------------------------

def test_post_escalation_journals_the_pair_before_the_post(tmp_path, caplog):
    """The escalation reason reaches the journal from `post_escalation`
    itself, at ERROR, carrying BOTH halves.

    The "before the POST" half is asserted from INSIDE the injected poster
    rather than after the call returns, because that is the property that
    actually survives an unreachable escalation server: a write ordered
    after a successful POST would still be missing in exactly the case an
    operator needs it.
    """
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))
    seen_inside_poster = []

    def _recording_poster(url, envelope):
        seen_inside_poster.extend(_nightly_warnings(caplog))

    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        ok = nightly.post_escalation(
            cfg, 'summary text', 'detail text', poster=_recording_poster,
        )

    assert ok is True

    loud = _nightly_warnings(caplog)
    assert len(loud) == 1, (
        f'expected exactly one record; got {[r.getMessage() for r in loud]}'
    )
    assert loud[0].levelno == logging.ERROR, (
        'a fail-loud escalation accompanies a non-zero exit, so it belongs '
        'in `journalctl -p err`'
    )
    message = loud[0].getMessage()
    assert 'summary text' in message
    assert 'detail text' in message, (
        'the DETAIL is the diagnosis; a summary-only journal line is exactly '
        f'what the 2026-08-18 incident already had. got {message!r}'
    )

    assert len(seen_inside_poster) == 1, (
        'the journal write must already have happened when the poster is '
        f'entered; saw {seen_inside_poster!r}'
    )


def test_post_escalation_journals_the_pair_even_when_the_poster_raises(
    tmp_path, caplog,
):
    """The escalation-server-down case, which loses the diagnosis entirely
    today. The pre-existing best-effort contract is unchanged and re-pinned
    here so the new write cannot be mistaken for a behaviour change."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def _raising_poster(url, envelope):
        raise RuntimeError('escalation server unreachable')

    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        ok = nightly.post_escalation(
            cfg, 'summary text', 'detail text', poster=_raising_poster,
        )

    # Unchanged best-effort behaviour: never raises, reports False, and
    # still emits its own post-failure WARNING.
    assert ok is False
    loud = _nightly_warnings(caplog)
    warned = [r.getMessage() for r in loud if r.levelno == logging.WARNING]
    assert any('escalation post failed' in m for m in warned), warned

    errors = [r for r in loud if r.levelno == logging.ERROR]
    assert len(errors) == 1, (
        f'expected exactly one ERROR; got {[r.getMessage() for r in errors]}'
    )
    message = errors[0].getMessage()
    assert 'summary text' in message
    assert 'detail text' in message, (
        'a down escalation server must not cost the diagnosis -- that is the '
        f'whole point of journaling before the POST. got {message!r}'
    )


class _FakeHttpxResponse:
    """A 200 `application/json` reply with an empty JSON-RPC body.

    `status_code`, `headers` and `json()` were added for task 3644:
    `_default_poster` now delegates to `census_trigger.post_mcp_envelope`,
    which reads `status_code` (to detect the stateful escalation server's 400
    "Missing session ID") and `content-type` (to detect an SSE-framed body)
    before decoding. Without them this fake would die on an AttributeError
    and stop exercising the real code path -- so the task-2953 header
    contract below keeps being asserted against the transport that actually
    ships, not against a fake the implementation has outgrown."""

    status_code = 200
    headers = {'content-type': 'application/json'}

    def raise_for_status(self):
        pass

    def json(self):
        return {}


def test_default_poster_sends_streamable_http_accept_headers(install_fake_httpx):
    """Task 2953: the streamable-HTTP MCP transport 406s any tools/call POST
    lacking an Accept header covering both application/json and
    text/event-stream (verified live against a local MCP /mcp endpoint).
    `_default_poster`'s httpx import is lazy, but httpx IS importable here --
    a direct dependency of `shared` (shared/pyproject.toml, `httpx>=0.27`,
    task 2965) -- so an un-faked call would really hit the network. The
    shared `install_fake_httpx` fixture substitutes a stub so the outbound
    request shape is assertable independent of any live listener."""
    captured_kwargs = {}

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse()

    install_fake_httpx(_fake_post)

    nightly._default_poster('http://localhost:8199/mcp', {'jsonrpc': '2.0'})

    headers = captured_kwargs.get('headers') or {}
    assert 'application/json' in headers.get('Accept', '')
    assert 'text/event-stream' in headers.get('Accept', '')
    # Content-Type is part of the same transport contract -- pin it too so a
    # future edit dropping it can't pass on the Accept assertions alone.
    assert headers.get('Content-Type') == 'application/json'
    # The envelope must still ride along unchanged on the same POST.
    assert captured_kwargs.get('json') == {'jsonrpc': '2.0'}


# ---------------------------------------------------------------------------
# step-11/12: evaluate_census_step
# ---------------------------------------------------------------------------

def test_evaluate_census_step_no_fire_never_launches(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=False, reasons=['max-interval: 1.0d (threshold 10d)'])

    launcher_calls = []
    line, fire = nightly.evaluate_census_step(
        cfg, now=None, status_fetcher=None, decide=fake_decide,
        entrypoint_exists=lambda: True, launcher=lambda: launcher_calls.append(1),
    )

    assert fire is False
    assert 'NO-FIRE' in line
    assert launcher_calls == []


def test_evaluate_census_step_fire_without_entrypoint_logs_loud_no_launch(tmp_path, caplog):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=['max-interval: 11.0d -> FIRE'])

    launcher_calls = []
    with caplog.at_level('WARNING'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=lambda: False, launcher=lambda: launcher_calls.append(1),
        )

    assert fire is True
    assert 'FIRE' in line
    assert launcher_calls == []
    assert any('FIRE-WITHOUT-LAUNCH' in record.message for record in caplog.records)


def test_evaluate_census_step_fire_with_entrypoint_launches(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=['max-interval: 11.0d -> FIRE'])

    launcher_calls = []
    line, fire = nightly.evaluate_census_step(
        cfg, now=None, status_fetcher=None, decide=fake_decide,
        entrypoint_exists=lambda: True, launcher=lambda: launcher_calls.append(1),
    )

    assert fire is True
    assert launcher_calls == [1]


def test_evaluate_census_step_logs_the_decision_before_launching(tmp_path, caplog):
    """task 4148 (amendment): the decision line must reach the journal BEFORE
    the census subprocess starts, and must survive a launcher that dies.

    `launcher()` runs the census SYNCHRONOUSLY -- `_default_census_launcher`
    subprocess-runs census.py, whose stages carry the 120/900/1800s timeouts
    in config.py's Timeouts. So a line logged at run_nightly's call site, i.e.
    after this function has returned, reaches the journal only once the whole
    census has finished (tens of minutes later) and never at all if the unit
    is killed, times out, or the box reboots mid-census -- exactly the FIRE
    case the line exists to make visible. An operator tailing a RUNNING unit
    would otherwise watch a census start with no logged reason for it.
    """
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=[
            'tasks-landed: 130 landed since last census (threshold 120) -> FIRE',
        ])

    # Snapshots the journal as the census subprocess would start, then dies
    # the way a timed-out/killed census does.
    logged_at_launch = []

    def _dying_launcher():
        logged_at_launch.extend(r.getMessage() for r in caplog.records)
        raise subprocess.TimeoutExpired(cmd='census.py', timeout=1800)

    with caplog.at_level(logging.INFO, logger='legibility.nightly'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=lambda: True, launcher=_dying_launcher,
        )

    assert fire is True

    # The reason is on the journal BEFORE the launch, not after it.
    assert any(
        'legibility trickle: census trigger: FIRE' in m for m in logged_at_launch
    ), logged_at_launch
    assert any('tasks-landed: 130 landed' in m for m in logged_at_launch), logged_at_launch

    # ...and a launcher that never returns cleanly cannot take it away.
    messages = [r.getMessage() for r in caplog.records]
    assert any(f'legibility trickle: {line}' == m for m in messages), messages
    # Exactly once: run_nightly no longer logs its own copy at the call site.
    assert len([m for m in messages if m.startswith('legibility trickle: census trigger:')]) == 1


def test_evaluate_census_step_logs_a_failed_evaluation_too(tmp_path, caplog):
    """task 4148 (amendment): ONE log site covers both paths, so the journal
    line and the `census_line` returned on NightlyResult cannot drift -- a
    failed evaluation reports its synthetic NO-FIRE line on the same channel
    an operator greps for the healthy one."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        raise TypeError("unsupported operand type(s) for -: 'int' and 'str'")

    with caplog.at_level(logging.INFO, logger='legibility.nightly'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=lambda: True, launcher=lambda: None,
        )

    assert fire is False
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert [m for m in infos if m == f'legibility trickle: {line}'], infos
    assert 'trigger evaluation failed' in line


def test_evaluate_census_step_survives_a_raising_decide(tmp_path, caplog):
    """task 4085: the docstring's "this function never raises and never fails
    the run" must hold for the DECIDE call, not only for the launcher call.

    Today it guards only `launcher()` and parenthetically ASSUMES `decide` is
    fail-safe ("never raises -- fail-safe") -- a promise about a callee it
    cannot enforce, and `decide` is an injected seam any caller can replace.
    The exception raised here is the real one task 4085 reports, from an
    unvalidated `last_census_done_count` baseline reaching
    `current_done - baseline`.

    Two properties, and the second is a safety property rather than a style
    one: a failed evaluation has established NOTHING, so it must not be able to
    start an expensive census -- hence the return happens before
    `entrypoint_exists` is even consulted."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        raise TypeError("unsupported operand type(s) for -: 'int' and 'str'")

    launcher_calls = []
    entrypoint_calls = []

    def _recording_entrypoint_exists():
        entrypoint_calls.append(1)
        return True

    with caplog.at_level('WARNING', logger='legibility.nightly'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=_recording_entrypoint_exists,
            launcher=lambda: launcher_calls.append(1),
        )

    assert fire is False
    # A downstream reader of NightlyResult.census_line must still see a
    # well-formed decision line -- the one case an operator most needs to read
    # is the worst possible place to emit a novel shape.
    assert 'NO-FIRE' in line
    assert line.startswith('census trigger: ')
    assert 'trigger evaluation failed' in line
    assert 'TypeError' in line

    assert launcher_calls == [], 'a failed evaluation must never launch a census'
    assert entrypoint_calls == [], 'the failure path must return before the launcher block'

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert 'TypeError' in warnings[0].getMessage()


def test_evaluate_census_step_bounds_a_huge_decide_failure(tmp_path, caplog):
    """task 4085 (amendment): the fail-safe path must keep the module's
    one-line guarantee. `decide` is an injected seam and an escaping
    exception's message is arbitrary -- a StatusFetchUnavailable chained from
    a big get_statuses payload, a multi-line YAML error -- yet BOTH sinks here
    are single-line: the WARNING is one nightly journal line and `census_line`
    is a one-line field on NightlyResult. Dumping a whole payload into either
    is the opposite of the legible failure this trigger exists for."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        raise RuntimeError('boom ' + 'x' * 50_000 + '\nsecond line')

    with caplog.at_level('WARNING', logger='legibility.nightly'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=lambda: True, launcher=lambda: None,
        )

    assert fire is False
    # Bounded, still well-formed, and still names the fault.
    assert len(line) < 1000
    assert line.startswith('census trigger: NO-FIRE -- trigger evaluation failed')
    assert 'RuntimeError' in line
    assert 'repr truncated' in line
    # One line means one line: the repr escapes the embedded newline.
    assert '\n' not in line

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert len(message) < 1000
    assert '\n' not in message


def test_default_census_launcher_logs_loud_on_nonzero_exit(monkeypatch, caplog):
    """A non-zero census subprocess exit must be logged LOUD (naming the
    returncode) rather than silently discarded -- the silent-census incident
    this fixes (PRD decision 8: degradation never silent). The census files
    its OWN escalation; the launcher's loud log is the trickle-side trace."""
    def fake(args, **kwargs):
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(nightly.subprocess, "run", fake)

    with caplog.at_level("WARNING", logger="legibility.nightly"):
        result = nightly._default_census_launcher()

    assert result is None, "the launcher never raises and returns None (never-crash-the-nightly)"
    assert any(
        "census" in r.getMessage() and "1" in r.getMessage()
        for r in caplog.records if r.levelno >= logging.WARNING
    ), "a non-zero census exit must be logged loud, naming the returncode"


def test_default_census_launcher_quiet_on_zero_exit(monkeypatch, caplog):
    """A zero-exit census (a deferred/no-fire outcome, or a clean run) must
    stay quiet -- only a genuine non-zero exit gets the loud failure log."""
    def fake0(args, **kwargs):
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nightly.subprocess, "run", fake0)

    with caplog.at_level("WARNING", logger="legibility.nightly"):
        result = nightly._default_census_launcher()

    assert result is None
    assert not any(
        "census" in r.getMessage()
        for r in caplog.records if r.levelno >= logging.WARNING
    ), "a zero-exit census must not emit a census-failure warning"


# ---------------------------------------------------------------------------
# task 4148: run_nightly must DEFAULT the census status_fetcher seam
#
# The production path is the systemd `nightly.py run --project-id %i`
# ExecStart -> main() -> run_nightly, and main() holds only --config /
# --project-id -- no project_root to build a fetcher from. So NOTHING on that
# path ever constructed one, `status_fetcher=None` reached
# compute_tasks_landed on every real run, and its `status_fetcher is None`
# fail-safe arm logged "tasks-landed: no status_fetcher configured" instead of
# a delta: condition (b) could not fire AT ALL. Every OTHER seam in
# run_nightly already resolves None to its real implementation (committer ->
# _git_commit_docs_only, recorder -> trickle_state.record_run, poster ->
# _default_poster); status_fetcher alone resolved to nothing, which is how
# three prior repairs of this exact code (2953, 3291, 4085) each left the
# wiring loop open.
# ---------------------------------------------------------------------------

class TestRunNightlyDefaultsTheCensusStatusFetcher:
    """Pin run_nightly's status_fetcher seam: None means "build the real
    MCP-backed fetcher for cfg.project_root", an explicit value is honoured.

    The spy replaces ``evaluate_census_step`` ITSELF rather than the inner
    ``decide`` seam. That is load-bearing: task 4085 wrapped the inner
    ``decide`` call in a never-raises try/except (nightly.py:590-606) that
    converts any failure into a synthetic ``NO-FIRE`` line, so a spy placed
    inside it could turn a genuine wiring fault into a quiet pass.
    """

    @staticmethod
    def _install_spies(monkeypatch):
        """Record the factory's argument and what evaluate_census_step got.

        Returns ``(sentinel, factory_calls, seen)`` where *sentinel* is the
        object the stubbed factory hands back -- asserted by IDENTITY, so
        "some fetcher got built" cannot pass for "the real factory's fetcher
        got through".
        """
        sentinel = object()
        factory_calls = []
        seen = {}

        def _recording_factory(project_root):
            factory_calls.append(project_root)
            return sentinel

        def _spy_evaluate(cfg, *, now=None, status_fetcher=None):
            seen['status_fetcher'] = status_fetcher
            return 'census trigger: NO-FIRE -- stub', False

        monkeypatch.setattr(
            nightly.census_trigger, 'default_status_fetcher', _recording_factory,
        )
        monkeypatch.setattr(nightly, 'evaluate_census_step', _spy_evaluate)
        return sentinel, factory_calls, seen

    @staticmethod
    def _run_quiet_night(tmp_path, **kwargs):
        """Drive run_nightly over the light quiet-night path (empty
        projects_root -> empty sample -> no LLM, no commit) and return the
        loaded config, so a caller can assert against ``cfg.project_root``."""
        config_path = _write_config(tmp_path, project_id='proj_a')
        nightly.run_nightly(
            config_path=config_path,
            projects_root=tmp_path / 'projects',
            target_date=date(2026, 7, 13),
            invoke=lambda prompt, model: '{"proposals": []}',
            poster=lambda url, envelope: None,
            **kwargs,
        )
        return load_config(config_path)

    def test_defaults_to_the_real_mcp_backed_fetcher(self, tmp_path, monkeypatch):
        sentinel, _factory_calls, seen = self._install_spies(monkeypatch)

        # No status_fetcher argument at all -- exactly what main() passes.
        self._run_quiet_night(tmp_path)

        assert seen['status_fetcher'] is sentinel, (
            'run_nightly handed evaluate_census_step '
            f'{seen["status_fetcher"]!r} instead of the fetcher built by '
            'census_trigger.default_status_fetcher -- with None, '
            'compute_tasks_landed fails safe and condition (b) can never fire'
        )

    def test_the_fetcher_is_built_for_the_configs_project_root(self, tmp_path, monkeypatch):
        _sentinel, factory_calls, _seen = self._install_spies(monkeypatch)

        cfg = self._run_quiet_night(tmp_path)

        assert len(factory_calls) == 1, (
            'default_status_fetcher must be called exactly once per run, '
            f'got {len(factory_calls)} call(s)'
        )
        # The SAME project_root evaluate_census_step hands to decide()
        # (nightly.py:591), so the fetcher and the decision can never disagree
        # about which project they are for.
        assert factory_calls[0] == cfg.project_root
        # Absolute is the wire contract task 3291 fixed: fused-memory's
        # _normalize_project_root hard-rejects any relative path.
        assert Path(str(factory_calls[0])).is_absolute()

    def test_an_injected_fetcher_is_not_overridden(self, tmp_path, monkeypatch):
        """DI-seam regression guard: the default must not clobber an explicit
        fetcher (every run_nightly test that injects one depends on this)."""
        _sentinel, factory_calls, seen = self._install_spies(monkeypatch)

        def my_fake():
            return {'statuses': {}}

        self._run_quiet_night(tmp_path, status_fetcher=my_fake)

        assert seen['status_fetcher'] is my_fake
        assert factory_calls == [], (
            'an injected status_fetcher must short-circuit the default factory'
        )


# ---------------------------------------------------------------------------
# step-1/2: resolve_config_path
# ---------------------------------------------------------------------------

def test_resolve_config_path_matches_by_project_id_field(tmp_path):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")

    resolved = nightly.resolve_config_path("proj_a", search_roots=[tmp_path])

    assert resolved == config_a


def test_resolve_config_path_unknown_id_raises(tmp_path):
    _write_config(tmp_path / "proj-a", project_id="proj_a")

    with pytest.raises(FileNotFoundError):
        nightly.resolve_config_path("does-not-exist", search_roots=[tmp_path])


def test_resolve_config_path_env_override(tmp_path, monkeypatch):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")
    monkeypatch.setenv("LEGIBILITY_SEARCH_ROOTS", str(tmp_path))

    resolved = nightly.resolve_config_path("proj_a")

    assert resolved == config_a


# ---------------------------------------------------------------------------
# step-13/14: run_nightly -- end-to-end happy path (real tmp git repo)
# ---------------------------------------------------------------------------

def _write_known_cause_codebook(repo: Path) -> Path:
    """Write a v2 confusion-codebook.yaml under *repo* with one entry,
    id='known-cause', sightings: [] -- the pre-existing codebook state for
    the end-to-end fixture (step-13)."""
    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    cb = {
        'version': 2,
        'entries': [
            {
                'id': 'known-cause',
                'title': 'Known Cause',
                'severity': 'medium',
                'status': 'open',
                'origin_phase': 'implement',
                'manifested_phase': 'implement',
                'sightings': [],
            },
        ],
        'candidates': [],
    }
    codebook.dump(cb, codebook_path)
    return codebook_path


def _init_e2e_repo(tmp_path: Path, *, work_cwd: str) -> tuple[Path, Path]:
    """Build a tmp git repo with a committed legibility.yaml (project_id=
    'testproj') + a v2 codebook carrying one entry (id='known-cause',
    sightings: []), tree clean -- the end-to-end fixture (step-13)."""
    repo = tmp_path / 'e2e-repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)

    config_path = _write_config(
        repo, project_id='testproj', escalation_port=8199, cwd_prefixes=[work_cwd],
    )
    _write_known_cause_codebook(repo)

    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'initial'], cwd=repo, check=True)
    return repo, config_path


def _fake_invoke_known_cause(prompt: str, model: str) -> str:
    return json.dumps({
        'matches': [{
            'entry_id': 'known-cause',
            'origin_phase': 'implement',
            'manifested_phase': 'implement',
            'note': 'matched the known cause',
        }],
        'candidates': [],
    })


def test_run_nightly_happy_path_end_to_end(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code == 0
    assert result.commit_made is True
    assert result.applied == 1
    assert result.coder_status == 'ok'
    assert result.census_line is not None and 'NO-FIRE' in result.census_line
    assert escalation_calls == []

    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert len(after_log) == len(before_log) + 1

    shown = subprocess.run(
        ['git', 'show', '--stat', '--format=', 'HEAD'],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'confusion-codebook.yaml' in shown
    touched_files = [line for line in shown.splitlines() if '|' in line]
    assert len(touched_files) == 1

    committed = codebook.load(repo / 'docs' / 'legibility' / 'confusion-codebook.yaml')
    entry = next(e for e in committed['entries'] if e['id'] == 'known-cause')
    assert len(entry['sightings']) == 1
    sighting = entry['sightings'][0]
    assert sighting['date'] == '2026-07-13'
    assert sighting['project'] == 'testproj'
    assert sighting['session'] == 'session-1'

    # Idempotency: a second run makes no new commit and leaves exactly one sighting.
    result_2 = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result_2.exit_code == 0
    assert result_2.commit_made is False
    assert result_2.applied == 0
    assert escalation_calls == []

    after_log_2 = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log_2 == after_log

    committed_2 = codebook.load(repo / 'docs' / 'legibility' / 'confusion-codebook.yaml')
    entry_2 = next(e for e in committed_2['entries'] if e['id'] == 'known-cause')
    assert len(entry_2['sightings']) == 1


# ---------------------------------------------------------------------------
# task 3270: run_nightly emits the sampler accounting on EVERY run
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = ('enumerated=', 'zero_signal_dropped=', 'below_sampling_cut=',
                 'budget_skipped=', 'selected=', 'bytes_used=')


def _summary_lines(caplog):
    """Every captured record carrying the sampler summary line's keys."""
    return [
        r.getMessage() for r in caplog.records
        if all(key in r.getMessage() for key in _SUMMARY_KEYS)
    ]


def test_run_nightly_logs_the_sampler_summary_on_a_healthy_night(tmp_path, caplog):
    """The operator's grep anchor is emitted on EVERY run, healthy included.

    Without it, the only trace a night leaves in the journal is whether it
    committed — so a night that sampled real signal and a night that sampled
    nothing at all are the same observation.
    """
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
        cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    with caplog.at_level('INFO', logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_known_cause,
            status_fetcher=None,
            poster=lambda url, envelope: None,
        )

    assert result.exit_code == 0
    lines = _summary_lines(caplog)
    assert len(lines) == 1, f'expected exactly one summary line, got {lines}'
    line = lines[0]

    # The REAL numbers, not a zeros template: one session enumerated, one
    # affordable and selected.
    assert 'enumerated=1' in line
    assert 'selected=1' in line
    assert 'budget_skipped=0' in line
    assert f'/{load_config(config_path).budgets.max_daily_digest_bytes}' in line

    # One journal interleaves every project's timer, so the line has to say
    # which project and which night it is describing.
    assert 'testproj' in line
    assert '2026-07-13' in line


def test_run_nightly_logs_the_sampler_summary_when_there_are_no_sessions(tmp_path, caplog):
    """A night with nothing to sample reports zeros — not silence."""
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'  # never populated

    with caplog.at_level('INFO', logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_known_cause,
            status_fetcher=None,
            poster=lambda url, envelope: None,
        )

    assert result.exit_code == 0
    lines = _summary_lines(caplog)
    assert len(lines) == 1, f'expected exactly one summary line, got {lines}'
    line = lines[0]
    assert 'enumerated=0' in line
    assert 'selected=0' in line
    assert 'budget_skipped=0' in line
    assert 'testproj' in line
    assert '2026-07-13' in line


def _nightly_warnings(caplog):
    """Records at >= WARNING on this module's OWN logger.

    Filtered by logger name deliberately: ``legibility.census_trigger`` emits
    its own fail-safe WARNING on a fixture with no census baseline, which says
    nothing about whether the trickle reported suppression.
    """
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == 'legibility.nightly'
    ]


def test_run_nightly_reports_a_totally_budget_suppressed_night(tmp_path, caplog):
    """THE ACCEPTANCE REGRESSION (task 3270).

    Replays the sampler STATE that every night from 2026-07-16 to 2026-07-29
    actually produced -- real signal enumerated, then every candidate
    discarded on the byte budget (``selected == []`` with
    ``budget_skipped > 0``) -- and asserts it is now reported at every layer
    an operator can observe: a structured flag on the result, a WARNING in
    the journal, and a durable escalation.

    Achievability basis, not a guessed threshold: ``_write_transcript``'s
    digest measures 446 bytes (recorded at the top of
    ``test_select_digest_sessions_charges_real_digest_bytes``), so a 10-byte
    budget cannot fit the single reserve group and skips it whole. The
    squeeze goes through the supported ``budgets`` config seam rather than
    resurrecting task 3268's already-fixed raw-transcript-bytes cost basis,
    which is no longer reachable from main.
    """
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    # Squeeze the daily budget below one digest. Overwrites _init_e2e_repo's
    # own config in place, so run_nightly loads the squeezed one.
    _write_config(
        repo, project_id='testproj', escalation_port=8199, cwd_prefixes=[work_cwd],
        max_daily_digest_bytes=10,
    )
    assert load_config(config_path).budgets.max_daily_digest_bytes == 10

    projects_root = tmp_path / 'projects'
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
        cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    escalation_calls = []
    with caplog.at_level('INFO', logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_known_cause,
            status_fetcher=None,
            poster=lambda url, envelope: escalation_calls.append((url, envelope)),
        )

    # Exit code stays 0 ON PURPOSE: a non-zero exit would flip
    # legibility-trickle@testproj.service to failed and make
    # check_trickle_liveness.sh scream every night about a timer that is in
    # fact running perfectly.
    assert result.exit_code == 0
    assert result.budget_suppressed is True
    assert result.escalated is True

    # Structured facts, not log-scraping: ONE durable escalation, on the
    # existing decision-8 envelope, unchanged.
    assert len(escalation_calls) == 1
    _url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert arguments['severity'] == 'info'
    summary = arguments['summary'].lower()
    assert 'budget' in summary
    assert 'suppress' in summary
    detail = arguments['detail']
    assert 'budget_skipped=1' in detail
    assert 'selected=0' in detail
    assert '/10' in detail, 'the detail must name the byte budget that did the cutting'

    # And loud in the journal, on this module's own logger.
    loud = _nightly_warnings(caplog)
    assert len(loud) >= 1
    assert any(
        'budget' in r.getMessage().lower() and 'budget_skipped=1' in r.getMessage()
        for r in loud
    ), f'expected a WARNING naming the suppression counts; got {[r.getMessage() for r in loud]}'

    # Nothing was written: no digests were affordable, so no coding, no dump,
    # no commit.
    assert result.commit_made is False
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log


def test_run_nightly_quiet_night_is_not_reported_as_suppressed(tmp_path, caplog):
    """The CONTRAST case -- and the actual property under test.

    A genuinely quiet night (sessions existed, none carried any confusion
    signal, so none ever reached the budget phase) must stay quiet: no
    suppression flag, no escalation, nothing loud in the journal. Paired with
    the test above, this is what proves the two nights are now
    DISTINGUISHABLE rather than merely proving that some string was logged.
    """
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    cfg = load_config(config_path)
    assert cfg.budgets.max_daily_digest_bytes == 300000, 'contrast case must use the STOCK budget'

    projects_root = tmp_path / 'projects'
    for session_id in ('session-quiet-a', 'session-quiet-b'):
        _write_quiet_transcript(
            projects_root / _encode_cwd(work_cwd) / f'{session_id}.jsonl',
            cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id=session_id,
        )

    # The fixture's zero-signal claim, ASSERTED rather than assumed: these
    # records are dropped before the budget phase, so they can never be
    # budget-skipped candidates.
    scored = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))
    assert len(scored) == 2
    assert [r.score for r in scored] == [0, 0]

    escalation_calls = []
    with caplog.at_level('INFO', logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_known_cause,
            status_fetcher=None,
            poster=lambda url, envelope: escalation_calls.append((url, envelope)),
        )

    assert result.exit_code == 0
    assert result.budget_suppressed is False
    assert result.escalated is False
    assert escalation_calls == []
    assert _nightly_warnings(caplog) == [], 'a quiet night must emit nothing loud'

    # ...while the always-on INFO line still reports exactly WHY the night
    # was empty: nothing had signal, and the budget cut nothing.
    lines = _summary_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert 'enumerated=2' in line
    assert 'zero_signal_dropped=2' in line
    assert 'budget_skipped=0' in line
    assert 'selected=0' in line


# ---------------------------------------------------------------------------
# step-15/16: run_nightly -- fail-loud on coder storm (decision 8, §8.6)
# ---------------------------------------------------------------------------

def _fake_invoke_unparseable(prompt: str, model: str) -> str:
    return 'not valid json at all'


def test_run_nightly_fail_loud_on_coder_storm(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    # A single selected session whose invoke fails to parse -> 1/1 failed ->
    # failed/total (1.0) strictly exceeds 0.5 -> coder.code_digests reports
    # status="failure" (the storm threshold), even with just one digest.
    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_unparseable,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code != 0
    assert result.coder_status == 'failure'
    assert result.commit_made is False
    assert result.escalated is True

    assert len(escalation_calls) == 1
    url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert 'storm' in arguments['summary'].lower()
    assert '1/1' in arguments['summary']

    # Merge/dump/commit skipped entirely: codebook untouched, no new commit.
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log


# ---------------------------------------------------------------------------
# step-19/20: run_nightly -- fail-loud on commit failure (decision 8)
# ---------------------------------------------------------------------------

def _failing_committer(repo, paths, message):
    """A fake committer simulating a persistent ref-lock/commit failure --
    mirrors _git_commit_docs_only's own ok=False-after-final-attempt
    contract without touching a real repo."""
    return nightly.GitCommitResult(
        ok=False, sha=None, stderr='fatal: cannot lock ref (simulated)', attempts=5,
    )


def test_run_nightly_fail_loud_on_commit_failure(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    # A valid matching invoke -> the merge/dump genuinely happens; only the
    # commit attempt itself fails (persistent ref-lock, simulated).
    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
        committer=_failing_committer,
    )

    assert result.exit_code != 0
    assert result.commit_made is False
    assert result.escalated is True
    assert result.applied == 1  # the merge/dump DID happen before the commit attempt

    assert len(escalation_calls) == 1
    url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert 'commit' in arguments['summary'].lower()

    # No NEW commit exists -- the escalation + non-zero exit is the loud
    # signal (the dump already landed in the working tree, uncommitted).
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log

    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'confusion-codebook.yaml' in status


# ---------------------------------------------------------------------------
# task-4144: run_nightly -- one deletion-directive record must not kill the
# whole night's merge. This is the LIVE trickle path (nothing invokes
# `codebook.py apply` outside its own tests), and it is worse than the CLI:
# run_nightly has no enclosing try/except and main() calls it bare, so a
# NeverDeleteError surfaces as an uncaught traceback -- no NightlyResult, no
# escalation, dump/commit never reached.
# ---------------------------------------------------------------------------

def _fake_invoke_deletion_directive(prompt: str, model: str) -> str:
    """Reachable in production: coder.py rebuilds the record from a fixed key
    allowlist (stripping a top-level delete/remove/retract/drop key) but
    copies `matches` VERBATIM, and _MATCH_SCHEMA only requires `entry_id` --
    so a match-level `action: delete` passes validate_coding_record and lands
    in run.records."""
    return json.dumps({
        'matches': [{
            'entry_id': 'known-cause',
            'origin_phase': 'implement',
            'manifested_phase': 'implement',
            'action': 'delete',
        }],
        'candidates': [],
    })


def test_run_nightly_skips_deletion_directive_record_without_crashing(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    # Load-bearing: this RETURNS today only once the merge loop is guarded --
    # an unguarded NeverDeleteError escapes run_nightly entirely.
    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_deletion_directive,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code == 0
    assert result.applied == 0
    assert result.coder_status == 'ok'

    assert len(escalation_calls) == 1
    _url, envelope = escalation_calls[0]
    assert 'deletion directive' in envelope['params']['arguments']['summary'].lower()

    # Nothing applied -> the existing no-change-night path holds.
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log
    assert result.reason is not None
    assert 'deletion directive' in result.reason.lower()


def test_run_nightly_deletion_directive_does_not_cost_the_rest_of_the_batch(tmp_path):
    """The headline claim the single-session test above CANNOT make.

    That test seeds one session, so every record in the batch is the
    deletion-shaped one: turning the merge loop's `continue` into a `break`
    (or an early `return result`) leaves all of its assertions green. This
    one seeds TWO sessions and hands the deletion-shaped judgment to the
    FIRST digest coded -- so a `break` would discard the good record that
    follows it, and the committed-codebook assertions below go red."""
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_dir = projects_root / _encode_cwd(work_cwd)
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_dir / 'session-1.jsonl', cwd=work_cwd,
        timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )
    # A DIFFERENT first user turn: sampling.dedupe_shapes fingerprints on
    # (stratum, signal-shape, normalized first turn), so two byte-similar
    # transcripts would collapse into one and this would silently become the
    # single-record test again.
    _write_transcript(
        session_dir / 'session-2.jsonl', cwd=work_cwd,
        timestamp='2026-07-13T11:00:00Z', session_id='session-2',
        user_text='an entirely different request that also went sideways',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    # Order-independent: the FIRST digest coded gets the deletion directive,
    # whichever session the sampler ranked first, so the good record always
    # sits AFTER the skipped one in run.records.
    invocations = []

    def _fake_invoke_deletion_then_match(prompt: str, model: str) -> str:
        invocations.append(prompt)
        if len(invocations) == 1:
            return _fake_invoke_deletion_directive(prompt, model)
        return _fake_invoke_known_cause(prompt, model)

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_deletion_then_match,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert len(invocations) == 2, 'both sessions must reach the coder'
    assert result.exit_code == 0
    assert result.coder_status == 'ok'
    # The GOOD record still landed: skipped-and-counted, never batch-fatal.
    assert result.applied == 1
    assert result.commit_made is True

    committed = codebook.load(codebook_path)
    entry = next(e for e in committed['entries'] if e['id'] == 'known-cause')
    assert len(entry['sightings']) == 1
    sighted = entry['sightings'][0]['session']
    assert sighted in {'session-1', 'session-2'}

    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert len(after_log) == len(before_log) + 1

    # ...and the BAD one was still surfaced, naming the OTHER session.
    assert len(escalation_calls) == 1
    _url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert 'deletion directive' in arguments['summary'].lower()
    assert f'session={sighted}' not in arguments['detail']
    assert result.reason is not None
    assert 'deletion directive' in result.reason.lower()


def test_run_nightly_aggregates_deletion_directive_escalations(tmp_path):
    """A SYSTEMIC cause -- a coder prompt regression, a model that starts
    emitting `action: delete` for every digest -- makes every record in the
    night deletion-shaped. That must post ONE escalation naming every
    skipped session, the shape the extractor-crash and coder-storm triggers
    already use, not one POST per record."""
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_dir = projects_root / _encode_cwd(work_cwd)
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_dir / 'session-1.jsonl', cwd=work_cwd,
        timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )
    _write_transcript(
        session_dir / 'session-2.jsonl', cwd=work_cwd,
        timestamp='2026-07-13T11:00:00Z', session_id='session-2',
        user_text='an entirely different request that also went sideways',
    )

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_deletion_directive,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code == 0
    assert result.applied == 0
    assert result.commit_made is False
    assert result.escalated is True

    assert len(escalation_calls) == 1, 'one aggregated POST, never one per record'
    _url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert '2 coding record(s)' in arguments['summary']
    assert 'deletion directive' in arguments['summary'].lower()
    # Both skipped sessions are named, so the aggregate loses no detail.
    assert 'session=session-1' in arguments['detail']
    assert 'session=session-2' in arguments['detail']
    assert result.reason == arguments['summary']


def test_run_nightly_records_a_reason_when_the_escalation_post_fails(tmp_path):
    """With the escalation server down, `post_escalation` returns False --
    so `escalated` is False and `exit_code` is 0, leaving the returned
    result structurally identical to a clean night. `reason` is what keeps
    a never-delete contract violation from surviving as nothing but a
    WARNING line in the journal."""
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    def _dead_poster(url, envelope):
        raise OSError('connection refused')

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
        invoke=_fake_invoke_deletion_directive,
        status_fetcher=None,
        poster=_dead_poster,
    )

    assert result.exit_code == 0
    assert result.escalated is False  # the POST genuinely failed
    assert result.reason is not None  # ...but the night is still not unremarkable
    assert '1 coding record(s)' in result.reason
    assert 'deletion directive' in result.reason.lower()


# ---------------------------------------------------------------------------
# step-9/10: a conflict-only night must PERSIST its recurrence sighting.
# apply_coding_record's adjudicated-title branch appends the sighting but
# bumps neither `matched` nor `candidates_applied` -- so a night whose ONLY
# merge effect is a conflict append would end with applied == 0, skip the
# `if applied > 0` dump gate, and drop the mutated codebook on the floor.
# ---------------------------------------------------------------------------

def _fake_invoke_rejected_candidate(prompt: str, model: str) -> str:
    """Mine a candidate whose title is ALREADY adjudicated in the codebook.

    `matches` is deliberately empty: a single match would bump
    stats['matched'], push `applied` above 0 for an unrelated reason and mask
    the defect entirely. coder.py copies `candidates` verbatim from the parsed
    judgment and _CANDIDATE_RECORD_SCHEMA requires only `title`, so this
    record passes validate_coding_record and reaches the merger on the live
    path."""
    return json.dumps({
        'matches': [],
        'candidates': [{
            'title': 'recurring rejected cause',
            'cause': 'c',
            'area': 'a',
            'origin_phase': 'implement',
            'manifested_phase': 'implement',
        }],
    })


def test_run_nightly_persists_a_disposition_conflict_sighting(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    # Seed an already-REJECTED candidate and commit it, so the tree is clean
    # before the run (otherwise _git_status_changed would report true for an
    # unrelated reason and the commit assertion would pass vacuously). The
    # seeded sighting carries session 'session-old' -- NOT 'session-1' -- so
    # apply_coding_record's `already_seen` guard does not short-circuit the
    # conflict branch, and the later "sightings grew 1 -> 2" assertion
    # unambiguously proves an APPEND rather than a create.
    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    cb = codebook.load(codebook_path)
    cb['candidates'].append({
        'id': 'cand-20260722-28',
        'title': 'recurring rejected cause',
        'first_seen': '2026-07-22',
        'disposition': 'rejected',
        'sightings': [{
            'date': '2026-07-22',
            'project': 'testproj',
            'session': 'session-old',
            'origin_phase': 'implement',
            'manifested_phase': 'implement',
        }],
    })
    codebook.dump(cb, codebook_path)
    subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
    subprocess.run(
        ['git', 'commit', '-q', '-m', 'seed rejected candidate'], cwd=repo, check=True,
    )

    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_rejected_candidate,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code == 0
    assert result.commit_made is True
    assert result.applied == 1
    assert result.coder_status == 'ok'

    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert len(after_log) == len(before_log) + 1

    # The recurrence sighting survived to the COMMITTED file.
    committed = codebook.load(codebook_path)
    assert len(committed['candidates']) == 1
    candidate = committed['candidates'][0]
    assert candidate['id'] == 'cand-20260722-28'
    assert candidate['disposition'] == 'rejected'
    assert len(candidate['sightings']) == 2
    assert candidate['sightings'][1]['session'] == 'session-1'
    assert candidate['sightings'][1]['date'] == '2026-07-13'

    # Idempotency: the widened dump gate must not re-commit the same sighting.
    result_2 = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_rejected_candidate,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result_2.exit_code == 0
    assert result_2.commit_made is False
    assert result_2.applied == 0

    after_log_2 = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log_2 == after_log

    committed_2 = codebook.load(codebook_path)
    assert len(committed_2['candidates'][0]['sightings']) == 2


# ---------------------------------------------------------------------------
# step-21/22: run_nightly -- no-change night commits nothing (§6.7)
# ---------------------------------------------------------------------------

def _fake_invoke_empty(prompt: str, model: str) -> str:
    return json.dumps({'matches': [], 'candidates': []})


def test_run_nightly_no_change_night_commits_nothing(tmp_path, caplog):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    # A valid-but-empty judgment ("coded fine, found nothing") -- a genuine
    # success, never conflated with a coding failure.
    with caplog.at_level('INFO', logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=target_date,
            now=fixed_now,
            invoke=_fake_invoke_empty,
            status_fetcher=None,
            poster=lambda url, envelope: escalation_calls.append((url, envelope)),
        )

    assert result.exit_code == 0
    assert result.commit_made is False
    assert result.applied == 0
    assert result.coder_status == 'ok'
    assert escalation_calls == []

    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log

    assert result.census_line is not None and 'NO-FIRE' in result.census_line
    assert any('no-change night' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# step-17/18: run_nightly -- fail-loud on extractor crash (decision 8)
# ---------------------------------------------------------------------------

def _crashing_build_digests(selected, **kwargs):
    """A fake build_digests: one digest builds fine, one session crashes --
    extractor_failures is non-empty even though a usable digest also
    exists, so the RED test can prove the coder is never reached."""
    ok_digest = (
        '---\n'
        'session: "ok-session"\n'
        'date: "2026-07-13"\n'
        'agent_class: "interactive"\n'
        '---\n'
        'body text\n'
    )
    return [ok_digest], [('session-1.jsonl', 'boom: corrupt transcript')]


def test_run_nightly_fail_loud_on_extractor_crash(tmp_path, monkeypatch):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    monkeypatch.setattr(nightly, 'build_digests', _crashing_build_digests)

    coder_calls = []

    def _unexpected_invoke(prompt, model):
        coder_calls.append(prompt)
        return 'not valid json'

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC)
    escalation_calls = []

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_unexpected_invoke,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code != 0
    assert result.escalated is True
    assert coder_calls == []  # the coder must never be reached after a crash

    assert len(escalation_calls) == 1
    url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert 'extractor' in arguments['summary'].lower()
    assert '1' in arguments['summary']

    # Merge/dump/commit skipped entirely: codebook untouched, no new commit.
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log


# ---------------------------------------------------------------------------
# step-23/24: main(argv) -- CLI: `run` + `resolve-config` subcommands
# ---------------------------------------------------------------------------

def test_main_resolve_config_prints_path_and_returns_zero(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / 'proj-a' / 'docs' / 'legibility' / 'legibility.yaml'

    def _fake_resolve(project_id, search_roots=None):
        assert project_id == 'proj_a'
        return config_path

    monkeypatch.setattr(nightly, 'resolve_config_path', _fake_resolve)

    exit_code = nightly.main(['resolve-config', 'proj_a'])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(config_path)


def test_main_resolve_config_unknown_id_returns_nonzero(monkeypatch, capsys):
    def _fake_resolve(project_id, search_roots=None):
        raise FileNotFoundError(f'no legibility.yaml found for project_id={project_id!r}')

    monkeypatch.setattr(nightly, 'resolve_config_path', _fake_resolve)

    exit_code = nightly.main(['resolve-config', 'proj_unknown'])

    assert exit_code != 0


def test_main_run_with_config_invokes_run_nightly_and_returns_exit_code(monkeypatch, tmp_path):
    config_path = tmp_path / 'legibility.yaml'
    projects_root = tmp_path / 'projects'
    calls = []

    def _fake_run_nightly(**kwargs):
        calls.append(kwargs)
        return nightly.NightlyResult(exit_code=0)

    monkeypatch.setattr(nightly, 'run_nightly', _fake_run_nightly)

    exit_code = nightly.main([
        'run', '--config', str(config_path),
        '--projects-root', str(projects_root),
        '--date', '2026-07-13',
    ])

    assert exit_code == 0
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs['config_path'] == str(config_path)
    assert kwargs['project_id'] is None
    assert kwargs['projects_root'] == str(projects_root)
    assert kwargs['target_date'] == date(2026, 7, 13)


def test_main_run_with_project_id_resolves_then_runs(monkeypatch):
    calls = []

    def _fake_run_nightly(**kwargs):
        calls.append(kwargs)
        return nightly.NightlyResult(exit_code=0)

    monkeypatch.setattr(nightly, 'run_nightly', _fake_run_nightly)

    exit_code = nightly.main(['run', '--project-id', 'proj_a'])

    assert exit_code == 0
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs['config_path'] is None
    assert kwargs['project_id'] == 'proj_a'


def test_main_run_propagates_fail_loud_exit_code(monkeypatch):
    def _fake_run_nightly(**kwargs):
        return nightly.NightlyResult(exit_code=1, escalated=True, reason='boom')

    monkeypatch.setattr(nightly, 'run_nightly', _fake_run_nightly)

    exit_code = nightly.main(['run', '--project-id', 'proj_a'])

    assert exit_code == 1


# ---------------------------------------------------------------------------
# step-11: main() must configure logging -- gap (b) of the trickle-legibility
# incident. Every INFO line in this module (the always-on sampler summary from
# step-8, plus the pre-existing no-change-night and empty-codebook lines) is
# discarded before it reaches the journal unless SOMETHING calls
# logging.basicConfig: nothing under scripts/legibility/ did, so root stayed at
# its WARNING default under the systemd ExecStart and all 14 silent nights of
# 2026-07-16..29 wrote nothing an operator could read.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _isolated_root_logging():
    """Yield the ROOT logger with its handlers emptied, restoring it after.

    ``configure_logging`` deliberately goes through ``logging.basicConfig``,
    which is a NO-OP when root already has handlers -- that no-op is exactly
    what keeps it safe to call under pytest and from a library importer. So
    its effect is only observable with root cleared first, which is what this
    helper does.

    Restoring both the level and the exact handler list matters: this file and
    test_legibility_census.py both run many ``caplog.at_level(...)`` blocks,
    and leaked root state would silently perturb them.

    DUPLICATED verbatim in test_legibility_census.py, which is a known wart,
    not a norm: its proper home is scripts/tests/conftest.py (today only
    sys.path plumbing, no fixtures yet), which task 3270 holds no lock on --
    so the move is deferred to the next review cycle rather than smuggled in.
    Until then, a fix here (e.g. also restoring ``logging.root.filters``, or
    closing handlers) has to be applied in BOTH copies.
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


def _main_run_with_stubbed_run_nightly(monkeypatch, tmp_path):
    """Drive ``nightly.main(['run', ...])`` over a stub, returning
    ``(exit_code, effective_root_level, root_handler_count)`` sampled while
    still inside the isolated-root block."""
    def _fake_run_nightly(**kwargs):
        return nightly.NightlyResult(exit_code=0)

    monkeypatch.setattr(nightly, 'run_nightly', _fake_run_nightly)

    with _isolated_root_logging() as root:
        exit_code = nightly.main(['run', '--config', str(tmp_path / 'legibility.yaml')])
        # Sample inside the block; assert outside, so a failing assertion is
        # reported by pytest with root logging already restored.
        return exit_code, root.getEffectiveLevel(), len(root.handlers)


def test_main_configures_logging_so_info_lines_reach_the_journal(monkeypatch, tmp_path):
    # The DEFAULT-level case, so the ambient env has to be cleared: this same
    # change teaches the CLIs to honour LEGIBILITY_LOG_LEVEL, and a developer
    # debugging the trickle (or a host whose unit env is sourced) exporting
    # LEGIBILITY_LOG_LEVEL=WARNING would otherwise turn a working fix red.
    # Deliberately NOT hoisted into _main_run_with_stubbed_run_nightly: the
    # sibling tests monkeypatch.setenv BEFORE calling it, so a delenv in there
    # would clobber exactly the value they are asserting on.
    monkeypatch.delenv('LEGIBILITY_LOG_LEVEL', raising=False)

    exit_code, effective_level, handler_count = _main_run_with_stubbed_run_nightly(
        monkeypatch, tmp_path,
    )

    assert exit_code == 0
    assert effective_level <= logging.INFO, (
        'main() must lower root to INFO -- otherwise the step-8 sampler summary, '
        'the no-change-night line and the empty-codebook line are all dropped '
        'before they reach the journal'
    )
    assert handler_count >= 1, 'root needs a handler, or INFO records go nowhere'


def test_main_honours_the_legibility_log_level_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv('LEGIBILITY_LOG_LEVEL', 'WARNING')

    exit_code, effective_level, handler_count = _main_run_with_stubbed_run_nightly(
        monkeypatch, tmp_path,
    )

    assert exit_code == 0
    assert effective_level == logging.WARNING, (
        'an operator turning the trickle down via LEGIBILITY_LOG_LEVEL must be honoured'
    )
    assert handler_count >= 1


def test_main_unparseable_log_level_degrades_to_info_without_raising(monkeypatch, tmp_path):
    monkeypatch.setenv('LEGIBILITY_LOG_LEVEL', 'chatty')

    # No pytest.raises wrapper on purpose: a bad env var must never take the
    # nightly timer down. If configure_logging raises, this call propagates it
    # and the test fails loudly, which is the point.
    exit_code, effective_level, handler_count = _main_run_with_stubbed_run_nightly(
        monkeypatch, tmp_path,
    )

    assert exit_code == 0, 'an unparseable LEGIBILITY_LOG_LEVEL must not fail the run'
    assert effective_level <= logging.INFO, 'an unparseable level degrades to the INFO default'
    assert handler_count >= 1


# ---------------------------------------------------------------------------
# configure_logging's never-raise contract, driven directly rather than
# through main(): main() always passes the stock default_level, so the
# caller-side half of the contract is unreachable from there.
# ---------------------------------------------------------------------------

def _configure_logging_and_sample(**kwargs):
    """Call ``config.configure_logging(**kwargs)`` against a cleared root,
    returning ``(effective_level, handler_count)`` sampled inside the block."""
    with _isolated_root_logging() as root:
        config_mod.configure_logging(**kwargs)
        return root.getEffectiveLevel(), len(root.handlers)


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('10', logging.DEBUG),
        ('20', logging.INFO),
        ('30', logging.WARNING),
        (' 40 ', logging.ERROR),
    ],
)
def test_configure_logging_honours_a_numeric_log_level(monkeypatch, raw, expected):
    """A numeric LEGIBILITY_LOG_LEVEL is the OTHER spelling operators reach
    for -- it is what logging's own API takes -- and it used to be rejected
    as a typo, because getLevelName('10') returns the string 'Level 10'."""
    monkeypatch.setenv('LEGIBILITY_LOG_LEVEL', raw)

    effective_level, handler_count = _configure_logging_and_sample()

    assert effective_level == expected
    assert handler_count >= 1


def test_configure_logging_never_raises_on_an_unknown_default_level(monkeypatch):
    """The never-raise guarantee covers the CALLER's input too.

    ``logging.getLevelName('CHATTY')`` returns the string 'Level CHATTY',
    and ``basicConfig(level='Level CHATTY')`` raises ValueError -- out of
    the helper whose stated job is to never take the timer down. A future
    caller passing a typo'd default must degrade, not crash.
    """
    monkeypatch.delenv('LEGIBILITY_LOG_LEVEL', raising=False)

    # No pytest.raises: a raise here propagates and fails the test loudly.
    effective_level, handler_count = _configure_logging_and_sample(default_level='CHATTY')

    assert effective_level == logging.INFO, 'an unknown default_level degrades to INFO'
    assert handler_count >= 1


def test_configure_logging_env_var_still_wins_over_an_unknown_default_level(monkeypatch):
    """Degrading the default must not swallow a VALID operator override."""
    monkeypatch.setenv('LEGIBILITY_LOG_LEVEL', 'DEBUG')

    effective_level, _ = _configure_logging_and_sample(default_level='CHATTY')

    assert effective_level == logging.DEBUG


# ---------------------------------------------------------------------------
# task 3340 step-9/10: run_nightly records trickle run state on EVERY path
# ---------------------------------------------------------------------------

def _recorder_spy():
    """A ``recorder=`` seam spy that DELEGATES to the real
    ``trickle_state.record_run``, so the assertions below run against real
    classifier + streak output rather than a mock that could drift from it.
    Returns (fn, calls) where each call is (kwargs, returned_doc)."""
    calls = []

    def _record(project_id, **kwargs):
        doc = trickle_state.record_run(project_id, **kwargs)
        calls.append((dict(kwargs, project_id=project_id), doc))
        return doc

    return _record, calls


def _one_recorded(calls):
    assert len(calls) == 1, (
        f'expected exactly ONE recorder call, got {len(calls)}: '
        f'{[c[0] for c in calls]}'
    )
    return calls[0][1]


class TestRunNightlyRecordsTrickleState:
    """``run_nightly`` records WHY the night went the way it did — on every
    exit path, including the fail-loud ones and an unexpected raise.

    One recording point (a single try/finally) cannot forget a branch.
    Recording only on the happy path would let the progress probe read a
    stale streak as healthy after repeated crashes.
    """

    def _run(self, tmp_path, *, recorder=None, budget_bytes=None,
             invoke=_fake_invoke_known_cause, committer=None, poster=None,
             transcript=True):
        work_cwd = str(tmp_path / 'work')
        repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
        if budget_bytes is not None:
            _write_config(
                repo, project_id='testproj', escalation_port=8199,
                cwd_prefixes=[work_cwd], max_daily_digest_bytes=budget_bytes,
            )
        projects_root = tmp_path / 'projects'
        if transcript:
            _write_transcript(
                projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
                cwd=work_cwd, timestamp='2026-07-13T10:00:00Z',
                session_id='session-1',
            )
        else:
            projects_root.mkdir(parents=True, exist_ok=True)

        # Annotated: a heterogeneous dict (Paths, a date, a datetime, injected
        # callables, None) whose inferred value union would otherwise be
        # re-reported once per union member at the run_nightly(**kwargs) call.
        kwargs: dict[str, Any] = dict(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=invoke,
            status_fetcher=None,
            poster=poster if poster is not None else (lambda url, envelope: None),
        )
        if recorder is not None:
            kwargs['recorder'] = recorder
        if committer is not None:
            kwargs['committer'] = committer
        return nightly.run_nightly(**kwargs), repo

    def test_happy_path_records_a_productive_run(self, tmp_path):
        recorder, calls = _recorder_spy()
        result, _repo = self._run(tmp_path, recorder=recorder)

        assert result.exit_code == 0
        assert result.commit_made is True

        doc = _one_recorded(calls)
        assert doc['outcome'] == 'productive'
        assert doc['consecutive_barren_runs'] == 0
        assert doc['commit_made'] is True
        assert doc['exit_code'] == 0
        assert doc['applied'] == result.applied
        assert doc['target_date'] == '2026-07-13'
        assert doc['counters']['selected_count'] == 1
        assert doc['counters']['budget_skipped'] == 0

    def test_budget_suppressed_night_records_barren(self, tmp_path):
        """The 2026-07-16..29 incident replay, now RECORDED as barren
        instead of being indistinguishable from a quiet night."""
        recorder, calls = _recorder_spy()
        result, _repo = self._run(tmp_path, recorder=recorder, budget_bytes=10)

        assert result.exit_code == 0
        assert result.budget_suppressed is True

        doc = _one_recorded(calls)
        assert doc['outcome'] == 'barren'
        assert doc['budget_suppressed'] is True
        assert doc['exit_code'] == 0
        assert doc['consecutive_barren_runs'] == 1
        assert doc['counters']['budget_skipped'] > 0
        assert doc['counters']['selected_count'] == 0

    def test_quiet_night_records_quiet_with_streak_zero(self, tmp_path):
        recorder, calls = _recorder_spy()
        result, _repo = self._run(tmp_path, recorder=recorder, transcript=False)

        assert result.exit_code == 0
        assert result.budget_suppressed is False

        doc = _one_recorded(calls)
        assert doc['outcome'] == 'quiet'
        assert doc['consecutive_barren_runs'] == 0
        assert doc['exit_code'] == 0

    def test_extractor_crash_still_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nightly, 'build_digests', _crashing_build_digests)
        recorder, calls = _recorder_spy()
        result, _repo = self._run(tmp_path, recorder=recorder)

        assert result.exit_code == 1
        doc = _one_recorded(calls)
        assert doc['exit_code'] == 1
        assert doc['counters']['selected_count'] == 1

    def test_coder_storm_still_records(self, tmp_path):
        recorder, calls = _recorder_spy()
        result, _repo = self._run(
            tmp_path, recorder=recorder, invoke=_fake_invoke_unparseable,
        )

        assert result.exit_code == 1
        assert result.coder_status == 'failure'
        doc = _one_recorded(calls)
        assert doc['exit_code'] == 1
        assert doc['counters']['selected_count'] == 1

    def test_codebook_validation_failure_still_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            codebook, 'validate', lambda cb: ['synthetic validation error'],
        )
        recorder, calls = _recorder_spy()
        result, _repo = self._run(tmp_path, recorder=recorder)

        assert result.exit_code == 1
        doc = _one_recorded(calls)
        assert doc['exit_code'] == 1
        assert doc['commit_made'] is False

    def test_commit_failure_still_records(self, tmp_path):
        recorder, calls = _recorder_spy()
        result, _repo = self._run(
            tmp_path, recorder=recorder, committer=_failing_committer,
        )

        assert result.exit_code == 1
        doc = _one_recorded(calls)
        assert doc['exit_code'] == 1
        assert doc['commit_made'] is False

    def test_unexpected_exception_mid_run_still_records(self, tmp_path, monkeypatch):
        """A crashed night that recorded NOTHING would let the progress
        probe read a stale streak as healthy. The exception must still
        propagate — recording is not swallowing."""
        def _boom(selected, **kwargs):
            raise RuntimeError('synthetic mid-run explosion')

        monkeypatch.setattr(nightly, 'build_digests', _boom)
        recorder, calls = _recorder_spy()

        with pytest.raises(RuntimeError, match='synthetic mid-run explosion'):
            self._run(tmp_path, recorder=recorder)

        doc = _one_recorded(calls)
        assert doc['exit_code'] != 0, (
            'a crashed night must record its crash honestly'
        )
        assert doc['counters']['selected_count'] == 1, (
            'the sample was computed before the crash, so its real counters '
            'are still recordable'
        )

    def test_a_raising_recorder_never_breaks_the_run(self, tmp_path, caplog):
        """Observability must never become a new failure mode — mirroring
        post_escalation's established best-effort contract."""
        def _raising_recorder(project_id, **kwargs):
            raise OSError('synthetic state-write failure')

        with caplog.at_level('WARNING', logger='legibility.nightly'):
            result, _repo = self._run(tmp_path, recorder=_raising_recorder)

        assert result.exit_code == 0
        assert result.commit_made is True
        assert any(
            r.levelname == 'WARNING' for r in caplog.records
        ), 'a swallowed state-write failure must still be announced once'

    def test_wired_for_real_end_to_end(self, tmp_path):
        """No recorder= override: the two modules are wired together for
        real, not just against a spy."""
        result, _repo = self._run(tmp_path)
        assert result.exit_code == 0

        status, doc = trickle_state.load_state(
            trickle_state.trickle_state_path('testproj')
        )
        assert status == 'ok', f'expected a real state file; got {status!r}'
        assert doc is not None
        assert doc['project_id'] == 'testproj'
        assert doc['outcome'] == 'productive'
        assert doc['target_date'] == '2026-07-13'


# ---------------------------------------------------------------------------
# task 3340 step-11/12: edge-triggered barren-streak escalation
# ---------------------------------------------------------------------------

def _streak_escalations(calls):
    """Escalations posted by the STREAK trigger, distinguished from
    _report_sample_outcome's per-night suppression notice (a different
    predicate, window and remedy prompt, posted on the same envelope)."""
    return [
        (url, env) for url, env in calls
        if 'consecutive' in env['params']['arguments']['summary'].lower()
    ]


class _NightRunner:
    """Drive REAL consecutive run_nightly calls against one repo, so the
    streak is built by the actual pipeline + recorder rather than a
    hand-seeded state file."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.work_cwd = str(tmp_path / 'work')
        self.repo, self.config_path = _init_e2e_repo(
            tmp_path, work_cwd=self.work_cwd,
        )
        self.projects_root = tmp_path / 'projects'
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.escalations = []

    def night(self, day, kind):
        """Run one night. *kind* is 'barren' (real signal, squeezed budget),
        'productive' (real signal, stock budget) or 'quiet' (no sessions at
        all for this date)."""
        target = date(2026, 7, day)
        _write_config(
            self.repo, project_id='testproj', escalation_port=8199,
            cwd_prefixes=[self.work_cwd],
            max_daily_digest_bytes=10 if kind == 'barren' else None,
        )
        if kind in ('barren', 'productive'):
            _write_transcript(
                self.projects_root / _encode_cwd(self.work_cwd)
                / f'session-{day}.jsonl',
                cwd=self.work_cwd,
                timestamp=f'2026-07-{day:02d}T10:00:00Z',
                session_id=f'session-{day}',
            )
        return nightly.run_nightly(
            config_path=self.config_path,
            projects_root=self.projects_root,
            target_date=target,
            now=datetime(2026, 7, day + 1, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_known_cause,
            status_fetcher=None,
            poster=lambda url, env: self.escalations.append((url, env)),
        )


class TestBarrenStreakEscalation:
    """The LIVE CALLER, so the new probe is not "a probe nobody runs".

    nightly.py owns this because it is the only thing that already runs
    every night. EDGE-triggered (fires on the run where the streak first
    REACHES the threshold, re-arms only after a productive/quiet run
    resets it) — which sidesteps task 3270's alarm-fatigue debate rather
    than re-opening it, and avoids the one-shot latch's worse failure mode
    that 3270 explicitly rejected.
    """

    def test_threshold_default_is_three(self):
        """One barren night can be an ordinary bad day; three consecutive
        means PERSISTENT CONFIG STATE. The real 2026-07-16..29 incident
        would have fired on night 3 of 14 instead of never."""
        assert trickle_state.DEFAULT_MAX_BARREN_RUNS == 3

    def test_nights_one_and_two_do_not_escalate_the_streak(self, tmp_path):
        runner = _NightRunner(tmp_path)

        first = runner.night(13, 'barren')
        assert first.barren_escalated is False
        assert _streak_escalations(runner.escalations) == []

        second = runner.night(14, 'barren')
        assert second.barren_escalated is False
        assert _streak_escalations(runner.escalations) == []

    def test_night_three_escalates_exactly_once(self, tmp_path):
        runner = _NightRunner(tmp_path)
        runner.night(13, 'barren')
        runner.night(14, 'barren')
        third = runner.night(15, 'barren')

        assert third.barren_escalated is True

        streak = _streak_escalations(runner.escalations)
        assert len(streak) == 1, (
            f'expected exactly ONE streak escalation; got '
            f'{[e["params"]["arguments"]["summary"] for _u, e in streak]}'
        )
        arguments = streak[0][1]['params']['arguments']

        # The decision-8 envelope, reused unchanged, so it never gates
        # anything.
        assert streak[0][1]['params']['name'] == 'escalate_info'
        assert arguments['category'] == 'infra_issue'
        assert arguments['severity'] == 'info'

        assert '3' in arguments['summary'], 'summary must name the streak count'
        detail = arguments['detail']
        assert 'testproj' in detail
        assert 'budget_skipped' in detail
        assert 'max_daily_digest_bytes' in detail, (
            'the budget door has its OWN remedy and must be named'
        )

    def test_night_four_does_not_escalate_again(self, tmp_path):
        """EDGE-triggered, not level-triggered."""
        runner = _NightRunner(tmp_path)
        for day in (13, 14, 15):
            runner.night(day, 'barren')
        assert len(_streak_escalations(runner.escalations)) == 1

        fourth = runner.night(16, 'barren')

        assert fourth.barren_escalated is False
        assert len(_streak_escalations(runner.escalations)) == 1, (
            'a level-triggered alarm would re-fire every night from here on'
        )

    def test_a_productive_night_rearms_the_trigger(self, tmp_path):
        """A one-shot latch would reproduce exactly the pathology 3270
        warned about: whoever misses the single first escalation gets
        silence thereafter."""
        runner = _NightRunner(tmp_path)
        for day in (13, 14, 15):
            runner.night(day, 'barren')
        assert len(_streak_escalations(runner.escalations)) == 1

        recovered = runner.night(16, 'productive')
        assert recovered.barren_escalated is False

        for day in (17, 18):
            assert runner.night(day, 'barren').barren_escalated is False
        assert len(_streak_escalations(runner.escalations)) == 1

        refired = runner.night(19, 'barren')
        assert refired.barren_escalated is True
        assert len(_streak_escalations(runner.escalations)) == 2

    def test_a_quiet_night_also_rearms_the_trigger(self, tmp_path):
        runner = _NightRunner(tmp_path)
        for day in (13, 14, 15):
            runner.night(day, 'barren')
        runner.night(16, 'quiet')

        for day in (17, 18):
            assert runner.night(day, 'barren').barren_escalated is False
        assert runner.night(19, 'barren').barren_escalated is True

    def test_exit_code_stays_zero_on_the_streak_escalation_path(self, tmp_path):
        """A non-zero exit flips legibility-trickle@testproj.service to
        Result=failed, permanently false-alarming check_trickle_liveness.sh
        — the exact trade _report_sample_outcome already refused. It would
        trade a silent-failure mode for a permanent false alarm and burn
        the one signal that works."""
        runner = _NightRunner(tmp_path)
        runner.night(13, 'barren')
        runner.night(14, 'barren')
        third = runner.night(15, 'barren')

        assert third.barren_escalated is True
        assert third.exit_code == 0

    def test_a_quiet_streak_never_escalates(self, tmp_path):
        """PRD decision 7's no-false-alarm guarantee, asserted END-TO-END
        through the real pipeline rather than only at the classifier."""
        runner = _NightRunner(tmp_path)
        for day in (13, 14, 15, 16):
            result = runner.night(day, 'quiet')
            assert result.barren_escalated is False
            assert result.exit_code == 0

        assert runner.escalations == [], (
            f'a quiet project must post NOTHING however long it stays quiet; '
            f'got {runner.escalations!r}'
        )


# ---------------------------------------------------------------------------
# task 3340 step-11/12: the sibling barren door gets journal visibility
# ---------------------------------------------------------------------------

def _sample(*, selected=(), zero_signal_dropped=0, dedupe_collapsed=0,
            below_sampling_cut=0, budget_skipped=0):
    """Hand-build a SampleResult satisfying the conservation identity. The
    sampling-cut-only barren night is NOT reachable end-to-end (a stratum's
    cut is max(ceil(top_fraction*n), per_stratum_min) >= 1, so something is
    always selected unless the BUDGET also cuts), so this mode is exercised
    at the _report_sample_outcome boundary directly."""
    selected = list(selected)
    return nightly.sampling.SampleResult(
        selected=selected,
        per_stratum_counts={},
        zero_signal_dropped=zero_signal_dropped,
        bytes_used=0,
        dedupe_collapsed=dedupe_collapsed,
        budget_skipped=budget_skipped,
        below_sampling_cut=below_sampling_cut,
        total_records=(
            zero_signal_dropped + dedupe_collapsed + below_sampling_cut
            + budget_skipped + len(selected)
        ),
    )


def _cfg_for(tmp_path):
    return load_config(_write_config(tmp_path, project_id='testproj'))


def test_report_sample_outcome_warns_on_a_sampling_cut_only_night(tmp_path, caplog):
    """The sibling absence mode task 3270 left ENTIRELY unobserved: real,
    distinct signal held back by the sampling cut, nothing digested. It
    gets journal visibility now, and escalation only via the streak, so
    per-night alarm volume does not rise."""
    cfg = _cfg_for(tmp_path)
    posted = []

    with caplog.at_level('INFO', logger='legibility.nightly'):
        escalated = nightly._report_sample_outcome(
            cfg, _sample(below_sampling_cut=3), date(2026, 7, 13),
            poster=lambda url, env: posted.append((url, env)),
        )

    assert escalated is False
    assert posted == [], (
        'the streak owns escalation for this mode; a per-night escalation '
        'here would raise alarm volume for a signal that already repeats'
    )

    loud = _nightly_warnings(caplog)
    assert len(loud) == 1, (
        f'expected exactly one WARNING; got {[r.getMessage() for r in loud]}'
    )
    message = loud[0].getMessage()
    assert 'top_fraction' in message or 'per_stratum_min' in message, (
        f'the sampling-cut door has its OWN remedy and must name it; got '
        f'{message!r}'
    )
    # Raising the byte budget does nothing whatever for a
    # below_sampling_cut record (SampleResult says so explicitly), so the
    # message may only mention that knob in order to DISCLAIM it. Silence
    # would be weaker: an operator primed by task 3270's budget escalation
    # will reach for max_daily_digest_bytes by default, and only an
    # explicit "that will not help" stops them.
    if 'max_daily_digest_bytes' in message:
        assert 'not help' in message, (
            f'the byte-budget knob may only appear as an explicit '
            f'non-remedy; got {message!r}'
        )


def test_report_sample_outcome_budget_door_is_unchanged(tmp_path, caplog):
    """The budget door keeps its per-night escalation exactly as 3270 left
    it — this change adds a sibling branch, it does not alter this one."""
    cfg = _cfg_for(tmp_path)
    posted = []

    with caplog.at_level('INFO', logger='legibility.nightly'):
        escalated = nightly._report_sample_outcome(
            cfg, _sample(budget_skipped=4), date(2026, 7, 13),
            poster=lambda url, env: posted.append((url, env)),
        )

    assert escalated is True
    assert len(posted) == 1
    assert 'suppress' in posted[0][1]['params']['arguments']['summary'].lower()


def test_report_sample_outcome_stays_quiet_on_a_quiet_night(tmp_path, caplog):
    cfg = _cfg_for(tmp_path)
    posted = []

    with caplog.at_level('INFO', logger='legibility.nightly'):
        escalated = nightly._report_sample_outcome(
            cfg, _sample(zero_signal_dropped=9, dedupe_collapsed=2),
            date(2026, 7, 13),
            poster=lambda url, env: posted.append((url, env)),
        )

    assert escalated is False
    assert posted == []
    assert _nightly_warnings(caplog) == []


def test_report_sample_outcome_partial_truncation_stays_at_info(tmp_path, caplog):
    """Selected something AND skipped some on budget: the budget working as
    designed, never an absence."""
    cfg = _cfg_for(tmp_path)
    posted = []

    with caplog.at_level('INFO', logger='legibility.nightly'):
        escalated = nightly._report_sample_outcome(
            cfg, _sample(selected=['digest'], budget_skipped=2),
            date(2026, 7, 13),
            poster=lambda url, env: posted.append((url, env)),
        )

    assert escalated is False
    assert posted == []
    assert _nightly_warnings(caplog) == []


# ---------------------------------------------------------------------------
# task 3644: the trickle's fail-loud escalation must survive a STATEFUL server
# ---------------------------------------------------------------------------
#
# `_default_poster` bare-POSTed `tools/call`, which the escalation server
# (:8103) rejects at the transport layer before the tool ever runs, with
# `400 Bad Request` / "Missing session ID" (captured live 2026-08-05).
# `post_escalation` swallows that best-effort and returns False, so EVERY
# trickle fail-loud escalation -- extractor crash, coder storm, commit
# failure -- was silently dropped. Identical root cause and identical fix as
# census.py's poster; the transport lives single-sourced in census_trigger.


class _FakeStatefulResponse:
    """An `httpx.Response` stand-in for the stateful-server handshake."""

    def __init__(self, *, status_code=200, headers=None, payload=None, text=''):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            # A plain RuntimeError, not httpx.HTTPStatusError: the shared
            # install_fake_httpx stub exposes only `post` and pytest.fails on
            # any other attribute (task 3376).
            raise RuntimeError(f'HTTP {self.status_code}')

    def json(self):
        if self._payload is None:
            raise ValueError('response has no JSON body')
        return self._payload


def _stateful_escalation_post(recorded, *, session_id='sid-nightly'):
    """A fake `httpx.post` behaving like the STATEFUL escalation server:
    session-less `tools/call` -> 400; `initialize` -> 200 + the assigned id
    as a response header; `notifications/initialized` -> 202; `tools/call`
    WITH that header -> 200."""
    def _post(url, **kwargs):
        recorded.append((url, kwargs))
        envelope = kwargs.get('json') or {}
        method = envelope.get('method')
        if method == 'initialize':
            return _FakeStatefulResponse(
                headers={'mcp-session-id': session_id,
                         'content-type': 'application/json'},
                payload={'jsonrpc': '2.0', 'id': 1, 'result': {}},
            )
        if method == 'notifications/initialized':
            return _FakeStatefulResponse(status_code=202)
        if (kwargs.get('headers') or {}).get('mcp-session-id') != session_id:
            return _FakeStatefulResponse(
                status_code=400,
                headers={'content-type': 'application/json'},
                payload={'jsonrpc': '2.0', 'id': 'server-error',
                         'error': {'code': -32600,
                                   'message': 'Bad Request: Missing session ID'}},
            )
        return _FakeStatefulResponse(
            headers={'content-type': 'application/json'},
            payload={'jsonrpc': '2.0', 'id': 1,
                     'result': {'structuredContent': {'id': 'esc-9', 'status': 'queued'}}},
        )

    return _post


def _recording_delete(deleted):
    """A fake `httpx.delete` recording the MCP session-termination call the
    transport makes after a handshake, so the long-lived escalation server does
    not leak a session per trickle escalation."""
    def _delete(url, **kwargs):
        deleted.append((url, kwargs))
        return _FakeStatefulResponse(status_code=200)

    return _delete


def test_post_escalation_lands_against_a_stateful_server(tmp_path, install_fake_httpx):
    """The DEFAULT poster (no `poster=` injection) must handshake and land."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))
    recorded = []
    deleted = []
    install_fake_httpx(
        _stateful_escalation_post(recorded), delete=_recording_delete(deleted),
    )

    assert nightly.post_escalation(cfg, 'summary text', 'detail text') is True

    methods = [(kwargs.get('json') or {}).get('method') for _url, kwargs in recorded]
    assert methods == [
        'tools/call', 'initialize', 'notifications/initialized', 'tools/call',
    ]
    # The retried call carries the server-assigned session id...
    assert (recorded[-1][1].get('headers') or {}).get('mcp-session-id') == 'sid-nightly'
    # ...and is still the escalate_info the trickle meant to file.
    params = (recorded[-1][1].get('json') or {})['params']
    assert params['name'] == 'escalate_info'
    assert params['arguments']['summary'] == 'summary text'
    # ...and the session opened to file it is released again.
    assert [(kw.get('headers') or {}).get('mcp-session-id') for _u, kw in deleted] == [
        'sid-nightly'
    ]


def test_post_escalation_reports_false_on_a_tool_error_envelope(
    tmp_path, install_fake_httpx, caplog
):
    """HTTP 200 is not success. `post_escalation` discards the response body
    and reports True on "no exception", so a tool-level failure
    (`result.isError: true`) would otherwise read as a landed escalation --
    the same green-on-paper/nothing-filed failure task 3644 exists to close,
    one layer up from the transport."""
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def _post(url, **kwargs):
        return _FakeStatefulResponse(
            headers={'content-type': 'application/json'},
            payload={'jsonrpc': '2.0', 'id': 1, 'result': {
                'isError': True,
                'content': [{'type': 'text', 'text': "unknown category 'nope'"}],
            }},
        )

    install_fake_httpx(_post)

    with caplog.at_level(logging.WARNING):
        assert nightly.post_escalation(cfg, 'summary text', 'detail text') is False

    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('escalation post failed' in m for m in warned), warned
    assert any('isError' in m or 'reported an error' in m for m in warned), warned


# ---------------------------------------------------------------------------
# task 4148: the tasks-landed condition, end to end through the PRODUCTION
# entrypoint.
#
# Every other census test in this file enters at run_nightly or below, and
# run_nightly has ALWAYS accepted a status_fetcher -- so all of them passed
# throughout the dead period. The defect was specifically that the systemd
# ExecStart (`nightly.py run --project-id %i` -> main()) wired nothing, which
# only a test entering at main() can catch. The journal shows the resulting
# fail-safe line on 2026-08-10, 2026-08-11 and 2026-08-12:
# "tasks-landed: no status_fetcher configured -- condition (b) fails safe".
#
# HAZARD, now that the fetcher default is live (amendment): ANY run_nightly or
# main() test that writes a valid non-negative `last_census_done_count` into
# census-state.json and injects no `status_fetcher` will issue a REAL POST to
# $FUSED_MEMORY_MCP_URL (default localhost:8002) -- and on a dev box where
# fused-memory is actually up, a stale fixture baseline against the real
# done-count can produce a genuine FIRE that subprocess-launches
# scripts/legibility/census.py, i.e. real LLM spend and real git writes.
# Such a test MUST both fake httpx (conftest's `install_fake_httpx`) AND stub
# `nightly._default_census_launcher`. The two tests below are the pattern.
# ---------------------------------------------------------------------------

def test_main_run_fires_the_tasks_landed_condition_end_to_end(
    tmp_path, monkeypatch, caplog, install_fake_httpx,
):
    """Drive `nightly.main(['run', '--config', ...])` -- the systemd-invoked
    CLI -- against a project whose census baseline is 8 days and 130 done
    tasks stale, and assert condition (b) fires, that the absolute
    project_root crossed the MCP wire, and that the decision reaches the
    journal.

    8 days is chosen against the shipped config.py:66-77 defaults so ONLY
    condition (b) can fire: clear of (a) max_interval_days=10, at/above (b)'s
    tasks_landed_min_days=7, and above the floor_days=5 hard floor. main()
    exposes no `now` seam, so the anchor is relative to real now.
    """
    config_path = _write_config(tmp_path, project_id='proj_a')
    legibility_dir = tmp_path / 'docs' / 'legibility'

    # A genuine non-negative int baseline: task 4085's validity arm
    # (census_trigger.py:724-733) returns None BEFORE the fetcher is reached
    # for a bool/float/str, which would make this test pass for the wrong
    # reason -- or rather, fail for one.
    (legibility_dir / 'census-state.json').write_text(
        json.dumps({
            'last_census_at': (datetime.now(UTC) - timedelta(days=8)).isoformat(),
            'last_census_report': 'docs/legibility/census-2026-08-09.md',
            'last_census_done_count': 1000,
        }),
        encoding='utf-8',
    )
    # Present-but-empty, so condition (c) sees 0 candidates AND
    # decide_for_project logs no unreadable-codebook warning.
    codebook.dump(
        {'version': 2, 'entries': [], 'candidates': []},
        legibility_dir / 'confusion-codebook.yaml',
    )

    # 1130 done - 1000 baseline = 130 landed, over the 120 threshold.
    recorded = []

    def _fake_post(url, **kwargs):
        recorded.append((url, kwargs))
        return _FakeStatefulResponse(
            headers={'content-type': 'application/json'},
            payload={
                'jsonrpc': '2.0',
                'id': 1,
                'result': {
                    'structuredContent': {
                        'statuses': {f't{i}': 'done' for i in range(1130)},
                    },
                },
            },
        )

    install_fake_httpx(_fake_post)

    # MANDATORY, not cosmetic: on FIRE the real launcher subprocess-runs
    # scripts/legibility/census.py (real LLM spend + git writes), and
    # _default_entrypoint_exists is true in a real checkout -- and reaching
    # FIRE is the entire point of this test.
    launcher_calls = []
    monkeypatch.setattr(
        nightly, '_default_census_launcher', lambda: launcher_calls.append(1),
    )

    # Empty -> empty sample -> no digests -> `invoke` is never called and
    # nothing is committed, so no LLM and no git.
    projects_root = tmp_path / 'projects'
    projects_root.mkdir()

    # caplog rather than `_isolated_root_logging`: that helper empties
    # root.handlers, which is exactly where pytest's own capture handler
    # lives, so nothing would be captured. It is also unnecessary here --
    # `configure_logging` goes through `logging.basicConfig`, a documented
    # no-op while root has handlers, so main()'s call cannot leak root state
    # for as long as caplog's handler is installed.
    with caplog.at_level(logging.INFO, logger='legibility.nightly'):
        exit_code = nightly.main([
            'run', '--config', str(config_path),
            '--projects-root', str(projects_root),
            '--date', '2026-07-13',
        ])

    messages = [r.getMessage() for r in caplog.records]

    # (i) condition (b) fired all the way through the production entrypoint.
    assert exit_code == 0
    assert launcher_calls == [1], (
        'the census launcher never fired end-to-end. Read the captured log '
        'BEFORE suspecting the wiring: task 4085 turns any exception out of '
        '`decide` into a quiet synthetic NO-FIRE line rather than a '
        f'traceback. messages={messages}'
    )

    # (ii) task 3291's wire contract, now exercised from the nightly path:
    # fused-memory's _normalize_project_root hard-rejects a relative path.
    cfg = load_config(config_path)
    get_statuses_calls = [
        kwargs for _url, kwargs in recorded
        if (kwargs.get('json') or {}).get('params', {}).get('name') == 'get_statuses'
    ]
    assert len(get_statuses_calls) == 1, recorded
    sent_root = get_statuses_calls[0]['json']['params']['arguments']['project_root']
    assert sent_root == str(cfg.project_root)
    assert Path(sent_root).is_absolute()

    # (iii) the decision reaches the journal. Without this, a healthy
    # condition-(b) evaluation is indistinguishable from the still-dead one
    # in `journalctl --user -u legibility-trickle@dark_factory` -- the very
    # channel that diagnosed task 4148.
    assert any(
        'tasks-landed: 130 landed since last census (threshold 120) -> FIRE' in m
        for m in messages
    ), messages


def test_main_run_fails_safe_when_the_defaulted_fetcher_cannot_reach_fused_memory(
    tmp_path, monkeypatch, caplog, install_fake_httpx,
):
    """The twin of the test above, on the FAILURE path (task 4148 amendment).

    Now that `run_nightly` always BUILDS a real MCP-backed fetcher, "the
    fused-memory server is down" is reachable on every production night --
    not only when a caller injects a raising fake. The existing fail-safe
    coverage injects at the `compute_tasks_landed` level, which is exactly
    the shape of coverage that let the wiring gap survive three prior
    repairs; this test instead enters at `main()` and lets the DEFAULTED
    fetcher be the thing that fails. A transport error must degrade to a
    NO-FIRE line plus warnings -- never a non-zero exit, and never a census
    launch off an unobserved done-count.
    """
    config_path = _write_config(tmp_path, project_id='proj_a')
    legibility_dir = tmp_path / 'docs' / 'legibility'

    # The same 8-day, valid-int baseline as the happy-path twin: it clears
    # task 4085's validity arm and (b)'s tasks_landed_min_days=7, so the fetch
    # is genuinely REACHED and its failure is the only reason (b) goes N/A.
    # 8d is also under max_interval_days=10, so no OTHER condition fires and
    # the launcher assertion below cannot pass for the wrong reason.
    (legibility_dir / 'census-state.json').write_text(
        json.dumps({
            'last_census_at': (datetime.now(UTC) - timedelta(days=8)).isoformat(),
            'last_census_report': 'docs/legibility/census-2026-08-09.md',
            'last_census_done_count': 1000,
        }),
        encoding='utf-8',
    )
    codebook.dump(
        {'version': 2, 'entries': [], 'candidates': []},
        legibility_dir / 'confusion-codebook.yaml',
    )

    # fused-memory down. `post_mcp_envelope` touches only `httpx.post` and
    # `httpx.delete` (task 3376) and lets transport exceptions escape
    # verbatim, so this reaches `default_status_fetcher`'s wrapper and comes
    # out as StatusFetchUnavailable -- the real production shape.
    posts = []

    def _refusing_post(url, **kwargs):
        posts.append((url, kwargs))
        raise ConnectionError('[Errno 111] Connection refused')

    install_fake_httpx(_refusing_post)

    launcher_calls = []
    monkeypatch.setattr(
        nightly, '_default_census_launcher', lambda: launcher_calls.append(1),
    )

    projects_root = tmp_path / 'projects'
    projects_root.mkdir()

    with caplog.at_level(logging.INFO, logger='legibility.nightly'):
        exit_code = nightly.main([
            'run', '--config', str(config_path),
            '--projects-root', str(projects_root),
            '--date', '2026-07-13',
        ])

    messages = [r.getMessage() for r in caplog.records]

    # The nightly run is unaffected by a dead fused-memory...
    assert exit_code == 0
    assert launcher_calls == [], messages

    # ...but the fetch really was attempted through the DEFAULTED seam, so
    # this test cannot go green on a fetcher that was never built.
    get_statuses_calls = [
        kwargs for _url, kwargs in posts
        if (kwargs.get('json') or {}).get('params', {}).get('name') == 'get_statuses'
    ]
    assert len(get_statuses_calls) == 1, posts

    # The failure is legible on both channels an operator has: the specific
    # WARNING naming the fetch, and the one-line NO-FIRE decision.
    assert any('tasks-landed: status_fetcher failed' in m for m in messages), messages
    assert any(
        m.startswith('legibility trickle: census trigger: NO-FIRE')
        and 'tasks-landed: delta unavailable (no baseline/fetcher) -> N/A' in m
        for m in messages
    ), messages


# ---------------------------------------------------------------------------
# task 4511 step-1/2 (b): EVERY decision-8 fail-loud branch reaches the
# journal, with the escalation server DOWN.
#
# This is the incident shape, not a hypothetical: the reason is only ever
# durable if it is written before the POST that may never land. Driven end to
# end through the four real branches (extractor crash, coder storm, codebook
# validation failure, commit failure) so a NEW fail-loud branch added later
# inherits the guarantee for free -- it cannot reach `escalated=...` without
# passing through the one log site.
#
# Nothing here asserts on the escalation payload: `_build_escalation_arguments`
# and the envelope are deliberately untouched by this task.
# ---------------------------------------------------------------------------

def _run_decision8_branch(tmp_path, monkeypatch, branch, *, poster):
    """Drive `run_nightly` down ONE named decision-8 fail-loud branch, on the
    existing e2e repo/transcript fixtures.

    Modelled on `TestRunNightlyRecordsTrickleState._run` (which already
    demonstrates one helper reaching all four branches) and reusing that
    class's drivers verbatim, so this test pins logging and nothing else.
    """
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
        cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    kwargs: dict[str, Any] = dict(
        config_path=config_path,
        projects_root=projects_root,
        target_date=date(2026, 7, 13),
        now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=poster,
    )
    if branch == 'extractor':
        monkeypatch.setattr(nightly, 'build_digests', _crashing_build_digests)
    elif branch == 'storm':
        kwargs['invoke'] = _fake_invoke_unparseable
    elif branch == 'validation':
        monkeypatch.setattr(
            codebook, 'validate', lambda cb: ['synthetic validation error'],
        )
    elif branch == 'commit':
        kwargs['committer'] = _failing_committer
    else:  # pragma: no cover - guards a typo'd parametrize id
        raise AssertionError(f'unknown decision-8 branch {branch!r}')

    return nightly.run_nightly(**kwargs)


@pytest.mark.parametrize(
    ('branch', 'summary_marker', 'detail_marker'),
    [
        ('extractor', 'extractor crashed', 'boom: corrupt transcript'),
        ('storm', 'coder storm', 'could not parse a JSON object'),
        ('validation', 'failed validation', 'synthetic validation error'),
        ('commit', 'commit failed', 'cannot lock ref (simulated)'),
    ],
)
def test_every_fail_loud_branch_journals_its_reason_with_the_server_down(
    tmp_path, monkeypatch, caplog, branch, summary_marker, detail_marker,
):
    def _raising_poster(url, envelope):
        raise RuntimeError('escalation server unreachable')

    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        result = _run_decision8_branch(
            tmp_path, monkeypatch, branch, poster=_raising_poster,
        )

    assert result.exit_code == 1
    assert result.escalated is False, (
        'the POST failed, so nothing was filed -- which is precisely why the '
        'journal has to carry the reason'
    )

    errors = [r for r in _nightly_warnings(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 1, (
        f'expected exactly one ERROR for the {branch} branch; got '
        f'{[r.getMessage() for r in errors]}'
    )
    message = errors[0].getMessage()
    assert summary_marker in message, f'{summary_marker!r} not in {message!r}'
    assert detail_marker in message, (
        f'the reason must survive whole -- {detail_marker!r} not in {message!r}'
    )


# ---------------------------------------------------------------------------
# task 4511 step-3/4: the three EXIT-0 escalation sites stay at WARNING, and
# are logged exactly ONCE.
#
# `post_escalation` now journals every escalation itself, so the two call
# sites that already logged the pair would double-log, and all three exit-0
# sites would be promoted to ERROR. Neither is acceptable: the budget door,
# the barren streak and the deletion-directive aggregate all deliberately
# leave `exit_code=0` (a non-zero exit would make check_trickle_liveness.sh
# scream every night about a timer that is running perfectly), so an ERROR
# here would put a healthy-but-barren timer into `journalctl -p err`.
#
# The rule these three pin: ERROR iff the escalation accompanies a NON-ZERO
# exit; WARNING iff the run still exits 0.
# ---------------------------------------------------------------------------

def test_budget_suppression_door_journals_once_at_warning(tmp_path, caplog):
    """The budget-suppression door escalates AND exits 0, so its journal
    line stays a WARNING -- and there is exactly one of it, not a call-site
    WARNING plus a post_escalation ERROR."""
    cfg = _cfg_for(tmp_path)
    posted = []

    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        escalated = nightly._report_sample_outcome(
            cfg, _sample(budget_skipped=4), date(2026, 7, 13),
            poster=lambda url, env: posted.append((url, env)),
        )

    assert escalated is True
    assert len(posted) == 1

    loud = _nightly_warnings(caplog)
    assert len(loud) == 1, (
        f'expected exactly one record for the summary/detail pair; got '
        f'{[(r.levelname, r.getMessage()) for r in loud]}'
    )
    assert loud[0].levelno == logging.WARNING, (
        'this door leaves exit_code=0 on purpose; promoting its journal '
        'line to ERROR is a weaker version of the same false alarm a '
        'non-zero exit would raise'
    )
    message = loud[0].getMessage()
    assert 'totally suppressed by the digest byte budget' in message
    assert 'max_daily_digest_bytes' in message, (
        f'the detail half names the remedy; got {message!r}'
    )


def test_barren_streak_journals_once_at_warning(tmp_path, caplog):
    """Same rule for the sibling streak escalation: it explicitly refuses to
    touch `result.exit_code`, so it stays out of `journalctl -p err`."""
    cfg = _cfg_for(tmp_path)
    result = nightly.NightlyResult(exit_code=0)
    posted = []
    doc = {
        'outcome': trickle_state.OUTCOME_BARREN,
        'consecutive_barren_runs': trickle_state.DEFAULT_MAX_BARREN_RUNS,
        'counters': {'budget_skipped': 2},
        'last_productive_at': '2026-07-12T03:00:00+00:00',
    }

    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        nightly._escalate_barren_streak(
            cfg, doc, date(2026, 7, 15), result,
            poster=lambda url, env: posted.append((url, env)),
        )

    assert result.barren_escalated is True
    assert len(posted) == 1

    loud = _nightly_warnings(caplog)
    assert [r.levelname for r in loud] == ['WARNING'], (
        f'expected exactly one WARNING; got '
        f'{[(r.levelname, r.getMessage()) for r in loud]}'
    )
    message = loud[0].getMessage()
    assert (
        f'produced nothing for {trickle_state.DEFAULT_MAX_BARREN_RUNS} '
        f'consecutive runs'
    ) in message
    assert 'check_trickle_progress.py' in message, (
        f'the detail half names the on-demand probe; got {message!r}'
    )


def test_deletion_directive_aggregate_journals_once_at_warning(tmp_path, caplog):
    """The third exit-0 escalation, which the task description does not name
    but which shares the rule: one deletion-shaped coder record is loud but
    NON-fatal (exit_code stays 0, the same shape census.py uses for its
    mass-rejection signal), so its aggregate belongs at WARNING."""
    work_cwd = str(tmp_path / 'work')
    _repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
        cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    escalations = []
    with caplog.at_level(logging.DEBUG, logger='legibility.nightly'):
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            invoke=_fake_invoke_deletion_directive,
            status_fetcher=None,
            poster=lambda url, env: escalations.append((url, env)),
        )

    assert result.exit_code == 0
    assert len(escalations) == 1

    loud = _nightly_warnings(caplog)
    # 'carried' (past tense) is the AGGREGATE; the per-record line at the
    # skip site says 'carries' and is deliberately left alone.
    aggregate = [r for r in loud if 'carried a deletion directive' in r.getMessage()]
    assert len(aggregate) == 1, (
        f'expected exactly one aggregate record; got '
        f'{[(r.levelname, r.getMessage()) for r in aggregate]}'
    )
    assert aggregate[0].levelno == logging.WARNING
    assert [r for r in loud if r.levelno >= logging.ERROR] == [], (
        'a run that exits 0 must never appear in `journalctl -p err`'
    )


# ---------------------------------------------------------------------------
# task 4511 step-5/6: THE 2026-08-18 INCIDENT REPLAY, end to end.
#
# On 2026-08-18 the trickle's systemd manager had a PATH without
# ~/.local/bin, every selected digest ENOENT'd on `claude`, and the operator
# looking at `journalctl --user -u legibility-trickle@reify.service` saw
# nothing but an unexplained benign 400 and `status=1/FAILURE` -- the only
# copy of the reason lived inside the archived escalation JSON.
#
# This is that failure turned into an automated test, and it is the ONLY test
# in this module that runs `run_nightly` with no `invoke=` override, so it is
# the only one exercising the production wiring
# `run_nightly -> coder.code_digests -> coder._invoke_cli` end to end. It
# stays hermetic and free: a nonexistent binary makes `subprocess.run` raise
# OSError BEFORE any process starts, so there is no LLM call, no network and
# no timing.
# ---------------------------------------------------------------------------

def _scrub_path_of_claude(tmp_path, monkeypatch):
    """Point PATH somewhere the REAL `claude` is NOT resolvable, and prove it.

    MANDATORY SAFETY for the test below, not tidiness -- the discipline task
    4510 established in test_legibility_coder.py, adopted here because this
    module now reaches the same seam. `_invoke_cli` resolves
    `claude_bin or os.environ.get(_CLAUDE_BIN_ENV_VAR) or "claude"`, and the
    real /home/leo/.local/bin/claude is on the test runner's PATH. So if that
    env-var lookup ever regresses -- exactly what 4510 exists to catch --
    pointing LEGIBILITY_CLAUDE_BIN at a nonexistent path becomes a no-op,
    resolution falls through to the bare name, and this test spawns up to
    N GENUINE Haiku CLI calls: real spend, real wall-clock, and green for the
    wrong reason. With `claude` unresolvable, that same regression ENOENTs
    instead: loud and free.

    Deliberately NOT an empty PATH, and for a reason specific to THIS module
    (4510's is different -- its fake binaries are `#!/usr/bin/env bash` and
    need `env`): nightly's e2e path shells out to git by BARE NAME
    (`['git', '-C', ...]`), so an empty PATH would break the commit stage for
    a reason with nothing to do with the branch under test -- the same class
    of misleading failure this scrub exists to prevent. Do not "simplify" the
    retained stdlib bin dir away.
    """
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', f'{empty_bin}{os.pathsep}/usr/bin')
    assert shutil.which('claude') is None, (
        'PATH scrub failed: a real `claude` is still resolvable, so a '
        'regression in _invoke_cli\'s env-var branch would silently spawn '
        'the GENUINE CLI (real spend) instead of failing loudly'
    )


def test_missing_claude_binary_journals_both_halves_end_to_end(
    tmp_path, monkeypatch, caplog,
):
    """One journal, both sinks: the per-digest WARNING from
    `legibility.coder` naming what went wrong for that session, AND the
    aggregate ERROR from `legibility.nightly` carrying the same reason --
    which is what an operator would actually have had to read on
    2026-08-18."""
    work_cwd = str(tmp_path / 'work')
    _repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)
    projects_root = tmp_path / 'projects'
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl',
        cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    _scrub_path_of_claude(tmp_path, monkeypatch)
    monkeypatch.setenv('LEGIBILITY_CLAUDE_BIN', str(tmp_path / 'nonexistent-claude'))

    escalations = []
    with caplog.at_level(logging.DEBUG):
        # NO invoke= override: the real coder._invoke_cli seam runs.
        result = nightly.run_nightly(
            config_path=config_path,
            projects_root=projects_root,
            target_date=date(2026, 7, 13),
            now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=UTC),
            status_fetcher=None,
            poster=lambda url, env: escalations.append((url, env)),
        )

    assert result.exit_code == 1
    assert result.coder_status == 'failure'
    assert result.commit_made is False

    # Half one: the coder announced the failure for that specific digest.
    coder_warnings = [
        r for r in caplog.records
        if r.name == 'legibility.coder' and r.levelno >= logging.WARNING
    ]
    assert len(coder_warnings) == 1, (
        f'expected one per-digest WARNING; got '
        f'{[r.getMessage() for r in coder_warnings]}'
    )
    coder_message = coder_warnings[0].getMessage()
    assert 'claude CLI could not be started' in coder_message, coder_message
    assert 'No such file or directory' in coder_message, coder_message

    # Half two: the aggregate reached the journal at ERROR, reason intact,
    # even though this run posted its escalation to a recording fake.
    errors = [r for r in _nightly_warnings(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 1, (
        f'expected one aggregate ERROR; got {[r.getMessage() for r in errors]}'
    )
    aggregate = errors[0].getMessage()
    assert 'coder storm' in aggregate, aggregate
    assert 'claude CLI could not be started' in aggregate, (
        'the aggregate must carry the REASON, not just the count -- a bare '
        f'"1/1 digests failed" is what the incident already had. got '
        f'{aggregate!r}'
    )
