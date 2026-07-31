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

import gzip
import json
import logging
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from legibility import census_trigger, codebook, digest, nightly
from legibility.config import load_config


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

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
    not a reuse of the module under test)."""
    return cwd.replace('/', '-').replace('.', '-')


def _write_transcript(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-1',
) -> None:
    """Write a minimal fixture session transcript JSONL with a genuine user
    turn plus a structural tool-error (nonzero confusion signal): a user
    turn, an assistant tool_use, and a user tool_result carrying
    ``is_error: true`` with 'No such file or directory' content (matches
    sampling._NOT_FOUND_PATTERNS too, for a score > 0 on more than one
    signal class)."""
    lines = [
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {'role': 'user', 'content': 'please fix this confusing bug'},
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


def _write_transcript_gz(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-1',
) -> None:
    """Gzip variant of :func:`_write_transcript` — the on-disk form of an
    archived fleet session (``shared.transcript_archive`` writes
    ``<sid>.jsonl.gz``). Same nonzero-confusion content, gzip-compressed."""
    lines = [
        {
            'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'sessionId': session_id,
            'message': {'role': 'user', 'content': 'please fix this confusing bug'},
        },
        {
            'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'sessionId': session_id,
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
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(line) for line in lines) + '\n')


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


def test_select_scored_records_reads_gz_archive_root(tmp_path):
    # End-to-end proof that the shipped agent_transcript_roots is LIVE with no
    # operator flip: resolve (cfg.project_root) -> enumerate (walk the archive)
    # -> gz-read (_iter_json_lines) -> classify (encoded worktree parent dir).
    # An archived role transcript at the production nested layout
    # <archive>/<task_id>/<enc>/<sid>.jsonl.gz is enumerated ALONGSIDE an
    # (empty) ~/.claude/projects tree and classified 'orchestrated-task'.
    main_cwd = '/home/leo/src/dark-factory'
    worktree_cwd = '/home/leo/src/dark-factory/.worktrees/2573'
    encoded = '-home-leo-src-dark-factory--worktrees-2573'
    cfg = load_config(_write_config(
        tmp_path, project_id='dark_factory', cwd_prefixes=[main_cwd],
        agent_transcript_roots=['archive'],
    ))
    gz_path = tmp_path / 'archive' / '2573' / encoded / 'sess-x.jsonl.gz'
    _write_transcript_gz(
        gz_path, cwd=worktree_cwd, timestamp='2026-07-13T10:00:00Z', session_id='sess-x',
    )
    empty_projects_root = tmp_path / 'projects'  # never created — projects tree absent

    scored = nightly.select_scored_records(cfg, empty_projects_root, date(2026, 7, 13))

    assert len(scored) == 1
    assert scored[0].session.path == gz_path
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
    selected = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))
    assert caps == [nightly.DEFAULT_MAX_DIGEST_BYTES], (
        'the sampler must charge at the nightly cap, not at build_digest\'s own default'
    )

    # RENDER side: build_digests, called by run_nightly with no max_bytes
    # override, must reach the renderer with that SAME cap.
    caps.clear()
    nightly.build_digests(selected, build=recording_build)
    assert caps == [nightly.DEFAULT_MAX_DIGEST_BYTES]


def test_select_digest_sessions_selects_multi_mb_session_under_stock_budget(tmp_path):
    """A session whose RAW transcript dwarfs the whole nightly budget must
    still be selected — the live defect, end to end.

    Measured basis: an 879,254-byte transcript of this shape renders to a
    15,123-byte digest, so it costs ~5% of the 300_000-byte budget rather
    than 293% of it. Charged at raw transcript size the single reserve
    group here is skipped whole, the greedy leftover fill has nothing to
    add, and select_digest_sessions returns [] — which is exactly what the
    live nightly run did every night from 2026-07-16 to 2026-07-29.
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

    selected = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))

    assert [r.path for r in selected] == [session_path]


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
        selected = nightly.select_digest_sessions(
            cfg, projects_root, date(2026, 7, 13), rendered=rendered,
        )
        assert renders == [session_path], 'costing must render exactly once'

        digests, extractor_failures = nightly.build_digests(
            selected, build=counting_build, rendered=rendered,
        )
    finally:
        digest.build_digest = real_build

    # Still exactly one render in total: build_digests served the cache.
    assert renders == [session_path]
    assert extractor_failures == []
    assert digests == [
        digest.build_digest(
            session_path, agent_class_override=selected[0].stratum,
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

    selected = nightly.select_digest_sessions(cfg, projects_root, date(2026, 7, 13))
    digests, extractor_failures = nightly.build_digests(selected)

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
        now=datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc),
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


class _FakeHttpxResponse:
    def raise_for_status(self):
        pass


def test_default_poster_sends_streamable_http_accept_headers(monkeypatch):
    """Task 2953: the streamable-HTTP MCP transport 406s any tools/call POST
    lacking an Accept header covering both application/json and
    text/event-stream (verified live against a local MCP /mcp endpoint).
    `_default_poster`'s httpx import is lazy (httpx is not importable in
    this test env), so a fake `httpx` module is injected via sys.modules."""
    import sys

    captured_kwargs = {}

    fake_httpx = type(sys)('httpx')

    def _fake_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeHttpxResponse()

    fake_httpx.post = _fake_post
    monkeypatch.setitem(sys.modules, 'httpx', fake_httpx)

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

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
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

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
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

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
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

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
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

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
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
