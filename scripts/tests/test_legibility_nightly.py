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

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from legibility import nightly
from legibility.config import load_config


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _write_config(
    root: Path, *, project_id: str, escalation_port: int = 8199, cwd_prefixes=None,
) -> Path:
    """Write a minimal valid docs/legibility/legibility.yaml under *root*."""
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
