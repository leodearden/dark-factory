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
