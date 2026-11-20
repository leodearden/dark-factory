"""Scaffolding shared by the orchestrator/tests/test_b3_gate*.py modules; import it, do not copy it."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from _git_fixtures import README_SEED, RepoSeed, seed_repo

RECORDED_HEAD_SHA = 'aaabbbccc'
RECORDED_MAIN_SHA = 'dddeeefff'


def low_risk_entry(**overrides: Any) -> dict[str, Any]:
    """A new low-risk proposal entry on foo.py anchored at the recorded shas, with *overrides* applied."""
    return {
        'proposal_text': 'Fix it',
        'risk_label': 'low',
        'files_referenced': ['foo.py'],
        'block_reason': 'test',
        'investigated_at': '2026-06-04T09:00:00+00:00',
        'head_sha': RECORDED_HEAD_SHA,
        'main_sha': RECORDED_MAIN_SHA,
        **overrides,
    }


def fake_git_fresh(args, cwd):
    """Fake git runner: HEAD == recorded sha, diff is empty (fresh)."""
    if 'rev-parse' in args:
        return (0, RECORDED_HEAD_SHA)
    return (0, '')


def fake_git_never_called(args, cwd):
    raise AssertionError(f'git should not have been called; args={args}')


def fake_git_diff_fails(args, cwd):
    """Fake git runner: rev-parse HEAD succeeds (P1 passes), but the footprint diff fails."""
    if 'rev-parse' in args:
        return (0, RECORDED_HEAD_SHA)
    return (128, f'fatal: bad revision {RECORDED_MAIN_SHA}..main')


def _git_stdout(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def diverged_repo(tmp_path: Path, seed: RepoSeed = README_SEED) -> tuple[Path, str, str]:
    """Seed <tmp_path>/repo from *seed* on main, then check out a 'feature' branch one commit ahead.

    Returns (repo, head_sha, main_sha), where head_sha is the feature commit.
    """
    repo = seed_repo(tmp_path / 'repo', seed)
    main_sha = _git_stdout(repo, 'rev-parse', 'main')
    _git_stdout(repo, 'checkout', '-b', 'feature')
    (repo / 'extra.txt').write_text('feature work')
    _git_stdout(repo, 'add', '.')
    _git_stdout(repo, 'commit', '-m', 'feature commit')
    head_sha = _git_stdout(repo, 'rev-parse', 'HEAD')
    assert head_sha != main_sha
    return repo, head_sha, main_sha
