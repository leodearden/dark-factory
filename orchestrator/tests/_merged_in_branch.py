"""A task branch with main merged into it after its creation-time base was stored.

The task 5695 shape, built from real git: the branch edits ``a.py``, main
edits ``m1.py`` in parallel, then the worktree merges main. A diff from
``creation_sha`` therefore includes main's churn; a diff from the fresh fork
point (``main_tip``) is the branch's own work alone.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DIFF_HEADER = re.compile(r'^diff --git a/.+ b/(.+)$', re.MULTILINE)


@dataclass(frozen=True)
class MergedInBranch:
    repo: Path
    worktree: Path
    creation_sha: str
    main_tip: str


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _commit(cwd: Path, files: Mapping[str, str], message: str) -> str:
    for rel, content in files.items():
        path = cwd / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(cwd, 'add', '-A')
    _git(cwd, 'commit', '-q', '-m', message)
    return _git(cwd, 'rev-parse', 'HEAD')


def branch_with_main_merged_in(
    tmp_path: Path, *, base_files: Mapping[str, str] | None = None,
) -> MergedInBranch:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'reviewer@example.com')
    _git(repo, 'config', 'user.name', 'Reviewer')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    creation_sha = _commit(
        repo, {'a.py': 'a = 1\n', 'm1.py': 'm1 = 1\n', **(base_files or {})}, 'base',
    )
    worktree = tmp_path / 'wt'
    _git(repo, 'worktree', 'add', '-q', '-b', 'task/50', str(worktree))
    _commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
    main_tip = _commit(repo, {'m1.py': 'm1 = 2\n'}, 'main advances')
    _git(worktree, 'merge', '-q', '--no-edit', 'main')
    return MergedInBranch(
        repo=repo, worktree=worktree, creation_sha=creation_sha, main_tip=main_tip,
    )


def diffed_paths(diff: str) -> set[str]:
    """The post-image paths named by a unified diff's ``diff --git`` headers."""
    return set(_DIFF_HEADER.findall(diff))
