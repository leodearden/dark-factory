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

from _git_fixtures import RepoSeed, seed_repo

_DIFF_HEADER = re.compile(r'^diff --git a/.+ b/(.+)$', re.MULTILINE)


@dataclass(frozen=True)
class MergedInBranch:
    repo: Path
    worktree: Path
    creation_sha: str
    main_tip: str


def git(cwd: Path, *args: str) -> str:
    """Run git in *cwd*; return its stripped stdout, raising on non-zero."""
    done = subprocess.run(
        ['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _commit(cwd: Path, files: Mapping[str, str], message: str) -> str:
    for rel, content in files.items():
        (cwd / rel).write_text(content)
    git(cwd, 'commit', '-q', '-am', message)
    return git(cwd, 'rev-parse', 'HEAD')


def branch_with_main_merged_in(
    tmp_path: Path, *, base_files: Mapping[str, str] | None = None,
) -> MergedInBranch:
    files = {'a.py': 'a = 1\n', 'm1.py': 'm1 = 1\n', **(base_files or {})}
    repo = seed_repo(tmp_path / 'repo', RepoSeed(files=tuple(files.items()), message='base'))
    creation_sha = git(repo, 'rev-parse', 'HEAD')
    worktree = tmp_path / 'wt'
    git(repo, 'worktree', 'add', '-q', '-b', 'task/50', str(worktree))
    _commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
    main_tip = _commit(repo, {'m1.py': 'm1 = 2\n'}, 'main advances')
    git(worktree, 'merge', '-q', '--no-edit', 'main')
    return MergedInBranch(
        repo=repo, worktree=worktree, creation_sha=creation_sha, main_tip=main_tip,
    )


def diffed_paths(diff: str) -> set[str]:
    """The post-image paths named by a unified diff's ``diff --git`` headers."""
    return set(_DIFF_HEADER.findall(diff))
