"""stack_base_census: task branches stacked on an unlanded sibling before task
5618's stack-base record existed (task 6125).

Real temp git repositories throughout.  Members live at
``<repo>/.worktrees/<name>`` on branch ``task/<name>``, the layout
orchestrator/tests/test_git_ops_unstack.py::_add_member uses, so GitOps finds
the worktree that holds a branch.  Every rebase runs inside the member's own
worktree, so the branch reflog records ``rebase (finish): refs/heads/task/<name>
onto <sha>`` exactly as the orchestrator's stacker leaves it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from committed_git_repo import committed_repo

# ---------------------------------------------------------------------------
# Real-git fixture (sync helpers, local to this file)
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ['git', *args], cwd=cwd, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f'git {args} failed: {proc.stderr}'
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo on ``main`` with one seed commit and a repo-local identity."""
    root = tmp_path / 'repo'
    committed_repo(root, {'seed.txt': 'seed\n'})
    git(root, 'config', 'user.email', 'test@test.com')
    git(root, 'config', 'user.name', 'Test')
    git(root, 'config', 'commit.gpgsign', 'false')
    with (root / '.git' / 'info' / 'exclude').open('a') as exclude:
        exclude.write('.worktrees/\n')
    return root


def commit(wt: Path, files: Mapping[str, str], msg: str) -> str:
    """Write and commit exactly *files* in *wt*; return the new HEAD."""
    for rel, text in files.items():
        (wt / rel).write_text(text)
    git(wt, 'add', '--', *files)
    git(wt, 'commit', '-m', msg)
    return rev(wt, 'HEAD')


def add_member(repo: Path, name: str, start_ref: str, files: Mapping[str, str]) -> Path:
    """Worktree ``.worktrees/<name>`` on new branch ``task/<name>`` from *start_ref*,
    with one commit of *files*."""
    wt = repo / '.worktrees' / name
    git(repo, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref)
    commit(wt, files, f'{name}: write files')
    return wt


def rebase(wt: Path, *args: str) -> None:
    git(wt, 'rebase', *args)


def rev(repo: Path, ref: str) -> str:
    return git(repo, 'rev-parse', '--verify', f'{ref}^{{commit}}')
