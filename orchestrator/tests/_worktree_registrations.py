"""What git has on record for a real repo's linked worktrees.

:func:`registered_worktree_paths` is git's own list
(``git worktree list --porcelain``); :func:`lane_admin_dir` follows a
worktree's ``.git`` pointer file to its ``.git/worktrees/<name>`` admin
entry.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def lane_admin_dir(worktree: Path) -> Path:
    """The admin entry named by *worktree*'s ``gitdir: <path>`` pointer file."""
    content = (worktree / '.git').read_text().strip()
    prefix = 'gitdir:'
    assert content.startswith(prefix), f'unexpected worktree .git pointer: {content!r}'
    return Path(content[len(prefix):].strip())


def registered_worktree_paths(repo: Path) -> set[str]:
    """Every worktree path git lists for *repo*, resolved, as strings."""
    porcelain = subprocess.run(
        ['git', 'worktree', 'list', '--porcelain'],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    return {
        str(Path(line[len('worktree '):]).resolve())
        for line in porcelain.splitlines()
        if line.startswith('worktree ')
    }
