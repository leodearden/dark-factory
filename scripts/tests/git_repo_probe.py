"""Read-only probes of a real git repo, for tests that assert what a git job did to it.

A plain module rather than a conftest fixture, the same convention as
``committed_git_repo``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    """``git -C <repo> <args>`` stdout; raises on a non-zero exit."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout


def loose_count(repo: Path) -> int:
    """The ``count`` line of ``git count-objects -v``: *repo*'s loose objects."""
    for line in git(repo, "count-objects", "-v").splitlines():
        key, _, value = line.partition(":")
        if key == "count":
            return int(value)
    raise AssertionError(f"count-objects -v printed no count for {repo}")
