"""A seeded git repo whose HEAD commits a change on top of a known base, for diff-measuring eval tests."""
from __future__ import annotations

import subprocess
from pathlib import Path

from _git_fixtures import seed_repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def repo_with_commit(dest: Path, files: dict[str, str]) -> tuple[Path, str]:
    """A repo at *dest* and its base sha; HEAD commits *files* on top of that base.

    An empty *files* leaves HEAD at the base, so the base-to-HEAD diff is empty.
    """
    repo = seed_repo(dest)
    base = _git(repo, 'rev-parse', 'HEAD')
    if files:
        for relpath, text in files.items():
            target = repo / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-m', 'change under measurement')
    return repo, base
