"""Deterministic, isolated real-git fixtures for the landed-sha ladder's pins.

``skills/_shared/deriving-landed-sha.md``'s characterization modules build
purpose-shaped histories in throwaway repos and query them with the same git
commands the ladder prescribes.  These helpers give every such repo a fixed
identity, no signing, optional date pinning, and the esc-3072-3 isolation
guards, so a fixture can never reach an enclosing repo and its shas are
stable across machines.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

# Deterministic identity + no signing, so subjects and shas are stable and
# the tests never depend on the developer's git config.
GIT_ENV_ARGS = [
    '-c', 'user.name=Test',
    '-c', 'user.email=test@test.com',
    '-c', 'commit.gpgsign=false',
]


def git_env(root: Path, date: str | None = None) -> dict[str, str]:
    """Env for git in *root*, discovery capped at *root* (esc-3072-3).

    Rationale: ``_orch_helpers.py::git_env_with_ceiling``.  *date* pins both
    committer and author date.
    """
    env = git_env_with_ceiling(root)
    if date is not None:
        env['GIT_COMMITTER_DATE'] = date
        env['GIT_AUTHOR_DATE'] = date
    return env


def git(root: Path, *args: str, date: str | None = None) -> str:
    """Run git in *root*; return stripped stdout, raising on non-zero.

    *date* as in :func:`git_env`.
    """
    return subprocess.run(
        ['git', *GIT_ENV_ARGS, *args],
        cwd=root, check=True, capture_output=True, text=True, env=git_env(root, date),
    ).stdout.strip()


def git_rc(root: Path, *args: str) -> int:
    """Run git in *root* and return only its exit code."""
    return subprocess.run(
        ['git', *GIT_ENV_ARGS, *args],
        cwd=root, capture_output=True, text=True, env=git_env(root),
    ).returncode


def commit(root: Path, subject: str, filename: str, date: str | None = None) -> str:
    """Write *filename*, commit it with *subject*, and return the new sha.

    Each branch writes its OWN file so sibling merges never conflict -- a
    conflict would abort the fixture mid-build and the test would pin a
    history other than the one it describes.
    """
    (root / filename).write_text(f'{subject}\n')
    git(root, 'add', filename)
    git(root, 'commit', '-m', subject, date=date)
    return git(root, 'rev-parse', 'HEAD')


def new_repo(root: Path) -> None:
    """Create *root* as a repository root, then pre-flight it (esc-3072-3).

    Rationale: ``_orch_helpers.py::assert_isolated_git_repo``; asserting here
    means a test added later cannot forget it.
    """
    root.mkdir()
    git(root, 'init', '-b', 'main')
    assert_isolated_git_repo(root)
