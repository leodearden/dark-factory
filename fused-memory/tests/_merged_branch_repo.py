"""Real-git building blocks for a task branch merged into ``main`` with ``--no-ff``.

The fixtures are real git, never a canned ``run_git`` fake, for the reason
``tests/test_landed_on_main.py`` gives: the verdict IS git semantics, so a
canned fake would only pin our assumptions about git.

Start every repo with ``_fm_helpers._init_git_repo`` (``main`` branch, one seed
commit). Lives beside ``_fm_helpers.py`` under the convention documented in
tests/conftest.py, which puts this directory on ``sys.path`` for ``tests/`` and
``tests/reconciliation/`` alike.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

INCIDENT_GUARD = 'tests/unit/guard.py'
INCIDENT_BRIEFING = 'review/briefing.yaml'


def git(repo: Path, *args: str) -> str:
    """Run ``git -C <repo> <args>``, failing loudly; return stripped stdout."""
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_file(repo: Path, path: str, text: str, subject: str) -> str:
    """Write *text* to *path* (creating parent dirs), commit it; return the sha."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    git(repo, 'add', '--', path)
    git(repo, 'commit', '-q', '-m', subject)
    return git(repo, 'rev-parse', 'HEAD')


def delete_file(repo: Path, path: str, subject: str) -> str:
    """``git rm`` *path* and commit; return the sha."""
    git(repo, 'rm', '-q', '--', path)
    git(repo, 'commit', '-q', '-m', subject)
    return git(repo, 'rev-parse', 'HEAD')


def merge_no_ff(repo: Path, branch: str) -> str:
    """Merge *branch* into ``main`` the way the merge lane does, delete it; return the merge sha."""
    git(repo, 'checkout', '-q', 'main')
    git(repo, 'merge', '-q', '--no-ff', '-m', f'Merge {branch} into main', branch)
    git(repo, 'branch', '-q', '-D', branch)
    return git(repo, 'rev-parse', 'HEAD')


@dataclass(frozen=True)
class IncidentMerge:
    merge: str
    guard_commits: tuple[str, ...]


def incident_merge(repo: Path, task_id: str) -> IncidentMerge:
    """The solar_challenge task 107 shape, merged from ``task/<task_id>``.

    ``main`` gains INCIDENT_BRIEFING; the branch adds, edits and deletes
    INCIDENT_GUARD (``guard_commits``, oldest first) and edits the briefing.
    Contents carry *task_id*, so one repo can hold several incidents.
    """
    branch = f'task/{task_id}'
    commit_file(repo, INCIDENT_BRIEFING, f'{task_id} v1\n', 'seed briefing')
    git(repo, 'checkout', '-q', '-b', branch, 'main')
    added = commit_file(repo, INCIDENT_GUARD, f'{task_id} one\n', 'add guard')
    commit_file(repo, INCIDENT_BRIEFING, f'{task_id} v2\n', 'edit briefing')
    edited = commit_file(repo, INCIDENT_GUARD, f'{task_id} two\n', 'edit guard')
    dropped = delete_file(repo, INCIDENT_GUARD, 'drop guard')
    return IncidentMerge(merge=merge_no_ff(repo, branch), guard_commits=(added, edited, dropped))
