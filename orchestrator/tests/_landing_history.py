"""A throwaway git repo on main plus the runs.db events that record landings on it.

Imported by bare module name, like ``_fake_systemctl``. The event payloads are
the ones ``shared/src/shared/quality_series/landings.py::landings`` reads, and
the runs.db carries ``orchestrator.event_store.EventStore``'s own schema.
Both esc-3072-3 guards from ``_orch_helpers`` apply: the ceiling environment
on every git call, and the repo-root pre-flight once ``git init`` has run.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Sequence
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.event_store import EventStore

_IDENTITY = (
    '-c', 'user.email=t@e', '-c', 'user.name=t',
    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
)


class LandingRepo:
    """A git repo at *root* on branch main whose commits are all empty; nothing is ever added."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._env = git_env_with_ceiling(root)
        self._git('init', '-b', 'main')
        assert_isolated_git_repo(root)
        self.fast_forward('seed')

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ['git', *_IDENTITY, *args], cwd=str(self.root), env=self._env,
            check=True, capture_output=True, text=True, timeout=30,
        )
        return completed.stdout.strip()

    def fast_forward(self, message: str) -> str:
        """An empty commit on main with one parent; returns its sha."""
        self._git('commit', '--allow-empty', '--no-verify', '-m', message)
        return self._git('rev-parse', 'HEAD')

    def merge(self, branch: str) -> str:
        """A ``--no-ff`` merge of a one-commit branch into main; returns the merge sha (two parents)."""
        self._git('checkout', '-b', branch, 'main')
        self._git('commit', '--allow-empty', '--no-verify', '-m', f'work on {branch}')
        self._git('checkout', 'main')
        self._git('merge', '--no-ff', '--no-verify', '-m', f'merge {branch}', branch)
        return self._git('rev-parse', 'HEAD')


def create_runs_db(path: Path) -> Path:
    """An empty runs.db at *path* carrying EventStore's schema."""
    EventStore(path, run_id='landing-history')
    return path


def _insert(conn: sqlite3.Connection, task_id: str, event_type: str, data: dict, at: str) -> None:
    conn.execute(
        'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
        "VALUES (?, 'landing-history', ?, ?, ?)",
        (at, task_id, event_type, json.dumps(data)),
    )


def record_merge_done(conn: sqlite3.Connection, task_id: str, sha: str, at: str) -> None:
    """A merge_finalized event with state done naming *sha*; the caller commits."""
    _insert(conn, task_id, 'merge_finalized', {'state': 'done', 'merge_sha': sha}, at)


def record_train(
    conn: sqlite3.Connection, tip_task: str, members: Sequence[str], sha: str, at: str,
) -> None:
    """A coalesced train_merged event whose members share *sha*; the caller commits."""
    _insert(conn, tip_task, 'train_merged',
            {'member_task_ids': list(members), 'merge_commit_sha': sha}, at)
