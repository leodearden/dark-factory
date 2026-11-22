"""A throwaway git repo on main plus the runs.db events that record landings on it.

A plain module rather than a conftest fixture, the same convention as
``committed_git_repo``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name. The event payloads are the ones
``shared/src/shared/quality_series/landings.py::landings`` reads.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Sequence
from pathlib import Path

_IDENTITY = (
    '-c', 'user.email=t@e', '-c', 'user.name=t',
    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
)


class LandingRepo:
    """A git repo at *root* on branch main whose commits are all empty; nothing is ever added."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._git('init', '-b', 'main')
        self.fast_forward('seed')

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ['git', '-C', str(self.root), *_IDENTITY, *args],
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
