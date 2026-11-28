"""A throwaway git repo plus the runs.db events that record landings on it.

The event payloads are the ones ``shared/src/shared/quality_series/landings.py::landings``
reads, so this module is their one test-side copy. Both test roots that build landing
history import it, each through a thin factory that supplies its own git containment:
``orchestrator/tests/_landing_history.py`` and ``scripts/tests/landing_history.py``.

It is a testing module: no src module imports it. It writes no runs.db schema of its
own; the caller's connection already carries ``orchestrator.event_store.EventStore``'s.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from shared.testing_own_time import run_on_own_time

__all__ = ['GIT_CALL_BUDGET_S', 'LandingRepo', 'record_merge_done', 'record_train']

#: Own time per git call; each is an empty commit, a checkout, a merge or a rev-parse.
GIT_CALL_BUDGET_S = 30.0

_CEILING = 'GIT_CEILING_DIRECTORIES'
_IDENTITY = (
    '-c', 'user.email=t@e', '-c', 'user.name=t',
    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
)


class LandingRepo:
    """A git repo at *root* on *branch* whose commits are all empty; nothing is ever added.

    Every git call runs in *root* under *env*, bounded by *budget* seconds of its own time.
    *env* must carry a ``GIT_CEILING_DIRECTORIES`` entry naming *root*'s resolved parent,
    so git's upward walk cannot leave *root* (incident esc-3072-3); a missing one raises
    ValueError before any git call.
    """

    def __init__(
        self, root: Path, *, env: Mapping[str, str], branch: str = 'main',
        budget: float = GIT_CALL_BUDGET_S,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        ceiling = str(root.resolve().parent)
        if ceiling not in env.get(_CEILING, '').split(':'):
            raise ValueError(f'{_CEILING} must name {ceiling}, or git may leave {root}')
        self.root = root
        self.branch = branch
        self._env = dict(env)
        self._budget = budget
        self._git('init', '-b', branch)
        self.fast_forward('seed')

    def _git(self, *args: str) -> str:
        completed = run_on_own_time(
            ['git', *_IDENTITY, *args], cwd=self.root, env=self._env, budget=self._budget,
        )
        completed.check_returncode()
        return completed.stdout.strip()

    def fast_forward(self, message: str) -> str:
        """An empty commit on the branch with one parent; returns its sha."""
        self._git('commit', '--allow-empty', '--no-verify', '-m', message)
        return self._git('rev-parse', 'HEAD')

    def merge(self, topic: str) -> str:
        """A ``--no-ff`` merge of a one-commit *topic* branch; returns the merge sha (two parents)."""
        self._git('checkout', '-b', topic, self.branch)
        self._git('commit', '--allow-empty', '--no-verify', '-m', f'work on {topic}')
        self._git('checkout', self.branch)
        self._git('merge', '--no-ff', '--no-verify', '-m', f'merge {topic}', topic)
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
