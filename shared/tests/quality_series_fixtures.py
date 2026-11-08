"""Builders for the quality_series tests: a runs.db carrying production DDL, and a real git repo.

The four CREATE TABLE statements copy the production column lists (names and types) of
orchestrator/src/orchestrator/event_store.py::_SCHEMA (events),
shared/src/shared/cost_store.py::_SCHEMA (invocations),
orchestrator/src/orchestrator/run_store.py (task_results) and
orchestrator/src/orchestrator/flake_ledger.py::_SCHEMA (flake_occurrence); shared tests may not
import orchestrator, so they are restated here. The builders hold no expectations: tests do.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Timestamp = datetime | str

_EVENTS_DDL = """\
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""

_INVOCATIONS_DDL = """\
CREATE TABLE IF NOT EXISTS invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL,
    model_id            TEXT,
    capped_reason       TEXT
);
"""

_TASK_RESULTS_COLUMNS = """\
    run_id              TEXT NOT NULL,
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    block_reason        TEXT,
    block_phase         TEXT"""

_FLAKE_DDL = """\
CREATE TABLE IF NOT EXISTS flake_occurrence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT    NOT NULL,
    test_id         TEXT    NOT NULL,
    project_id      TEXT    NOT NULL,
    verdict         TEXT    NOT NULL,
    call_site       TEXT    NOT NULL,
    runner          TEXT,
    merge_sha       TEXT,
    task_id         TEXT,
    psi_cpu_some10  REAL,
    detail          TEXT    DEFAULT '{}'
);
"""

_RUN_ID = 'run-fixture'
_PROJECT_ID = 'fixture_project'


def iso(ts: Timestamp) -> str:
    """A '+00:00' ISO string, the form production writes."""
    if isinstance(ts, str):
        return ts
    return ts.astimezone(UTC).isoformat()


def _task_results_ddl(with_charter_class: bool) -> str:
    extra = ',\n    charter_class       TEXT' if with_charter_class else ''
    return (
        'CREATE TABLE IF NOT EXISTS task_results (\n'
        f'{_TASK_RESULTS_COLUMNS}{extra},\n'
        '    PRIMARY KEY (run_id, task_id)\n);\n'
    )


class RunsDbBuilder:
    """An autocommit runs.db at ``path``; ``conn`` is the connection the tests hand to readers."""

    def __init__(self, path: Path, *, with_charter_class: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.with_charter_class = with_charter_class
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.executescript(
            _EVENTS_DDL + _INVOCATIONS_DDL + _task_results_ddl(with_charter_class) + _FLAKE_DDL
        )

    def close(self) -> None:
        self.conn.close()

    def event(
        self,
        ts: Timestamp,
        event_type: str,
        task_id: str | None,
        data: Mapping[str, Any],
        role: str | None = None,
        phase: str | None = None,
    ) -> None:
        self.conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, role, data)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?)',
            (iso(ts), _RUN_ID, task_id, event_type, phase, role, json.dumps(dict(data))),
        )

    def merge_done(
        self, ts: Timestamp, task_id: str, sha: str, landed_via_chain: str | None = None
    ) -> None:
        self.event(ts, 'merge_finalized', task_id, {
            'request_id': f'mr-{task_id}-{sha[:8]}',
            'branch': task_id,
            'state': 'done',
            'merge_sha': sha,
            'superseded_by': None,
            'landed_via_chain': landed_via_chain,
        }, phase='merge')

    def merge_state(
        self, ts: Timestamp, task_id: str, state: str, superseded_by: str | None = None
    ) -> None:
        self.event(ts, 'merge_finalized', task_id, {
            'request_id': f'mr-{task_id}-{state}',
            'branch': task_id,
            'state': state,
            'merge_sha': None,
            'superseded_by': superseded_by,
            'landed_via_chain': None,
        }, phase='merge')

    def train_merged(
        self, ts: Timestamp, tip_task: str, members: Sequence[str], sha: str, base_sha: str
    ) -> None:
        self.event(ts, 'train_merged', tip_task, {
            'train_id': f'coalesce-{tip_task}-{sha[:8]}',
            'member_task_ids': list(members),
            'merge_commit_sha': sha,
            'base_sha': base_sha,
        }, phase='merge')

    def merge_verify(
        self, ts: Timestamp, task_id: str, sha: str, passed: bool, chain_items: int | None = 1
    ) -> None:
        self.event(ts, 'merge_verify', task_id, {
            'runner': 'local',
            'merge_sha': sha,
            'passed': passed,
            'attempt': 0,
            'chain_items': chain_items,
        })

    def merge_blocked(self, ts: Timestamp, task_id: str, category: str, reason: str) -> None:
        self.event(ts, 'merge_blocked', task_id, {
            'reason': reason,
            'category': category,
        }, phase='merge')

    def review_completed(
        self, ts: Timestamp, task_id: str, cycle: int, has_blocking: bool
    ) -> None:
        self.event(ts, 'review_completed', task_id, {
            'cycle': cycle,
            'has_blocking_issues': has_blocking,
            'blocking_count': 1 if has_blocking else 0,
        }, phase='review')

    def invocation_end(
        self,
        ts: Timestamp,
        task_id: str,
        role: str,
        turns: int | None,
        capped_reason: str | None = None,
    ) -> None:
        self.event(ts, 'invocation_end', task_id, {
            'turns': turns,
            'success': capped_reason is None,
            'capped': capped_reason is not None,
            'capped_reason': capped_reason,
        }, role=role, phase='execute')

    def invocation(
        self,
        task_id: str,
        role: str,
        cost_usd: float,
        started_at: Timestamp,
        completed_at: Timestamp | None = None,
    ) -> None:
        self.conn.execute(
            'INSERT INTO invocations (run_id, task_id, project_id, account_name, model, role,'
            ' cost_usd, started_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                _RUN_ID, task_id, _PROJECT_ID, 'fixture-account', 'opus', role, cost_usd,
                iso(started_at), iso(completed_at if completed_at is not None else started_at),
            ),
        )

    def task_result(
        self,
        task_id: str,
        run_id: str,
        outcome: str,
        completed_at: Timestamp | None,
        execute_iterations: int = 0,
        block_reason: str | None = None,
        block_phase: str | None = None,
        charter_class: str | None = None,
    ) -> None:
        if charter_class is not None and not self.with_charter_class:
            raise ValueError(
                f'charter_class={charter_class!r} given to a builder made without the column'
            )
        columns = [
            'run_id', 'task_id', 'project_id', 'outcome', 'completed_at',
            'execute_iterations', 'block_reason', 'block_phase',
        ]
        values: list[Any] = [
            run_id, task_id, _PROJECT_ID, outcome,
            iso(completed_at) if completed_at is not None else None,
            execute_iterations, block_reason, block_phase,
        ]
        if self.with_charter_class:
            columns.append('charter_class')
            values.append(charter_class)
        self.conn.execute(
            f'INSERT INTO task_results ({", ".join(columns)})'
            f' VALUES ({", ".join("?" for _ in columns)})',
            values,
        )

    def flake(
        self,
        observed_at: Timestamp,
        task_id: str,
        verdict: str,
        test_id: str = 't::x',
        call_site: str = 'merge_gate',
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            'INSERT INTO flake_occurrence (observed_at, test_id, project_id, verdict, call_site,'
            ' task_id, detail) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                iso(observed_at), test_id, _PROJECT_ID, verdict, call_site, task_id,
                json.dumps(dict(detail or {})),
            ),
        )


FileContents = Mapping[str, str | bytes]


class GitRepoBuilder:
    """A real repository on branch ``main`` whose helpers return the commit shas they make."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._side_branches = 0
        self.git('init', '-q', '-b', 'main', str(root))
        self.git('config', 'user.email', 'quality-series-test@example.com')
        self.git('config', 'user.name', 'Quality Series Test')
        self.git('config', 'commit.gpgsign', 'false')

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ['git', '-C', str(self.root), *args], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    def head(self) -> str:
        return self.git('rev-parse', 'HEAD')

    def _write(self, files: FileContents) -> None:
        for rel, content in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)

    def commit(self, files: FileContents, message: str) -> str:
        self._write(files)
        self.git('add', '-A')
        self.git('commit', '-q', '--allow-empty', '--no-verify', '-m', message)
        return self.head()

    def branch_merge(
        self, task_id: str, files: FileContents, message: str | None = None
    ) -> str:
        branch = f'task/{task_id}'
        self.git('checkout', '-q', '-B', branch, 'main')
        self.commit(files, f'{task_id}: work')
        self.git('checkout', '-q', 'main')
        self.git(
            'merge', '-q', '--no-ff', '--no-verify', '--no-edit',
            '-m', message or f'Merge {branch} into main', branch,
        )
        return self.head()

    def ff_commits(self, files_list: Sequence[FileContents]) -> list[str]:
        return [
            self.commit(files, f'direct commit {index}') for index, files in enumerate(files_list)
        ]

    def side_commit(self, files: FileContents) -> str:
        self._side_branches += 1
        branch = f'side/{self._side_branches}'
        self.git('checkout', '-q', '-b', branch, 'main')
        sha = self.commit(files, f'side commit {self._side_branches}')
        self.git('checkout', '-q', 'main')
        return sha
