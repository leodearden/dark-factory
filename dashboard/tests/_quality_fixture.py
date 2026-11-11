"""A real quality project for the dashboard tests: a git repo on ``main`` plus its runs.db.

The builders hold no expectations; tests do. ``INVOCATIONS_DDL`` is the ``invocations`` table
of shared/src/shared/cost_store.py::_SCHEMA, ``FLAKE_OCCURRENCE_DDL`` the
``flake_occurrence`` table of orchestrator/src/orchestrator/flake_ledger.py::_SCHEMA and
``RUNS_DDL`` the ``runs`` table of orchestrator/src/orchestrator/run_store.py::_SCHEMA, copied
because shared/src/shared/quality_series/rows.py::rows_for and the quality marker derivation
read them and the dashboard does not depend on the orchestrator. Event payloads follow
shared/src/shared/quality_series/landings.py::_DONE_SQL and ::_COALESCED_TRAIN_SQL; the
``routing_decision`` and ``config_reload`` payloads follow
orchestrator/src/orchestrator/workflow.py::_record_routing_decision and the ``config_reload``
entry of orchestrator/src/orchestrator/event_store.py::EventType.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from _runs_db_fixture import EventRow, TaskResultRow, write_runs_db

INVOCATIONS_DDL = """\
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

FLAKE_OCCURRENCE_DDL = """\
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


RUNS_DDL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0,
    code_sha       TEXT
);
"""


def routing_decision(
    ts: datetime | str,
    role: str,
    *,
    model: str,
    effort: str,
    budget_usd: float,
    max_turns: int,
    source_layer: str = 'config',
    rule_id: str | None = None,
    prompt_digest: str | None = None,
) -> EventRow:
    """One ``routing_decision`` event; the role rides in the payload (write_runs_db sets no role column)."""
    data: dict[str, Any] = {
        'role': role, 'model': model, 'effort': effort, 'budget_usd': budget_usd,
        'max_turns': max_turns, 'source_layer': source_layer, 'rule_id': rule_id,
        'rejected': [], 'inputs_digest': 'fixture',
    }
    if prompt_digest is not None:
        data['prompt_digest'] = prompt_digest
    return (ts, None, 'routing_decision', data)


def config_reload(
    ts: datetime | str,
    *,
    applied: Mapping[str, object] | None = None,
    restart_required: Mapping[str, object] | None = None,
) -> EventRow:
    """One ``config_reload`` report; each change maps a config path to ``{old, new}`` (or, to
    write a malformed report, to anything else)."""
    return (ts, None, 'config_reload', {
        'reloaded': True,
        'applied': dict(applied or {}),
        'restart_required': dict(restart_required or {}),
        'unchanged': 0,
        'error': None,
    })


def write_runs_rows(
    store: Path, rows: Iterable[tuple[str, datetime | str, str | None]],
) -> None:
    """Add a ``runs`` table to *store* holding each (run_id, started_at, code_sha)."""
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(RUNS_DDL)
        conn.executemany(
            "INSERT INTO runs (run_id, project_id, started_at, code_sha) VALUES (?, 'fixture', ?, ?)",
            [(run_id, _iso(started_at), code_sha) for run_id, started_at, code_sha in rows],
        )
        conn.commit()


def write_invocations(
    store: Path, rows: Iterable[tuple[str, str | None, datetime | str]],
) -> None:
    """Add an ``invocations`` table to *store* holding each (model, model_id, started_at)."""
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(INVOCATIONS_DDL)
        conn.executemany(
            'INSERT INTO invocations (run_id, project_id, account_name, model, role, '
            'started_at, completed_at, model_id) '
            "VALUES ('run-fixture', 'fixture', 'fixture', ?, 'implementer', ?, ?, ?)",
            [
                (model, _iso(started_at), _iso(started_at), model_id)
                for model, model_id, started_at in rows
            ],
        )
        conn.commit()


def append_events(store: Path, events: Iterable[EventRow]) -> None:
    """Append *events* to an existing runs.db, after every row already in it."""
    with closing(sqlite3.connect(store)) as conn:
        conn.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, data) VALUES (?, ?, ?, ?, ?)',
            [
                (_iso(timestamp), f'run-{task_id}', task_id, event_type,
                 data if isinstance(data, str) else json.dumps(dict(data)))
                for timestamp, task_id, event_type, data in events
            ],
        )
        conn.commit()


def _iso(instant: datetime | str) -> str:
    return instant.isoformat() if isinstance(instant, datetime) else instant


def write_quality_runs_db(
    root: Path, events: Iterable[EventRow], task_results: Iterable[TaskResultRow] = (),
) -> Path:
    """*root*'s runs.db with every table rows_for reads; event timestamps must be tz-aware."""
    store = write_runs_db(root, events, task_results)
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(INVOCATIONS_DDL + FLAKE_OCCURRENCE_DDL)
        conn.commit()
    return store


_IDENTITY = (
    '-c', 'user.name=fixture',
    '-c', 'user.email=fixture@example.invalid',
    '-c', 'commit.gpgsign=false',
)


class GitRepo:
    """A repository at ``root`` on branch ``main``; each writer returns the sha it made."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ['git', '-C', str(self.root), *_IDENTITY, *args],
            check=True, capture_output=True, text=True,
            env={**os.environ, 'GIT_CONFIG_NOSYSTEM': '1'},
        )
        return result.stdout.strip()

    def init(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        self.git('init', '-q', '-b', 'main')
        return self.commit({'README.md': 'fixture\n'}, 'initial commit')

    def commit(self, files: Mapping[str, str], message: str) -> str:
        for rel, content in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.git('add', '--', *files)
        self.git('commit', '-q', '--no-verify', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def merge(self, branch: str, files: Mapping[str, str], message: str) -> str:
        self.git('checkout', '-q', '-b', branch, 'main')
        self.commit(files, f'{branch}: work')
        self.git('checkout', '-q', 'main')
        self.git('merge', '-q', '--no-ff', '--no-verify', '-m', message, branch)
        return self.git('rev-parse', 'HEAD')


def _lines(count: int, tag: str) -> str:
    return ''.join(f'{tag} line {index}\n' for index in range(count))


@dataclass(frozen=True)
class QualityProject:
    """main: initial commit, 101's merge, the 102+103 train merge, then a merge runs.db never saw."""

    root: Path
    plain_sha: str
    train_sha: str
    plain_landed_at: datetime
    train_landed_at: datetime
    train_members: tuple[str, ...] = ('102', '103')


def build_quality_project(root: Path, *, first_landed_at: datetime) -> QualityProject:
    repo = GitRepo(root)
    repo.init()
    plain_sha = repo.merge(
        'task/101',
        {'pkg/src/pkg/a.py': _lines(5, 'a'), 'pkg/tests/test_a.py': _lines(3, 'test a')},
        'Merge task/101 into main',
    )
    train_sha = repo.merge(
        'task/103', {'pkg/src/pkg/b.py': _lines(4, 'b')}, 'Merge task/103 into main',
    )
    repo.merge('hand/fix', {'pkg/src/pkg/c.py': _lines(2, 'c')}, 'Merge hand/fix into main')
    train_landed_at = first_landed_at + timedelta(hours=1)
    project = QualityProject(
        root=root,
        plain_sha=plain_sha,
        train_sha=train_sha,
        plain_landed_at=first_landed_at,
        train_landed_at=train_landed_at,
    )
    write_quality_runs_db(root, [
        (first_landed_at, '101', 'merge_finalized', {'state': 'done', 'merge_sha': plain_sha}),
        (train_landed_at, project.train_members[-1], 'train_merged', {
            'member_task_ids': list(project.train_members), 'merge_commit_sha': train_sha,
        }),
    ])
    return project
