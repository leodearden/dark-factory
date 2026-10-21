"""The payload fleet's on-disk half: the files the routes read beside the MCP reads.

``_payload_fleet.py`` answers every MCP read from its census; the routes this
module feeds read files instead: escalation queues and their archives, each
root's run ledger, its task store and the memory-eval artifacts. Each seeder
writes one family through the production writer or schema wherever one is
importable, at the sizes its ``*Shape`` names, and knows nothing of MCP.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from escalation.archive import archive_dir_for_date
from escalation.models import Escalation
from escalation.queue import EscalationQueue


def text(seed: str, length: int) -> str:
    """*seed*, padded with ASCII filler to exactly *length* characters."""
    if len(seed) >= length:
        raise ValueError(f'seed {seed!r} does not fit in {length} characters')
    return seed + ' ' + 'x' * (length - len(seed) - 1)


@dataclass(frozen=True)
class EscalationShape:
    """The sizes an escalation record is written at, and how far back resolved history reaches."""

    detail_len: int
    summary_len: int
    triage_note_len: int
    option_len: int
    history_days: int


_RESOLVERS = ('interactive', 'escalation-watcher-auto', 'harness-orphan-reaper', 'claude-task-1-steward')
"""One resolver per tier the analytics split on: human, auto-watcher, reaper sweep, steward."""
_ROLES = ('implementer', 'steward', 'escalation-watcher-auto', 'merger')
_CATEGORIES = ('design_concern', 'infra_issue', 'risk_identified', 'cleanup_needed')


def _record(shape: EscalationShape, esc_id: str, task_id: str, filed: datetime, n: int) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=_ROLES[(n // 7) % len(_ROLES)],
        severity='blocking',
        category=_CATEGORIES[n % len(_CATEGORIES)],
        summary=text(f'fleet escalation {esc_id}', shape.summary_len),
        detail=text(f'fleet escalation detail {esc_id}', shape.detail_len),
        timestamp=filed.isoformat(),
        level=n % 3,
        options=[text('A: fleet option', shape.option_len)],
        triage_note=text('fleet triage note', shape.triage_note_len),
    )


def seed_pending_escalations(
    queue_dir: Path,
    shape: EscalationShape,
    task_ids: Sequence[str],
    *,
    now: datetime,
    worktree_root: Path | None = None,
) -> None:
    """File one pending escalation per entry of *task_ids* into the live queue at *queue_dir*.

    *worktree_root*, when given, puts each record's worktree under it: how a
    reconciliation record names the root that owns its task.
    """
    queue = EscalationQueue(queue_dir)
    for n, task_id in enumerate(task_ids):
        record = _record(shape, f'esc-{task_id}-{n + 1}', task_id, now - timedelta(hours=n + 1), n)
        if worktree_root is not None:
            record = replace(record, worktree=str(worktree_root / '.worktrees' / task_id))
        queue.submit(record)


def seed_resolved_history(queue_dir: Path, shape: EscalationShape, count: int, *, now: datetime) -> None:
    """Write *count* resolved escalations into *queue_dir*'s dated archive, spread over the history.

    Records pair up on one task a day apart, so every second filing re-files
    a task resolved the day before: the churn the workflow view counts.
    """
    for n in range(count):
        filed = now - timedelta(days=n % shape.history_days, hours=6)
        resolved_at = filed + timedelta(minutes=30 + 7 * (n % 300))
        record = replace(
            _record(shape, f'esc-h{n}-1', f'h{n // 2}', filed, n),
            status='resolved',
            resolved_at=resolved_at.isoformat(),
            resolved_by=_RESOLVERS[(n // 3) % len(_RESOLVERS)],
            resolution='fleet resolution',
        )
        archive = archive_dir_for_date(queue_dir, resolved_at.isoformat())
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f'{record.id}.json').write_text(record.to_json())


_TASK_RESULTS_DDL = """\
CREATE TABLE IF NOT EXISTS task_results (
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
    PRIMARY KEY (run_id, task_id)
);
"""
"""The columns the dashboard reads of the orchestrator's ledger
(orchestrator/src/orchestrator/run_store.py::_SCHEMA); the dashboard does not depend on that package."""


def seed_task_results(runs_db: Path, project_id: str, completed_hours: int, *, now: datetime) -> None:
    """One finished task per hour over the last *completed_hours* hours, in *runs_db*'s ledger.

    Every completion lands in its own hour, so the performance histories,
    which bucket by hour, hold one bucket per completion.
    """
    runs_db.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            'run-payload-fleet', str(hour), project_id, None, 'done', 0.5,
            600_000 + (997 * hour) % 3_600_000, 3, 2, 1 + hour % 2, hour % 2, 0.0, int(hour % 9 == 0),
            (now - timedelta(hours=hour)).isoformat(),
        )
        for hour in range(1, completed_hours + 1)
    ]
    with closing(sqlite3.connect(runs_db)) as conn, conn:
        conn.executescript(_TASK_RESULTS_DDL)
        conn.executemany(f'INSERT INTO task_results VALUES ({", ".join("?" * 14)})', rows)


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
"""
"""The orchestrator's event ledger (orchestrator/src/orchestrator/event_store.py::_SCHEMA), without its indexes."""

_MERGE_OUTCOMES = ('done', 'done', 'done', 'conflict', 'blocked', 'cas_retry')


def seed_merge_events(
    runs_db: Path, task_ids: Sequence[str], merges: int, trains: int, *, history_days: int, now: datetime,
) -> None:
    """*merges* merge attempts spread evenly over *history_days*, and the *trains* newest train merges.

    Each attempt names a task of *task_ids* in turn, so the recent table's
    title lookups resolve against the fleet.
    """
    runs_db.parent.mkdir(parents=True, exist_ok=True)
    span = timedelta(days=history_days)
    attempts = [
        (
            (now - span * (n + 1) / (merges + 1)).isoformat(), 'run-payload-fleet', task_ids[n % len(task_ids)],
            'merge_attempt', 'merge', json.dumps({'outcome': _MERGE_OUTCOMES[n % len(_MERGE_OUTCOMES)]}),
            60_000 + (7_919 * n) % 1_800_000,
        )
        for n in range(merges)
    ]
    train_merges = [
        (
            (now - timedelta(hours=6 * (n + 1))).isoformat(), 'run-payload-fleet', task_ids[n % len(task_ids)],
            'train_merged', 'merge', json.dumps({
                'train_id': f'coalesce-{task_ids[n % len(task_ids)]}-fleet',
                'member_task_ids': [task_ids[(n + k) % len(task_ids)] for k in range(2)],
                'merge_commit_sha': text('fleet-merge-sha', 40), 'base_sha': text('fleet-base-sha', 40),
            }),
            None,
        )
        for n in range(trains)
    ]
    with closing(sqlite3.connect(runs_db)) as conn, conn:
        conn.executescript(_EVENTS_DDL)
        conn.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data, duration_ms) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            [*attempts, *train_merges],
        )
