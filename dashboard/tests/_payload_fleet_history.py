"""The payload fleet's sampled histories: the stores something writes to on a clock.

``_payload_fleet_seed.py`` writes what the orchestrators file and bill; this
module writes what is SAMPLED: the burndown series and the metrics.db sample
tables the dashboard's own loops keep, and the reconciliation store and write
journal fused-memory keeps. Each seeder writes one store through its
production schema where the dashboard owns one, at the cadence its writer
keeps and the sizes its ``*Shape`` names, and knows nothing of MCP.

A sampler stamps its rows with ``datetime.now()``, so every instant written
here carries microseconds, as the live labels do: a label's width is part of
the payload being measured.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from _dashboard_helpers import RECONCILIATION_SCHEMA
from shared.task_statuses import TaskStatus

from dashboard.data.burndown import BURNDOWN_SCHEMA
from dashboard.data.census import SERIES_KEYS
from dashboard.data.metrics import METRICS_SCHEMA
from dashboard.data.retention import RAW_RETENTION

_SAMPLE_INTERVAL = timedelta(minutes=10)
"""How often the dashboard's loops sample (dashboard/loops.py::_SAMPLE_INTERVAL_SECONDS)."""
_SAMPLES_PER_DAY = int(timedelta(days=1) / _SAMPLE_INTERVAL)


def sample_instants(count: int, now: datetime) -> list[datetime]:
    """The newest *count* sampler ticks before *now*, newest first, at the retention policy's density.

    Every tick survives for ``RAW_RETENTION``; past it one row an hour does
    (dashboard/src/dashboard/data/retention.py::apply_retention).
    """
    newest = (now - timedelta(minutes=1)).replace(microsecond=500)
    raw = int(RAW_RETENTION / _SAMPLE_INTERVAL)
    return [
        newest - _SAMPLE_INTERVAL * i if i < raw else newest - RAW_RETENTION - timedelta(hours=i - raw + 1)
        for i in range(count)
    ]


def _insert(db: Path, schema: str, table: str, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executescript(schema)
        conn.executemany(
            f'INSERT INTO {table} ({", ".join(columns)}) VALUES ({", ".join("?" * len(columns))})',
            rows,
        )


@dataclass(frozen=True)
class BurndownShape:
    """How many measured snapshots each root's burndown series holds, and the cap they were taken under."""

    samples_per_root: int
    concurrency_cap: int


def _snapshot(project_id: str, at: datetime, counts: Mapping[TaskStatus, int], age: int, cap: int) -> dict[str, object]:
    """One measured row, *age* ticks old: the tree as it stood, one task fewer finished per 36 ticks back."""
    finished = min(age // 36, counts.get(TaskStatus.DONE, 0))
    then = {**counts, TaskStatus.DONE: counts.get(TaskStatus.DONE, 0) - finished}
    then[TaskStatus.PENDING] = counts.get(TaskStatus.PENDING, 0) + finished
    in_progress = then.get(TaskStatus.IN_PROGRESS, 0)
    return {
        'project_id': project_id, 'ts': at.isoformat(),
        **{column: then.get(member, 0) for member, column in SERIES_KEYS.items()},
        'in_progress_live': in_progress, 'in_progress_stranded': 0, 'in_progress_rows': in_progress,
        'concurrency_cap': cap, 'state': 'value', 'reason': None,
    }


def seed_burndown_history(
    burndown_db: Path, trees: Mapping[str, Mapping[TaskStatus, int]], shape: BurndownShape, *, now: datetime,
) -> None:
    """``samples_per_root`` measured snapshots of every tree in *trees*, keyed by root path.

    One collector writes every root into the primary root's store at one
    instant per tick, as ``burndown.py::collect_snapshot`` does.
    """
    instants = sample_instants(shape.samples_per_root, now)
    rows = [
        _snapshot(root, at, counts, age, shape.concurrency_cap)
        for root, counts in trees.items() for age, at in enumerate(instants)
    ]
    columns = list(rows[0])
    _insert(burndown_db, BURNDOWN_SCHEMA, 'snapshots', columns, [[row[c] for c in columns] for row in rows])


@dataclass(frozen=True)
class MemoryShape:
    """What fused-memory reports of itself: store totals, the projects it holds memory for, its WAL stores.

    Nearly all of the memory lives in the first project; the rest are the
    probe namespaces a live server accumulates, holding one node or none.
    """

    projects: int
    project_id_len: int
    graphiti_nodes: int
    mem0_memories: int
    wal_stores: tuple[str, ...]

    def project_counts(self) -> dict[str, dict[str, int]]:
        counts = {}
        for n in range(self.projects):
            head = f'fleet_memory_{n:03d}_'
            pid = head + 'x' * (self.project_id_len - len(head))
            nodes, memories = (n % 2, 0) if n else (self.graphiti_nodes, self.mem0_memories)
            counts[pid] = {'graphiti_nodes': nodes, 'mem0_memories': memories}
        return counts


@dataclass(frozen=True)
class ReconShape:
    """The reconciliation store: runs on record, events waiting in the buffer, agents writing."""

    runs: int
    buffered_events: int
    agents: int


def _agent_ids(shape: ReconShape) -> list[str]:
    return [f'claude-task-{1000 + 37 * n}-implementer' for n in range(shape.agents)]


def seed_metrics_history(
    metrics_db: Path,
    *,
    memory: MemoryShape,
    recon: ReconShape,
    running_roots: Sequence[str],
    merge_depth: int,
    now: datetime,
) -> None:
    """A day of the metrics loop's samples: store sizes, queue depth, recon, orchestrators, merges, the curator.

    The curator was capped and idle all day, as the live one was.
    """
    ticks = [at.isoformat() for at in sample_instants(_SAMPLES_PER_DAY, now)]
    project_counts = memory.project_counts()
    tables: list[tuple[str, tuple[str, ...], list[tuple[object, ...]]]] = [
        ('memory_snapshots', ('ts', 'project_id', 'graphiti_nodes', 'mem0_memories'), [
            (ts, pid, c['graphiti_nodes'], c['mem0_memories']) for ts in ticks for pid, c in project_counts.items()
        ]),
        ('queue_snapshots', ('ts', 'pending', 'retry', 'dead'), [
            (ts, None if i % 4 == 1 else i % 3, 0, 1) for i, ts in enumerate(ticks)
        ]),
        ('recon_snapshots', ('ts', 'buffered_count', 'active_agents'), [
            (ts, recon.buffered_events, recon.agents) for ts in ticks
        ]),
        ('orchestrator_snapshots', ('ts', 'project_id', 'running_count'), [
            (ts, root, 1) for ts in ticks for root in running_roots
        ]),
        ('merge_snapshots', ('ts', 'project_id', 'active_count'), [
            (ts, root, merge_depth) for ts in ticks for root in running_roots
        ]),
        ('curator_snapshots', ('ts', 'pending_total', 'capped_now', 'p50_active_ms', 'p90_active_ms', 'p99_active_ms'), [
            (ts, 0, 1, 0, 0, 0) for ts in ticks
        ]),
    ]
    for table, columns, rows in tables:
        _insert(metrics_db, METRICS_SCHEMA, table, columns, rows)


def seed_reconciliation(
    recon_db: Path, shape: ReconShape, project_ids: Sequence[str], *, now: datetime,
) -> None:
    """``runs`` targeted runs three minutes apart, one watermark per project, the buffer and its writers."""
    newest = (now - timedelta(minutes=1)).replace(microsecond=500)
    runs = [
        (
            str(uuid.UUID(int=n + 1)), project_ids[n % len(project_ids)], 'full' if n % 10 == 0 else 'targeted',
            f'task_done:{1000 + n}', (newest - timedelta(minutes=3 * n)).isoformat(),
            (newest - timedelta(minutes=3 * n) + timedelta(seconds=2.368315)).isoformat(), 1 + n % 5, 'completed',
        )
        for n in range(shape.runs)
    ]
    _insert(recon_db, RECONCILIATION_SCHEMA, 'runs', (
        'id', 'project_id', 'run_type', 'trigger_reason', 'started_at', 'completed_at', 'events_processed', 'status',
    ), runs)
    stamp = (newest - timedelta(hours=3)).isoformat()
    _insert(recon_db, RECONCILIATION_SCHEMA, 'watermarks', (
        'project_id', 'last_full_run_id', 'last_full_run_completed', 'last_episode_timestamp',
        'last_memory_timestamp', 'last_task_change_timestamp',
    ), [(pid, runs[0][0], stamp, stamp, stamp, stamp) for pid in project_ids])
    agents = _agent_ids(shape)
    _insert(recon_db, RECONCILIATION_SCHEMA, 'burst_state', (
        'agent_id', 'state', 'last_write_at', 'burst_started_at',
    ), [(agent, 'idle', (newest - timedelta(minutes=n + 1)).isoformat(), None) for n, agent in enumerate(agents)])
    _insert(recon_db, RECONCILIATION_SCHEMA, 'event_buffer', (
        'id', 'project_id', 'event_type', 'event_source', 'agent_id', 'timestamp', 'status',
    ), [
        (f'evt-{n}', project_ids[n % len(project_ids)], 'task_status_changed', 'taskmaster',
         agents[n % len(agents)], (newest - timedelta(hours=3) + timedelta(seconds=27 * n)).isoformat(), 'buffered')
        for n in range(shape.buffered_events)
    ])
    _insert(recon_db, RECONCILIATION_SCHEMA, 'judge_verdicts', (
        'run_id', 'reviewed_at', 'severity', 'findings', 'action_taken',
    ), [(runs[0][0], (newest - timedelta(minutes=10)).isoformat(), 'ok', '[]', 'none')])


@dataclass(frozen=True)
class MemoryOpsShape:
    """The write journal's operations as (operation, kind), most frequent first."""

    operations: tuple[tuple[str, str], ...]


_WRITE_OPS_DDL = """\
CREATE TABLE IF NOT EXISTS write_ops (
    id         TEXT PRIMARY KEY,
    operation  TEXT,
    project_id TEXT,
    agent_id   TEXT,
    kind       TEXT NOT NULL DEFAULT 'write',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wo_created ON write_ops(created_at);
"""
"""The columns and index the ops read touches of fused-memory's write_ops
(fused-memory/src/fused_memory/services/write_journal.py::SCHEMA_SQL); the dashboard does not depend on that package."""


def seed_write_journal(journal_db: Path, shape: MemoryOpsShape, *, now: datetime) -> None:
    """Every hour of the last day: the k-th most frequent operation journalled ``len(operations) - k`` times."""
    newest_hour = now.replace(minute=0, second=0, microsecond=0)
    rows = [
        (f'op-{hour}-{rank}-{n}', operation, 'dark_factory', 'fleet-agent', kind,
         (newest_hour - timedelta(hours=hour) + timedelta(seconds=7 * n + 1)).isoformat())
        for hour in range(24)
        for rank, (operation, kind) in enumerate(shape.operations)
        for n in range(len(shape.operations) - rank)
    ]
    _insert(journal_db, _WRITE_OPS_DDL, 'write_ops', (
        'id', 'operation', 'project_id', 'agent_id', 'kind', 'created_at',
    ), rows)
