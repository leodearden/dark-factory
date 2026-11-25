"""Builders for the verify-host trial report's tests: runs.db rows and load samples.

Each event builder returns one event dict for the conftest ``make_runs_db``
factory, its payload shaped like the live row of that type. Timestamps are
taken as aware datetimes and stored as aware UTC ISO strings, as the live
stores hold them. Samples are written by ``MetricSpec.name`` only, so no
metric name is spelled here. Project roots and samples stores come from
``resource_report_fixtures``.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import _checkout as _checkout
from resource_report_fixtures import HOLD_SAMPLE, project_root, samples_db
from sampler.metrics import MetricSpec

__all__ = [
    'admission_plan',
    'at_hours',
    'cap_hit',
    'dispatch_deferred',
    'insert_rows',
    'invocation',
    'lifecycle',
    'merge_finalized',
    'merge_verify',
    'project_root',
    'samples_db',
    'scheduler_tick',
    'task_completed',
    'task_result',
    'task_started',
    'write_samples',
]

RUN_ID = 'run-1'


def _iso(at: datetime) -> str:
    if at.tzinfo is None:
        raise ValueError(f'fixture timestamps are aware UTC, got naive {at!r}')
    return at.isoformat()


def _event(at: datetime, event_type: str, data: Mapping[str, Any], **columns: Any) -> dict[str, Any]:
    return {'timestamp': _iso(at), 'event_type': event_type, 'data': dict(data), **columns}


def merge_verify(
    end: datetime, runner: str, duration_s: float, task_id: str, *, passed: bool = True
) -> dict[str, Any]:
    """A ``merge_verify`` row: its timestamp is the END, its wall in ``data.duration_ms``."""
    data = {'runner': runner, 'passed': passed, 'duration_ms': round(duration_s * 1000), 'attempt': 0}
    return _event(end, 'merge_verify', data, task_id=task_id, run_id=RUN_ID)


def lifecycle(at: datetime, kind: str, host: str) -> dict[str, Any]:
    """A ``verify_host_admitted`` or ``verify_host_retired`` row carrying ``host`` (PRD §6.5)."""
    if kind not in ('verify_host_admitted', 'verify_host_retired'):
        raise ValueError(f'not a host lifecycle event: {kind!r}')
    return _event(at, kind, {'host': host}, run_id=RUN_ID)


def merge_finalized(at: datetime, task_id: str, state: str) -> dict[str, Any]:
    data = {'request_id': f'mr-{task_id}', 'branch': task_id, 'state': state, 'reason': None}
    return _event(at, 'merge_finalized', data, task_id=task_id, phase='merge', run_id=RUN_ID)


def task_started(at: datetime, task_id: str, run_id: str = RUN_ID) -> dict[str, Any]:
    return _event(at, 'task_started', {'title': f'task {task_id}'}, task_id=task_id, run_id=run_id)


def task_completed(
    at: datetime, task_id: str, outcome: str, reason: str = '', run_id: str = RUN_ID
) -> dict[str, Any]:
    data = {'outcome': outcome, 'agent_invocations': 1, 'reason': reason, 'block_phase': ''}
    return _event(at, 'task_completed', data, task_id=task_id, run_id=run_id)


def scheduler_tick(at: datetime, run_id: str = RUN_ID) -> dict[str, Any]:
    """The row a scheduler tick writes at its END."""
    data = {'duration_ms': 60_000, 'phases': {}, 'terminal_phase': 'select_scored', 'scope': 'any'}
    return _event(at, 'scheduler_tick', data, run_id=run_id)


def dispatch_deferred(at: datetime, metric: str, run_id: str = RUN_ID) -> dict[str, Any]:
    """A dispatch held by the shared.psi gate arm *metric*, with the live PSI sample shape."""
    data = {'metric': metric, 'in_flight': 18, 'floor': 3, 'sample': asdict(HOLD_SAMPLE)}
    return _event(at, 'dispatch_deferred', data, task_id='7919', run_id=run_id)


def admission_plan(
    task_id: str | None,
    role: str,
    arrival: datetime,
    completed: datetime,
    legs: Sequence[tuple[datetime, float]] = (),
) -> dict[str, Any]:
    """A ``verify_admission_plan`` row written at *completed*; each leg is ``(granted_at, held_secs)``."""
    data = {
        'plan_id': f'plan-{task_id}-{_iso(arrival)}',
        'task_id': task_id,
        'role': role,
        'outcome': 'returned',
        'plan_arrival_at': _iso(arrival),
        'completed_at': _iso(completed),
        'legs': [
            {
                'arrived_at': _iso(arrival),
                'granted_at': _iso(granted_at),
                'waited_secs': (granted_at - arrival).total_seconds(),
                'slot_held': True,
                'held_secs': held_secs,
                'rc': 0,
            }
            for granted_at, held_secs in legs
        ],
    }
    return _event(completed, 'verify_admission_plan', data, task_id=task_id, phase='verify', role=role, run_id=RUN_ID)


# ---------------------------------------------------------------------------
# Rows of the non-event tables
# ---------------------------------------------------------------------------

_TABLE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    'task_results': {'run_id': RUN_ID, 'project_id': 'dark_factory'},
    'invocations': {'run_id': RUN_ID, 'project_id': 'dark_factory', 'model': 'opus', 'role': 'implementer'},
    'account_events': {'project_id': 'dark_factory', 'run_id': RUN_ID},
}


def task_result(task_id: str, outcome: str, completed_at: datetime | None, run_id: str = RUN_ID) -> dict[str, Any]:
    return {
        'run_id': run_id,
        'task_id': task_id,
        'outcome': outcome,
        'completed_at': None if completed_at is None else _iso(completed_at),
    }


def invocation(completed_at: datetime, account: str, cost_usd: float) -> dict[str, Any]:
    stamp = _iso(completed_at)
    return {'account_name': account, 'cost_usd': cost_usd, 'started_at': stamp, 'completed_at': stamp}


def cap_hit(at: datetime, account: str, resets_at: datetime | None) -> dict[str, Any]:
    """An ``account_events`` cap_hit; *resets_at* None writes the details without one."""
    details: dict[str, Any] = {'reason': "You've hit your weekly limit"}
    if resets_at is not None:
        details['resets_at'] = _iso(resets_at)
    return {'account_name': account, 'event_type': 'cap_hit', 'details': details, 'created_at': _iso(at)}


def _cell(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, dict | list) else value


def insert_rows(db_path: Path, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Insert *rows* into *table* by explicit column, after that table's defaults."""
    defaults = _TABLE_DEFAULTS[table]
    conn = sqlite3.connect(db_path)
    try:
        for row in rows:
            full = {**defaults, **row}
            columns = ', '.join(full)
            marks = ', '.join('?' for _ in full)
            conn.execute(
                f'INSERT INTO {table} ({columns}) VALUES ({marks})',
                tuple(_cell(value) for value in full.values()),
            )
        conn.commit()
    finally:
        conn.close()


def write_samples(db: Path, rows: Sequence[tuple[int, MetricSpec, float]]) -> None:
    """Host-level samples ``(ts, spec, value)``, each written under its spec's name."""
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            'INSERT INTO samples (ts, metric, value, window_mean, window_max, leaf)'
            ' VALUES (?, ?, ?, NULL, NULL, NULL)',
            [(ts, spec.name, float(value)) for ts, spec, value in rows],
        )
        conn.commit()
    finally:
        conn.close()


def at_hours(base: datetime, hours: float) -> datetime:
    """*base* plus *hours*, for laying rows along a fixture timeline."""
    return base + timedelta(hours=hours)
