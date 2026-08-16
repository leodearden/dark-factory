"""Shared hold-history event-trace fixtures (task 3822 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task ζ).

Lives in ``_hold_history_fixtures.py`` (not ``conftest.py``) to follow the
``_recording_event_store.py`` / ``_orch_helpers.py`` pattern: importable from
any test file without triggering ``sys.modules['conftest']`` collisions across
subprojects.

WHAT THIS BUILDS
================
One deterministic, ordered row list shaped EXACTLY like the rows
``EventStore.fetch_events_by_type_all_runs`` returns — ``id``, ``timestamp``
(ISO-8601 UTC), ``run_id``, ``task_id``, ``event_type``, ``phase``, ``role``,
``data`` (already a dict), ``cost_usd``, ``duration_ms``.

Every timestamp is a whole-second offset from a fixed ``BASE_TS``, so every
expected duration and median below is hand-computed and asserted as a literal.
This module IS the oracle: no expected value here is read back out of the
implementation under test.

The trace covers all four residue classes named in
``plans/evidence/scheduler-scoring-2026-08-06/PARKING_MODEL_REPORT.md:100-104``:
CLEAN pair, ORPHAN-RELEASE, DOUBLE-ACQUIRE, and STUCK-AT-ERA-END (both era
sources: a ``service_restart`` row, and a ``run_id`` transition).

THE CANONICAL TRACE (offsets in seconds from BASE_TS)
=====================================================

  id  t     run    task  event            modules                     effect
  --  ----  -----  ----  ---------------  --------------------------  ---------------------------------------
   1     0  run-a  T1    lock_acquired    [orchestrator/src]          open
   2    60  run-a  T1    lock_released    [orchestrator/src]          CLEAN  orchestrator/src  60
   3   100  run-a  T2    lock_acquired    [orchestrator/src,          open x2
                                           shared/src]
   4   220  run-a  T2    lock_released    [orchestrator/src]          PARTIAL (reason=plan_refinement):
                                          reason=plan_refinement       closes orchestrator/src 120,
                                                                       LEAVES shared/src open
   5   300  run-a  T2    lock_released    [shared/src]                CLEAN  shared/src        200
   6   350  run-a  T3    lock_released    [orchestrator/src]          ORPHAN-RELEASE -> no span at all
   7   400  run-a  T4    lock_acquired    [orchestrator/src]          open
   8   580  run-a  T4    lock_acquired    [orchestrator/src]          DOUBLE-ACQUIRE: force-close prior span
                                                                       at 580 -> orchestrator/src 180
                                                                       (truncated), reopen at 580
   9   700  run-a  T4    lock_released    [orchestrator/src]          CLEAN  orchestrator/src 120
  10   800  run-a  T5    lock_acquired    [fused-memory/src]          open
  11  1000  run-a  T9x   service_restart  (n/a)                       ERA BOUNDARY -> closes T5 at 1000 ->
                                                                       fused-memory/src 200 (truncated).
                                                                       NOTE its task_id is a *trigger* task,
                                                                       NOT a lock holder — must be ignored.
  12  1100  run-a  T6    lock_acquired    [fused-memory/src]          open
  13  1200  run-a  T6    lock_released    [fused-memory/src]          CLEAN  fused-memory/src  100
  14  1300  run-a  T7    lock_acquired    [shared/src]                open — never released in run-a
  15  1360  run-a  T8    lock_acquired    [orchestrator/src]          open
  16  1400  run-a  T8    lock_released    [orchestrator/src]          CLEAN  orchestrator/src  40
                                                                       (last row of run-a, t=1400)
  17  1500  run-b  T9    lock_acquired    [shared/src]                RUN TRANSITION run-a -> run-b:
                                                                       closes T7 at the PREVIOUS row's
                                                                       timestamp (1400) -> shared/src 100
                                                                       (truncated).  Then open T9.
  18  1560  run-b  T9    lock_released    [shared/src]                CLEAN  shared/src         60
  19  1600  run-b  T10   lock_acquired    [orchestrator/src]          open at END OF STREAM -> DROPPED
  20  1650  run-b  T11   lock_acquired    [fused-memory/src]          open
  21  1950  run-b  T11   lock_released    [fused-memory/src]          CLEAN  fused-memory/src   300

HAND-COMPUTED EXPECTATIONS
==========================
  orchestrator/src : [60, 120, 180, 120, 40] -> sorted [40, 60, 120, 120, 180] -> median 120.0
  shared/src       : [200, 100, 60]          -> sorted [60, 100, 200]          -> median 100.0
  fused-memory/src : [200, 100, 300]         -> sorted [100, 200, 300]         -> median 200.0

  pooled over all three (n=11):
    sorted [40, 60, 60, 100, 100, 120, 120, 180, 200, 200, 300] -> median 120.0
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

# Fixed epoch anchor.  Every fixture timestamp is BASE_TS + a whole-second
# offset, so durations are exact integers and the medians above are literals.
BASE_TS = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)

RUN_A = 'run-a'
RUN_B = 'run-b'

#: task_id carried by the fixture's ``service_restart`` row.  The real emit
#: site (service_restart.py:747-759) stamps a *trigger* task id, which is NOT
#: a lock holder — a consumer that mistook it for one would open or close the
#: wrong span, so the fixture deliberately uses an id that holds nothing.
SERVICE_RESTART_TRIGGER_TASK = 'T9x'


def ts(offset_secs: float) -> str:
    """ISO-8601 UTC timestamp ``offset_secs`` after :data:`BASE_TS`."""
    return datetime.fromtimestamp(BASE_TS.timestamp() + offset_secs, tz=UTC).isoformat()


def row(
    row_id: int,
    offset_secs: float,
    event_type: str,
    *,
    task_id: str | None,
    run_id: str = RUN_A,
    data: dict[str, Any] | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """One event row shaped exactly like ``fetch_events_by_type_all_runs`` returns."""
    return {
        'id': row_id,
        'timestamp': ts(offset_secs),
        'run_id': run_id,
        'task_id': task_id,
        'event_type': event_type,
        'phase': phase,
        'role': None,
        'data': dict(data or {}),
        'cost_usd': None,
        'duration_ms': None,
    }


def acquire(
    row_id: int,
    offset_secs: float,
    task_id: str,
    modules: list[str],
    *,
    run_id: str = RUN_A,
    priority: str = 'medium',
) -> dict[str, Any]:
    """A ``lock_acquired`` row — payload mirrors scheduler.py:6313-6318 / :6426-6431."""
    return row(
        row_id, offset_secs, 'lock_acquired',
        task_id=task_id, run_id=run_id,
        data={'modules': list(modules), 'priority': priority},
    )


def release(
    row_id: int,
    offset_secs: float,
    task_id: str,
    modules: list[str],
    *,
    run_id: str = RUN_A,
    reason: str | None = None,
) -> dict[str, Any]:
    """A ``lock_released`` row.

    ``reason='plan_refinement'`` reproduces the PARTIAL release emitted by
    ``release_subset`` (scheduler.py:6954-6959), which names only the modules
    it narrows away; the full release (scheduler.py:7059-7064) carries
    ``{'modules': ...}`` alone.
    """
    data: dict[str, Any] = {'modules': list(modules)}
    if reason is not None:
        data['reason'] = reason
    return row(row_id, offset_secs, 'lock_released', task_id=task_id, run_id=run_id, data=data)


def service_restart(
    row_id: int,
    offset_secs: float,
    *,
    task_id: str | None = SERVICE_RESTART_TRIGGER_TASK,
    run_id: str = RUN_A,
    service: str = 'fused-memory',
) -> dict[str, Any]:
    """A ``service_restart`` row — payload mirrors service_restart.py:747-759."""
    return row(
        row_id, offset_secs, 'service_restart',
        task_id=task_id, run_id=run_id, phase='service_restart',
        data={
            'service': service,
            'script': f'/opt/restart-{service}.sh',
            'trigger_task_ids': [task_id] if task_id else [],
            'merge_shas': [],
            'reason': f"post_merge_{service.replace('-', '_')}_code_change",
        },
    )


def build_trace() -> list[dict[str, Any]]:
    """The canonical 21-row trace documented in this module's docstring."""
    return [
        acquire(1, 0, 'T1', ['orchestrator/src']),
        release(2, 60, 'T1', ['orchestrator/src']),
        acquire(3, 100, 'T2', ['orchestrator/src', 'shared/src']),
        release(4, 220, 'T2', ['orchestrator/src'], reason='plan_refinement'),
        release(5, 300, 'T2', ['shared/src']),
        release(6, 350, 'T3', ['orchestrator/src']),          # ORPHAN-RELEASE
        acquire(7, 400, 'T4', ['orchestrator/src']),
        acquire(8, 580, 'T4', ['orchestrator/src']),          # DOUBLE-ACQUIRE
        release(9, 700, 'T4', ['orchestrator/src']),
        acquire(10, 800, 'T5', ['fused-memory/src']),
        service_restart(11, 1000),                            # ERA BOUNDARY (restart)
        acquire(12, 1100, 'T6', ['fused-memory/src']),
        release(13, 1200, 'T6', ['fused-memory/src']),
        acquire(14, 1300, 'T7', ['shared/src']),              # open across the run change
        acquire(15, 1360, 'T8', ['orchestrator/src']),
        release(16, 1400, 'T8', ['orchestrator/src']),        # last row of run-a
        acquire(17, 1500, 'T9', ['shared/src'], run_id=RUN_B),  # ERA BOUNDARY (run change)
        release(18, 1560, 'T9', ['shared/src'], run_id=RUN_B),
        acquire(19, 1600, 'T10', ['orchestrator/src'], run_id=RUN_B),  # open at EOS -> DROPPED
        acquire(20, 1650, 'T11', ['fused-memory/src'], run_id=RUN_B),
        release(21, 1950, 'T11', ['fused-memory/src'], run_id=RUN_B),
    ]


#: Per-module durations (seconds) yielded by the canonical trace, in the order
#: the spans close.  Hand-computed from the timeline above — NOT read back out
#: of the implementation under test.
EXPECTED_SAMPLES: dict[str, list[float]] = {
    'orchestrator/src': [60.0, 120.0, 180.0, 120.0, 40.0],
    'shared/src': [200.0, 100.0, 60.0],
    'fused-memory/src': [200.0, 100.0, 300.0],
}

#: Hand-computed medians of :data:`EXPECTED_SAMPLES`.
EXPECTED_MEDIANS: dict[str, float] = {
    'orchestrator/src': 120.0,
    'shared/src': 100.0,
    'fused-memory/src': 200.0,
}

#: Median of every sample pooled across all three modules (n=11).
EXPECTED_POOLED_MEDIAN: float = 120.0

#: Total spans the canonical trace yields (5 + 3 + 3).  The orphan release and
#: the end-of-stream-open span contribute nothing.
EXPECTED_SPAN_COUNT: int = 11

#: (task_id, module) pairs whose span is era- or double-acquire-truncated.
EXPECTED_TRUNCATED_SPANS: set[tuple[str, str]] = {
    ('T4', 'orchestrator/src'),   # double-acquire force-close at id=8
    ('T5', 'fused-memory/src'),   # service_restart boundary at id=11
    ('T7', 'shared/src'),         # run-a -> run-b transition at id=17
}


_INSERT_SQL = (
    'INSERT INTO events '
    '(id, timestamp, run_id, task_id, event_type, phase, role, data, cost_usd, duration_ms) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)


def write_trace(event_store, rows: list[dict[str, Any]] | None = None) -> int:
    """Write *rows* (default: :func:`build_trace`) into a REAL ``EventStore``.

    ``EventStore.emit`` stamps its own ``timestamp`` (wall clock) and
    ``run_id`` (the store's own), so it cannot reproduce a fixed multi-run
    trace — this writes the rows verbatim through the store's own sqlite file
    instead, ids and all, so ``fetch_events_by_type_all_runs`` reads back
    exactly what :func:`build_trace` describes.

    ``_RecordingEventStore`` cannot stand in for this: it implements only
    ``emit()`` and has no fetch API at all.

    Returns the number of rows written.
    """
    payload = list(rows if rows is not None else build_trace())
    conn = sqlite3.connect(str(event_store.db_path))
    try:
        conn.executemany(
            _INSERT_SQL,
            [
                (
                    r['id'], r['timestamp'], r['run_id'], r['task_id'],
                    r['event_type'], r.get('phase'), r.get('role'),
                    json.dumps(r['data']), r.get('cost_usd'), r.get('duration_ms'),
                )
                for r in payload
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return len(payload)
