"""Non-fixture test helpers for dashboard tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import pytest


def live_aiosqlite_worker_threads() -> list[threading.Thread]:
    """Return every currently-alive aiosqlite worker thread in this process.

    ``aiosqlite.Connection.__init__`` builds its worker as
    ``Thread(target=_connection_worker_thread, args=(self._tx,))``, so a live
    worker is identified by ``thread._target.__name__ ==
    '_connection_worker_thread'``.  The thread's *name* is not usable: it is the
    generic auto-assigned ``Thread-N (_connection_worker_thread)`` on CPython
    3.10+ but plain ``Thread-N`` on older/alternate runtimes, and nothing in
    aiosqlite pins it.

    PRIVATE-ATTRIBUTE PIN.  ``Thread._target`` is a CPython implementation
    detail and ``_connection_worker_thread`` is an aiosqlite private module
    function.  Both are verified against **aiosqlite >=0.22.x** — bump and
    re-verify this pin if either moves, exactly as the ``_connection`` /
    ``_running`` / ``_thread`` pin in ``test_db.py`` documents.  The
    ``getattr(..., None)`` guards below mean a moved attribute degrades to "no
    workers found" rather than an ``AttributeError`` mid-assertion; the
    ``_PIN_OK`` assertion at import time is what makes such a move LOUD.
    """
    live: list[threading.Thread] = []
    for thread in threading.enumerate():
        target = getattr(thread, '_target', None)
        if getattr(target, '__name__', None) == '_connection_worker_thread':
            live.append(thread)
    return live


# Import-time guard for the private-attribute pin above: if aiosqlite ever
# renames `_connection_worker_thread`, `live_aiosqlite_worker_threads()` would
# silently return [] and every leak assertion built on it would pass vacuously.
# Fail LOUDLY at collection instead (INV: no-silent-fail-soft).
assert hasattr(aiosqlite.core, '_connection_worker_thread'), (
    'aiosqlite.core._connection_worker_thread not found — the worker-thread '
    'name pin in live_aiosqlite_worker_threads() must be updated for this '
    f'aiosqlite version ({aiosqlite.__version__})'
)


def apply_isolated_env(mp: pytest.MonkeyPatch, root: Path) -> None:
    """Point every DashboardConfig-derived path at *root* instead of the live checkout.

    Sets ``DASHBOARD_PROJECT_ROOT``, which redirects ``burndown_db``,
    ``metrics_db``, ``runs_db``, ``escalations_dir``, ``memory_evals_dir`` and
    ``load_samples_db`` (task 3503).  The dashboard app's ``lifespan()`` opens
    ``burndown_db`` and ``metrics_db`` as **writable WAL** stores, so without
    this every ``TestClient(app)`` in the suite wrote into the operator's live
    ``data/burndown/``.

    Also DELETES ``DASHBOARD_KNOWN_PROJECT_ROOTS``.  ``project_root`` is not the
    only root the app fans out over: ``from_env()`` reads that var into
    ``known_project_roots``, and ``_project_scoped_dbs`` / ``_cost_dbs`` /
    ``_performance_resources`` (dashboard/src/dashboard/app.py) plus
    ``data/burndown.py`` then ``DbPool.get(root / 'data/orchestrator/runs.db')``
    for EVERY entry.  Left ambient, that reopens live WAL databases in whatever
    checkouts the operator registered — the same task-3466
    ``SQLITE_READONLY_RECOVERY`` class this helper exists to close.  Ambient
    presence is live, not hypothetical: it is a shared registry var read across
    fused-memory, and the installed dashboard systemd unit sets it.

    Also DELETES ``RECONCILIATION_DATA_DIR`` and ``QUEUE_DATA_DIR``.  Those are
    read straight from ``os.environ`` by ``DashboardConfig._runtime_data_dir``
    and WIN over ``project_root`` for ``reconciliation_db``, ``tickets_db``,
    ``write_queue_db``, ``write_journal_db`` and
    ``reconciliation_escalations_dir`` — so setting only
    ``DASHBOARD_PROJECT_ROOT`` would leave them pointed wherever the ambient
    environment says.  ``reconciliation_db`` and ``tickets_db`` are exactly the
    two read-only ``DbPool.get()`` opens in ``_metrics_loop`` that produced the
    task-3466 ``SQLITE_READONLY_RECOVERY``, so that gap would leave the
    incident's own trigger path un-isolated.  Ambient presence is live, not
    hypothetical: the orchestrator's managed fused-memory spawn
    (``orchestrator/src/orchestrator/mcp_lifecycle.py``) injects both.

    DELETE rather than redirect, deliberately.  For the two runtime dirs,
    deleting makes the config fall back to ``project_root``-relative paths,
    which keeps ``test_scaffold.py``'s
    ``TestConfigDefaults.test_config_derived_paths`` assertions valid — and in
    fact makes them hermetic, since today they silently depend on the ambient
    environment happening to have these unset.  Redirecting under *root* would
    instead break them.  For ``DASHBOARD_KNOWN_PROJECT_ROOTS``, deleting yields
    the empty list, i.e. exactly one root to fan out over; there is no temp
    path to redirect it to that would be more isolated than none.

    A plain function rather than a fixture so the env contract is directly
    unit-testable against a simulated operator environment — a session-scoped
    autouse fixture cannot be re-run from inside a test.
    """
    mp.setenv('DASHBOARD_PROJECT_ROOT', str(root))
    mp.delenv('DASHBOARD_KNOWN_PROJECT_ROOTS', raising=False)
    mp.delenv('RECONCILIATION_DATA_DIR', raising=False)
    mp.delenv('QUEUE_DATA_DIR', raising=False)


RECONCILIATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    project_id TEXT PRIMARY KEY,
    last_full_run_id TEXT,
    last_full_run_completed TEXT,
    last_episode_timestamp TEXT,
    last_memory_timestamp TEXT,
    last_task_change_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_processed INTEGER DEFAULT 0,
    stage_reports TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running'
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    target_system TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    reasoning TEXT DEFAULT '',
    evidence TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal_entries(run_id);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id TEXT PRIMARY KEY,
    reviewed_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    findings TEXT DEFAULT '[]',
    action_taken TEXT DEFAULT 'none'
);
CREATE INDEX IF NOT EXISTS idx_verdicts_reviewed ON judge_verdicts(reviewed_at);

CREATE TABLE IF NOT EXISTS event_buffer (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    agent_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'buffered'
);
CREATE INDEX IF NOT EXISTS idx_eb_project_status ON event_buffer(project_id, status);
CREATE INDEX IF NOT EXISTS idx_eb_agent_timestamp ON event_buffer(agent_id, timestamp)
    WHERE agent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reconciliation_locks (
    project_id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burst_state (
    agent_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'idle',
    last_write_at TEXT NOT NULL,
    burst_started_at TEXT
);

CREATE TABLE IF NOT EXISTS chunk_boundaries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT,
    events_count INTEGER,
    status TEXT DEFAULT 'processing',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunk_boundaries(project_id);

CREATE TABLE IF NOT EXISTS run_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target TEXT NOT NULL,
    operation TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    causation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ra_run ON run_actions(run_id);
"""

assert 'started_at TEXT NOT NULL' in RECONCILIATION_SCHEMA, (
    "RECONCILIATION_SCHEMA must contain 'started_at TEXT NOT NULL'; "
    "update RELAXED_RECONCILIATION_SCHEMA derivation if the schema changes."
)

RELAXED_RECONCILIATION_SCHEMA = RECONCILIATION_SCHEMA.replace(
    'started_at TEXT NOT NULL', 'started_at TEXT'
)


@asynccontextmanager
async def make_recon_db(
    tmp_path: Path,
    inserts: Sequence[str | tuple[str, Any]],
    *,
    name: str = 'test.db',
    schema: str | None = None,
):
    """Async context manager that creates a temporary SQLite reconciliation DB.

    Creates a DB at ``tmp_path / name``, applies ``schema`` (defaults to
    ``RECONCILIATION_SCHEMA``), executes each statement in ``inserts``, then
    yields an :class:`aiosqlite.Connection` with ``row_factory`` set to
    :class:`aiosqlite.Row`.  The connection is closed on context exit.
    """
    if schema is None:
        schema = RECONCILIATION_SCHEMA

    db_path = tmp_path / name
    sync_conn = sqlite3.connect(str(db_path))
    sync_conn.executescript(schema)
    for stmt in inserts:
        if isinstance(stmt, str):
            sync_conn.execute(stmt)
        else:
            sql, params = stmt
            sync_conn.execute(sql, params)
    sync_conn.commit()
    sync_conn.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn
