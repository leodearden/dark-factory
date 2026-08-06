"""Durability tests for the burndown and metrics SQLite writer connections.

These tests verify that both long-lived writer connections opened inside
``lifespan()`` apply the full Phase-3 durability pragma triad
(``synchronous=FULL``, ``wal_autocheckpoint=100``, ``journal_size_limit=64 MiB``)
mandated by the 2026-05-14 stability directive.

Tests call ``lifespan(app)`` directly as an async context manager so that
pragma assertions can access the live writer connection — per-connection PRAGMAs
do NOT persist to disk and cannot be verified by opening a fresh reader.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from _dashboard_helpers import apply_isolated_env, live_aiosqlite_worker_threads
from fastapi import FastAPI
from shared.async_sqlite_base import CheckpointResult

from dashboard.app import (
    _burndown_loop,
    _BurndownStore,
    _metrics_loop,
    _MetricsStore,
    lifespan,
)
from dashboard.config import DashboardConfig

# ---------------------------------------------------------------------------
# Step-1: burndown store pragma triad
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burndown_store_applies_full_pragma_triad_after_lifespan(
    tmp_path: Path, monkeypatch
):
    """_BurndownStore open() applies the full Phase-3 durability pragma triad.

    Calls lifespan() directly as an async context manager so that the pragma
    assertions can run inside an async scope with access to the live writer
    connection. The five per-connection PRAGMAs cannot be verified by opening a
    fresh reader — they must be checked on the connection that set them.

    A fresh FastAPI instance is created per test to avoid shared app.state
    pollution between test runs. collect_snapshot and collect_metrics_snapshot
    are patched to AsyncMock so the lifespan startup does not hit the network
    or per-project DBs.
    """
    monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', str(tmp_path))
    local_app = FastAPI(lifespan=lifespan)

    with (
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
    ):
        async with lifespan(local_app):
            # local_app.state.burndown_store must exist after lifespan startup.
            assert hasattr(local_app.state, 'burndown_store'), (
                'local_app.state.burndown_store not set after lifespan startup — '
                '_BurndownStore wrapper not yet implemented in lifespan()'
            )
            store = local_app.state.burndown_store
            conn = store.connection  # raises RuntimeError if not opened

            async with conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
            assert row[0] == 'wal', f'journal_mode: expected wal, got {row[0]!r}'

            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 5000, f'busy_timeout: expected 5000, got {row[0]}'

            async with conn.execute('PRAGMA synchronous') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 2, f'synchronous: expected 2 (FULL), got {row[0]}'

            async with conn.execute('PRAGMA wal_autocheckpoint') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 100, f'wal_autocheckpoint: expected 100, got {row[0]}'

            async with conn.execute('PRAGMA journal_size_limit') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 67108864, (
                f'journal_size_limit: expected 67108864, got {row[0]}'
            )

    # After lifespan exits the store must be closed — verified via the public
    # checkpoint() method: it raises RuntimeError('not opened') on a closed store.
    with pytest.raises(RuntimeError, match='not opened'):
        await store.checkpoint()


# ---------------------------------------------------------------------------
# Step-3: metrics store pragma triad
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_store_applies_full_pragma_triad_after_lifespan(
    tmp_path: Path, monkeypatch
):
    """_MetricsStore open() applies the full Phase-3 durability pragma triad.

    Mirrors test_burndown_store_applies_full_pragma_triad_after_lifespan but
    targets local_app.state.metrics_store and the metrics writer connection.
    A fresh FastAPI instance is created per test to prevent shared app.state
    pollution.
    """
    monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', str(tmp_path))
    local_app = FastAPI(lifespan=lifespan)

    with (
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
    ):
        async with lifespan(local_app):
            # local_app.state.metrics_store must exist after lifespan startup.
            assert hasattr(local_app.state, 'metrics_store'), (
                'local_app.state.metrics_store not set after lifespan startup — '
                '_MetricsStore wrapper not yet implemented in lifespan()'
            )
            store = local_app.state.metrics_store
            conn = store.connection  # raises RuntimeError if not opened

            async with conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
            assert row[0] == 'wal', f'journal_mode: expected wal, got {row[0]!r}'

            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 5000, f'busy_timeout: expected 5000, got {row[0]}'

            async with conn.execute('PRAGMA synchronous') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 2, f'synchronous: expected 2 (FULL), got {row[0]}'

            async with conn.execute('PRAGMA wal_autocheckpoint') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 100, f'wal_autocheckpoint: expected 100, got {row[0]}'

            async with conn.execute('PRAGMA journal_size_limit') as cur:
                row = await cur.fetchone()
            assert int(row[0]) == 67108864, (
                f'journal_size_limit: expected 67108864, got {row[0]}'
            )

    # After lifespan exits the store must be closed — verified via checkpoint().
    with pytest.raises(RuntimeError, match='not opened'):
        await store.checkpoint()


# ---------------------------------------------------------------------------
# Step-5: burndown loop invokes periodic checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burndown_loop_invokes_periodic_checkpoint(tmp_path: Path):
    """_burndown_loop calls store.checkpoint() on the periodic interval timer.

    Drives _burndown_loop directly (no lifespan), patches _CHECKPOINT_INTERVAL_SECONDS
    to 0 so the checkpoint fires on the first loop body iteration. Waits on the
    checkpoint mock's own asyncio.Event so the assertion is racefree: the event is
    set inside store.checkpoint, meaning the checkpoint has actually fired before we
    cancel the task.
    """
    store = _BurndownStore(tmp_path / 'burndown.db', busy_timeout_ms=5000)
    await store.open()

    # Replace checkpoint with an AsyncMock that sets an event when called.
    checkpoint_called = asyncio.Event()

    async def _checkpoint_side_effect(*args, **kwargs):
        checkpoint_called.set()
        return CheckpointResult(0, 0, 0)

    checkpoint_mock = AsyncMock(side_effect=_checkpoint_side_effect)
    store.checkpoint = checkpoint_mock  # type: ignore[method-assign]

    # Build a minimal config (no network needed — collect_snapshot is patched).
    config = DashboardConfig(project_root=tmp_path)

    # _sleep_to_aligned_tick must yield to the event loop so checkpoint_called.set()
    # (scheduled via loop.call_soon inside asyncio.Event.set) is processed between
    # iterations.  A plain AsyncMock(return_value=None) never suspends, creating a
    # tight synchronous loop that starves asyncio.wait_for of event-loop cycles.
    async def _noop_sleep(*a: object, **kw: object) -> None:
        await asyncio.sleep(0)

    try:
        with (
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(side_effect=_noop_sleep)),
            patch('dashboard.app._CHECKPOINT_INTERVAL_SECONDS', 0),
        ):
            task = asyncio.create_task(
                _burndown_loop(store, config, MagicMock())
            )
            try:
                # Wait until store.checkpoint() is actually called — this is racefree
                # because the event is set inside the checkpoint mock itself.
                await asyncio.wait_for(checkpoint_called.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    assert checkpoint_mock.called, (
        '_burndown_loop did not call store.checkpoint() — periodic checkpoint not yet implemented'
    )


@pytest.mark.asyncio
async def test_burndown_loop_checkpoint_respects_interval_gate(tmp_path: Path):
    """_burndown_loop calls store.checkpoint() at most once per interval window.

    Runs the loop with _CHECKPOINT_INTERVAL_SECONDS=3600 for several iterations
    and asserts checkpoint is called at most once. The first iteration fires because
    time.monotonic() >> 3600s (system uptime); subsequent iterations are suppressed
    by the interval gate (now - last_checkpoint is only milliseconds).
    A regression that called checkpoint() on every iteration would show
    checkpoint_count >> 1, catching the gate logic being bypassed.
    """
    store = _BurndownStore(tmp_path / 'burndown_gate.db', busy_timeout_ms=5000)
    await store.open()

    checkpoint_count = 0

    async def _counting_checkpoint(*args, **kwargs):
        nonlocal checkpoint_count
        checkpoint_count += 1
        return CheckpointResult(0, 0, 0)

    store.checkpoint = AsyncMock(side_effect=_counting_checkpoint)  # type: ignore[method-assign]

    config = DashboardConfig(project_root=tmp_path)

    # Count collect_snapshot calls to know when enough loop-body iterations have run.
    collect_calls = 0
    many_iters_done = asyncio.Event()

    async def _counting_collect(*a: object, **kw: object) -> None:
        nonlocal collect_calls
        collect_calls += 1
        if collect_calls >= 6:  # 1 initial + 5 in-loop body
            many_iters_done.set()

    async def _noop_sleep(*a: object, **kw: object) -> None:
        await asyncio.sleep(0)

    try:
        with (
            patch('dashboard.app.collect_snapshot', new=AsyncMock(side_effect=_counting_collect)),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(side_effect=_noop_sleep)),
            patch('dashboard.app._CHECKPOINT_INTERVAL_SECONDS', 3600),
        ):
            task = asyncio.create_task(_burndown_loop(store, config, MagicMock()))
            try:
                await asyncio.wait_for(many_iters_done.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    # With 3600s interval and 5 in-loop iterations completing in milliseconds,
    # checkpoint fires at most once (first iteration where monotonic() >> 3600s),
    # then the gate suppresses it for the remainder of the test.
    assert checkpoint_count <= 1, (
        f'Expected checkpoint_count <= 1 with 3600s interval over 5 iterations, '
        f'got {checkpoint_count}. Interval gate not working correctly.'
    )


# ---------------------------------------------------------------------------
# Step-7: metrics loop invokes periodic checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_loop_invokes_periodic_checkpoint(tmp_path: Path):
    """_metrics_loop calls store.checkpoint() on the periodic interval timer.

    Drives _metrics_loop directly (no lifespan), patches _CHECKPOINT_INTERVAL_SECONDS
    to 0 so the checkpoint fires on the first loop body iteration. Waits on the
    checkpoint mock's own asyncio.Event so the assertion is racefree: the event is
    set inside store.checkpoint, meaning the checkpoint has actually fired before we
    cancel the task.
    """
    store = _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000)
    await store.open()

    # Replace checkpoint with an AsyncMock that sets an event when called.
    checkpoint_called = asyncio.Event()

    async def _checkpoint_side_effect(*args: object, **kwargs: object) -> CheckpointResult:
        checkpoint_called.set()
        return CheckpointResult(0, 0, 0)

    checkpoint_mock = AsyncMock(side_effect=_checkpoint_side_effect)
    store.checkpoint = checkpoint_mock  # type: ignore[method-assign]

    # Minimal app-state stub — pool.get() returns None because collect_metrics_snapshot
    # is patched and never inspects the connections it receives.
    config = DashboardConfig(project_root=tmp_path)
    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=None)
    mock_app = MagicMock()
    mock_app.state.config = config
    mock_app.state.db = mock_pool
    mock_app.state.http_client = MagicMock()

    # _sleep_to_aligned_tick must yield to the event loop so checkpoint_called.set()
    # (scheduled via loop.call_soon inside asyncio.Event.set) is processed between
    # iterations.  A plain AsyncMock(return_value=None) never suspends, creating a
    # tight synchronous loop that starves asyncio.wait_for of event-loop cycles.
    async def _noop_sleep(*a: object, **kw: object) -> None:
        await asyncio.sleep(0)

    try:
        with (
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(side_effect=_noop_sleep)),
            patch('dashboard.app._CHECKPOINT_INTERVAL_SECONDS', 0),
        ):
            task = asyncio.create_task(_metrics_loop(store, mock_app))
            try:
                # Wait until store.checkpoint() is actually called — this is racefree
                # because the event is set inside the checkpoint mock itself.
                await asyncio.wait_for(checkpoint_called.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    assert checkpoint_mock.called, (
        '_metrics_loop did not call store.checkpoint() — periodic checkpoint not yet implemented'
    )


@pytest.mark.asyncio
async def test_metrics_loop_checkpoint_respects_interval_gate(tmp_path: Path):
    """_metrics_loop calls store.checkpoint() at most once per interval window.

    Mirrors test_burndown_loop_checkpoint_respects_interval_gate for _metrics_loop.
    Runs with _CHECKPOINT_INTERVAL_SECONDS=3600 for several iterations and asserts
    checkpoint is called at most once, verifying the interval gate works correctly.
    """
    store = _MetricsStore(tmp_path / 'metrics_gate.db', busy_timeout_ms=5000)
    await store.open()

    checkpoint_count = 0

    async def _counting_checkpoint(*args: object, **kwargs: object) -> CheckpointResult:
        nonlocal checkpoint_count
        checkpoint_count += 1
        return CheckpointResult(0, 0, 0)

    store.checkpoint = AsyncMock(side_effect=_counting_checkpoint)  # type: ignore[method-assign]

    config = DashboardConfig(project_root=tmp_path)
    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=None)
    mock_app = MagicMock()
    mock_app.state.config = config
    mock_app.state.db = mock_pool
    mock_app.state.http_client = MagicMock()

    collect_calls = 0
    many_iters_done = asyncio.Event()

    async def _counting_collect(*a: object, **kw: object) -> None:
        nonlocal collect_calls
        collect_calls += 1
        if collect_calls >= 6:  # 1 initial + 5 in-loop body
            many_iters_done.set()

    async def _noop_sleep(*a: object, **kw: object) -> None:
        await asyncio.sleep(0)

    try:
        with (
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(side_effect=_counting_collect),
            ),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(side_effect=_noop_sleep)),
            patch('dashboard.app._CHECKPOINT_INTERVAL_SECONDS', 3600),
        ):
            task = asyncio.create_task(_metrics_loop(store, mock_app))
            try:
                await asyncio.wait_for(many_iters_done.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    # With 3600s interval and 5 in-loop iterations completing in milliseconds,
    # checkpoint fires at most once (first iteration where monotonic() >> 3600s),
    # then the gate suppresses it for the remainder of the test.
    assert checkpoint_count <= 1, (
        f'Expected checkpoint_count <= 1 with 3600s interval over 5 iterations, '
        f'got {checkpoint_count}. Interval gate not working correctly.'
    )


# ---------------------------------------------------------------------------
# Task 3466: a DbPool.get() cancelled mid-connect must not orphan a worker thread
# ---------------------------------------------------------------------------

# How long the patched connect holds a landed read-only connection before
# handing it back to DbPool.get().  Only has to outlast the lifespan teardown
# path (a few cancel/await hops), and must stay well INSIDE
# dashboard.data.db._INFLIGHT_DRAIN_TIMEOUT so close_all()'s drain reaps it.
_CONNECT_HOLD_SECONDS = 0.3


def _create_metrics_loop_db_files(config: DashboardConfig) -> list[Path]:
    """Materialise every database ``_metrics_loop`` read-only-opens via DbPool.

    ``DbPool.get()`` short-circuits with ``return None`` when
    ``resolved.exists()`` is False, so without real files on disk the connect
    path is never reached and any leak assertion built on it passes VACUOUSLY.

    Paths are derived from *config* rather than hardcoded so a moved property
    stops covering the connect path LOUDLY (the ``landed`` event below never
    fires → ``asyncio.wait_for`` TimeoutError) instead of silently.
    """
    paths = [
        config.reconciliation_db,  # _metrics_loop._run_once, 1st pool.get
        config.tickets_db,  # _metrics_loop._run_once, 2nd pool.get
        config.project_root / 'data' / 'orchestrator' / 'runs.db',  # _project_scoped_dbs_labeled
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(str(path)).close()
    return paths


def _holding_readonly_connect(
    landed: asyncio.Event, opened: list[aiosqlite.Connection]
):
    """Build an ``aiosqlite.connect`` replacement that stalls DbPool opens.

    Delegates to the real connect FIRST — so the worker thread is provably
    started and the ``Connection`` object provably exists — and only THEN
    sleeps.  Sleeping *before* delegating would be a vacuous test: cancelling
    during that sleep means no thread was ever created, so nothing could leak
    and the guard would pass even on the broken tree.

    Only read-only URI opens (``?mode=ro``, i.e. exactly the ``DbPool.get()``
    opens) are stalled.  ``lifespan()``'s writable ``_BurndownStore`` /
    ``_MetricsStore`` opens go through untouched; stalling those would delay
    startup by the same window and the metrics loop would never reach a
    ``pool.get()`` at all.

    Every stalled connection is appended to *opened*, which the test then holds
    for the duration.  That strong reference is LOAD-BEARING, not bookkeeping:
    ``aiosqlite.Connection.__del__`` (>=0.22.x) calls ``stop()``, so CPython
    refcounting would otherwise reap an abandoned connection the instant its
    last frame unwinds and hide the leak.  Depending on that is exactly the
    silent fail-soft this task removes — ``__del__`` announces itself only as a
    ``ResourceWarning``, it is not guaranteed prompt (or to run at all) off
    CPython, and when it fires *after* the loop has closed its ``stop()`` is
    what produces the ``RuntimeError: Event loop is closed`` escape from
    ``_connection_worker_thread`` that this whole task exists to eliminate.
    The pool must close what it opened, deterministically, before shutdown
    returns.
    """
    real_connect = aiosqlite.connect

    async def wrapper(*args, **kwargs):
        conn = await real_connect(*args, **kwargs)
        if args and 'mode=ro' in str(args[0]):
            opened.append(conn)
            landed.set()
            # Hold the connection hostage: DbPool.get() is now suspended
            # inside `await aiosqlite.connect(...)` with a LIVE worker thread
            # the pool has never seen.
            await asyncio.sleep(_CONNECT_HOLD_SECONDS)
        return conn

    return wrapper


@pytest.mark.asyncio
async def test_lifespan_shutdown_leaves_no_aiosqlite_worker_threads(
    tmp_path: Path, monkeypatch
):
    """Cancelling ``_metrics_loop`` mid-``DbPool.get()`` must not leak a thread.

    End-to-end reduction of the reported cross-file pytest ERROR (task 3466).
    ``lifespan()`` cancels ``metrics_task`` at shutdown; when that task is
    suspended inside ``await aiosqlite.connect(...)`` the ``Connection`` is
    never returned to ``DbPool.get()``, so it never lands in ``_conns`` and
    ``close_all()`` cannot see it.  ``aiosqlite.Connection._connect()`` catches
    only ``Exception`` — never ``CancelledError`` — so it never calls
    ``_stop_running()`` either.  The daemon worker thread is orphaned, and once
    this loop closes its ``call_soon_threadsafe`` raises ``RuntimeError: Event
    loop is closed`` OUT of ``_connection_worker_thread``, where pytest's
    ``threadexception`` plugin attributes it to whatever test happens to be
    running at that instant — hence "roaming target", "cross-file only",
    "passes standalone".

    Deterministic by construction: the patched connect signals ``landed`` only
    after a real ``Connection`` (and therefore a real worker thread) exists,
    and the test does not leave the lifespan block until that signal arrives.
    """
    apply_isolated_env(monkeypatch, tmp_path)
    config = DashboardConfig.from_env()
    assert config.project_root == tmp_path.resolve(), (
        f'isolation failed: config.project_root={config.project_root} is not '
        f'{tmp_path.resolve()} — this test would open ambient databases'
    )
    db_paths = _create_metrics_loop_db_files(config)

    landed = asyncio.Event()
    opened: list[aiosqlite.Connection] = []
    baseline = set(live_aiosqlite_worker_threads())

    local_app = FastAPI(lifespan=lifespan)
    with (
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
        patch('aiosqlite.connect', _holding_readonly_connect(landed, opened)),
    ):
        async with lifespan(local_app):
            # Do not tear down until a read-only connect has provably landed —
            # otherwise there is no thread to leak and this test is vacuous.
            await asyncio.wait_for(landed.wait(), timeout=5.0)
        # Lifespan teardown has now cancelled _metrics_loop mid-connect and run
        # close_all().  Give the held connect time to land and be reaped.
        await asyncio.sleep(_CONNECT_HOLD_SECONDS + 0.5)

    try:
        # Non-vacuity: a read-only connect must actually have happened.
        assert opened, (
            f'no read-only DbPool connect was observed for '
            f'{[str(p) for p in db_paths]} — the metrics loop never reached '
            f'aiosqlite.connect, so this guard asserted nothing'
        )

        leaked = [t for t in live_aiosqlite_worker_threads() if t not in baseline]
        assert not leaked, (
            f'cancelled DbPool.get() orphaned {len(leaked)} aiosqlite worker '
            f'thread(s): {[t.name for t in leaked]}. Each will raise '
            f"'Event loop is closed' out of _connection_worker_thread once this "
            f'loop closes, which pytest attributes to an unrelated test. DbPool '
            f'must own every in-flight connect for: {[str(p) for p in db_paths]}'
        )

        # Deterministic counterpart to the thread check: the pool must have
        # CLOSED each connection it caused to be opened, while the loop was
        # still running.  Verified against aiosqlite >=0.22.x — see the
        # identical private-attribute pin in test_db.py.
        for conn in opened:
            assert conn._running is False, (
                f'DbPool left a mid-flight connection running ({conn!r}); '
                f'close_all() returned without draining its in-flight connects'
            )
    finally:
        # MUST run even when the assertions above fail.  DbPool opens via plain
        # `aiosqlite.connect`, whose worker thread inherits daemon=False from
        # the main thread — so a surviving orphan does not merely leak, it
        # blocks interpreter exit and hangs the whole pytest process AFTER the
        # report is written.  Reap them here so a RED run fails fast and
        # legibly instead of wedging CI.
        for conn in opened:
            if getattr(conn, '_running', False):
                with contextlib.suppress(Exception):
                    await conn.close()


def test_metrics_loop_cancelled_mid_connect_reports_no_unhandled_thread_exception(
    tmp_path: Path, monkeypatch
):
    """Nothing may escape an aiosqlite worker thread after its loop is closed.

    This asserts the exact thing pytest's ``threadexception`` plugin reports —
    an exception escaping ``_connection_worker_thread`` — so it fails for the
    same reason the original suite ERROR fired, independently of test ordering.

    SYNCHRONOUS on purpose.  The defect only manifests once the event loop that
    owned the connect is CLOSED, and a test cannot close the loop it is running
    on.  So this drives ``lifespan()`` on a loop it owns, closes it, and only
    then drops the last reference to whatever the pool left behind.  That final
    ``gc.collect()`` is the trigger: ``aiosqlite.Connection.__del__`` calls
    ``stop()``, whose STOP sentinel makes the worker thread call
    ``future.get_loop().call_soon_threadsafe(...)`` on the now-closed loop →
    ``RuntimeError: Event loop is closed`` raised *inside the thread*, escaping
    through ``_connection_worker_thread`` to ``threading.excepthook``, which
    pytest then blames on whatever test is running at that instant.

    Post-fix the trigger cannot fire: ``close_all()`` closed the connection
    while the loop was still alive, so ``__del__`` finds ``_connection is None``
    and returns without queueing anything.
    """
    apply_isolated_env(monkeypatch, tmp_path)
    config = DashboardConfig.from_env()
    db_paths = _create_metrics_loop_db_files(config)

    escaped: list[threading.ExceptHookArgs] = []
    original_hook = threading.excepthook

    def _recording_hook(args) -> None:
        thread_name = getattr(args.thread, 'name', '') or ''
        target_name = getattr(getattr(args.thread, '_target', None), '__name__', '')
        if '_connection_worker_thread' in (thread_name + target_name):
            escaped.append(args)
            return
        original_hook(args)

    landed = asyncio.Event()
    opened: list[aiosqlite.Connection] = []

    async def _drive_lifespan() -> None:
        local_app = FastAPI(lifespan=lifespan)
        with (
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
            patch('aiosqlite.connect', _holding_readonly_connect(landed, opened)),
        ):
            async with lifespan(local_app):
                await asyncio.wait_for(landed.wait(), timeout=5.0)
            await asyncio.sleep(_CONNECT_HOLD_SECONDS + 0.5)

    threading.excepthook = _recording_hook
    loop = asyncio.new_event_loop()
    # set_event_loop so Connection.stop()'s `asyncio.get_event_loop()` resolves
    # to THIS (soon-closed) loop during the gc below — that is precisely the
    # production condition being reproduced.
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_drive_lifespan())
        assert opened, (
            f'no read-only DbPool connect was observed for '
            f'{[str(p) for p in db_paths]} — the metrics loop never reached '
            f'aiosqlite.connect, so this guard asserted nothing'
        )
        survivors_before_gc = [c for c in opened if getattr(c, '_running', False)]

        loop.close()
        # Drop the last references and force finalisation NOW, while the closed
        # loop is still the current one.
        opened.clear()
        del survivors_before_gc
        gc.collect()
        time.sleep(0.5)  # let any woken worker thread run to its escape
    finally:
        threading.excepthook = original_hook
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)

    assert not escaped, (
        f'{len(escaped)} exception(s) escaped an aiosqlite worker thread after '
        f'its event loop closed: '
        f'{[repr(a.exc_value) for a in escaped]}. This is the roaming pytest '
        f'ERROR — threading.excepthook attributes it to whichever test happens '
        f'to be running. DbPool.close_all() must close every connection it '
        f'caused to be opened while the loop is still alive.'
    )
