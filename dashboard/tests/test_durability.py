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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
