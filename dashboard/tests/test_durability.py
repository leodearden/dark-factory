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

from dashboard.app import _BurndownStore, _MetricsStore, _burndown_loop, _metrics_loop, app, lifespan
from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool
from shared.async_sqlite_base import CheckpointResult


# ---------------------------------------------------------------------------
# Step-1: burndown store pragma triad
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burndown_store_applies_full_pragma_triad_after_lifespan(
    tmp_path: Path, monkeypatch
):
    """_BurndownStore open() applies the full Phase-3 durability pragma triad.

    Calls lifespan(app) directly as an async context manager so that the
    pragma assertions can run inside an async scope with access to the live
    writer connection. The five per-connection PRAGMAs cannot be verified by
    opening a fresh reader — they must be checked on the connection that set them.

    collect_snapshot and collect_metrics_snapshot are patched to AsyncMock so
    the lifespan startup does not hit the network or per-project DBs.
    """
    monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', str(tmp_path))

    with (
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
    ):
        async with lifespan(app):
            # app.state.burndown_store must exist after lifespan startup.
            assert hasattr(app.state, 'burndown_store'), (
                'app.state.burndown_store not set after lifespan startup — '
                '_BurndownStore wrapper not yet implemented in lifespan()'
            )
            store = app.state.burndown_store
            assert store._conn is not None, 'burndown_store._conn is None — store was not opened'
            conn = store._require_conn()

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

    # After lifespan exits the store must be closed.
    assert app.state.burndown_store._conn is None, (
        'burndown_store._conn should be None after lifespan exit (store not closed)'
    )


# ---------------------------------------------------------------------------
# Step-3: metrics store pragma triad
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_store_applies_full_pragma_triad_after_lifespan(
    tmp_path: Path, monkeypatch
):
    """_MetricsStore open() applies the full Phase-3 durability pragma triad.

    Mirrors test_burndown_store_applies_full_pragma_triad_after_lifespan but
    targets app.state.metrics_store and the metrics writer connection.
    """
    monkeypatch.setenv('DASHBOARD_PROJECT_ROOT', str(tmp_path))

    with (
        patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
        patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock(return_value=None)),
    ):
        async with lifespan(app):
            # app.state.metrics_store must exist after lifespan startup.
            assert hasattr(app.state, 'metrics_store'), (
                'app.state.metrics_store not set after lifespan startup — '
                '_MetricsStore wrapper not yet implemented in lifespan()'
            )
            store = app.state.metrics_store
            assert store._conn is not None, 'metrics_store._conn is None — store was not opened'
            conn = store._require_conn()

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

    # After lifespan exits the store must be closed.
    assert app.state.metrics_store._conn is None, (
        'metrics_store._conn should be None after lifespan exit (store not closed)'
    )


# ---------------------------------------------------------------------------
# Step-5: burndown loop invokes periodic checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burndown_loop_invokes_periodic_checkpoint(tmp_path: Path):
    """_burndown_loop calls store.checkpoint() on the periodic interval timer.

    Drives _burndown_loop directly (no lifespan), patches _CHECKPOINT_INTERVAL_SECONDS
    to 0 so the checkpoint fires on the first loop body iteration, and asserts that
    store.checkpoint was called at least once before the task is cancelled.
    """
    store = _BurndownStore(tmp_path / 'burndown.db', busy_timeout_ms=5000)
    await store.open()

    # Replace checkpoint with an AsyncMock so we can count invocations.
    checkpoint_mock = AsyncMock(return_value=CheckpointResult(0, 0, 0))
    store.checkpoint = checkpoint_mock  # type: ignore[method-assign]

    # Build a minimal config (no network needed — collect_snapshot is patched).
    config = DashboardConfig(project_root=tmp_path)

    called_event = asyncio.Event()
    call_count = 0

    async def _snapshot_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            called_event.set()

    mock_collect = AsyncMock(side_effect=_snapshot_side_effect)

    try:
        with (
            patch('dashboard.app.collect_snapshot', mock_collect),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(return_value=None)),
            patch('dashboard.app._CHECKPOINT_INTERVAL_SECONDS', 0),
        ):
            task = asyncio.create_task(
                _burndown_loop(store, config, MagicMock())
            )
            try:
                await asyncio.wait_for(called_event.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    assert checkpoint_mock.called, (
        '_burndown_loop did not call store.checkpoint() — periodic checkpoint not yet implemented'
    )
