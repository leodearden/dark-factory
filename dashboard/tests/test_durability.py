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

from dashboard.app import app, lifespan


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
