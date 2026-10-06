"""The contract of ``_dashboard_helpers.py::drive_metrics_loop``.

Five ``_metrics_loop`` tests share that driver; these pin the properties they
rely on but do not themselves observe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _dashboard_helpers import drive_metrics_loop

from dashboard.config import DashboardConfig
from dashboard.loops import _metrics_loop, _MetricsStore


def _live_metrics_loop_tasks() -> list[asyncio.Task[Any]]:
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done() and getattr(task.get_coro(), 'cr_code', None) is _metrics_loop.__code__
    ]


def _null_pool() -> MagicMock:
    pool = MagicMock()
    pool.get = AsyncMock(return_value=None)
    return pool


def _app_with_its_own_handles(config: DashboardConfig) -> MagicMock:
    app = MagicMock()
    app.state.config = config
    app.state.db = _null_pool()
    app.state.http_client = MagicMock()
    return app


@pytest.mark.asyncio
async def test_drives_cycles_until_the_event_and_records_each_call(tmp_path: Path) -> None:
    # A REAL config, not a MagicMock -- check_bare_magicmock_config.py Rule A.
    config = DashboardConfig(project_root=tmp_path)
    pool = _null_pool()
    client = MagicMock()
    app = _app_with_its_own_handles(config)
    state_db = app.state.db
    state_http_client = app.state.http_client

    seen: list[dict[str, Any]] = []
    until = asyncio.Event()

    def _hook(kwargs: dict[str, Any]) -> None:
        seen.append(kwargs)
        if len(seen) >= 3:
            until.set()

    async with _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000) as store:
        calls = await drive_metrics_loop(
            store, app, pool=pool, http_client=client, until=until, on_collect=_hook
        )

    assert until.is_set(), 'the hook never saw three cycles, so `until` was never set'
    assert len(calls) >= 3, (
        f'only {len(calls)} cycles ran -- the patched sleep must suspend so the loop '
        f'keeps cycling instead of starving the event loop'
    )
    assert calls == seen, 'recording and on_collect must run in lockstep and in order'
    assert all(call['http_client'] is client for call in calls), (
        'every cycle must forward the http_client passed to the driver'
    )
    assert all(call['config'] is config for call in calls), (
        'every cycle must forward the app.state.config the caller installed'
    )
    assert pool.get.await_count >= 1, 'the loop never opened a connection through the passed pool'
    assert app.state.db is state_db, 'the driver must never write app.state.db'
    assert app.state.http_client is state_http_client, (
        'the driver must never write app.state.http_client'
    )
    assert _live_metrics_loop_tasks() == [], 'the driver must not leave a _metrics_loop running'


@pytest.mark.asyncio
async def test_without_until_it_stops_at_the_first_collect(tmp_path: Path) -> None:
    app = _app_with_its_own_handles(DashboardConfig(project_root=tmp_path))
    backstop = 10.0
    clock = asyncio.get_running_loop()

    async with _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000) as store:
        started = clock.time()
        calls = await drive_metrics_loop(
            store, app, pool=_null_pool(), http_client=MagicMock(), timeout=backstop
        )
        elapsed = clock.time() - started

    assert calls, 'the first cycle runs before any sleep, so it must have been recorded'
    assert elapsed < backstop / 2, (
        f'with no `until` the driver must stop at the first collect, but it ran '
        f'{elapsed:.1f}s -- out to its {backstop}s backstop'
    )
    assert _live_metrics_loop_tasks() == [], 'the driver must not leave a _metrics_loop running'


@pytest.mark.asyncio
async def test_returns_rather_than_raises_when_the_event_never_fires(tmp_path: Path) -> None:
    config = DashboardConfig(project_root=tmp_path)
    app = _app_with_its_own_handles(config)
    never_set = asyncio.Event()

    async with _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000) as store:
        calls = await drive_metrics_loop(
            store,
            app,
            pool=_null_pool(),
            http_client=MagicMock(),
            until=never_set,
            timeout=0.05,
        )

    assert calls, 'the first cycle runs before any sleep, so it must have been recorded'
    assert _live_metrics_loop_tasks() == [], (
        'the driver must cancel the loop even when its backstop timeout fires'
    )
