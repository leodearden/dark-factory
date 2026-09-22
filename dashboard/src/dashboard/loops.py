"""The two wall-clock-aligned background samplers and the stores they own.

The dashboard writes its own history: ``_burndown_loop`` snapshots task
counts and ``_metrics_loop`` snapshots orchestrator/queue metrics, each on
a fixed interval aligned to the wall clock so samples from separate
processes land on the same tick boundaries. Both own a writable WAL-mode
SQLite store, downsample old rows on a slower cadence, and checkpoint
periodically so the WAL cannot grow without bound.

``_BurndownStore`` and ``_MetricsStore`` live here rather than in
``app.py`` because they are these loops' writable stores — the loops are
their only writer, and the migration each performs on ``open()`` exists so
its loop can run. ``app.py``'s ``lifespan`` constructs both and spawns both
loops, importing the four names back.

Those four names carry a leading underscore from when this was all one
file, and they are now this module's interface rather than private detail.
Read the underscore as vestigial, not as a private-use signal — the move
that created this module was a pure extraction, with renaming outside its
scope.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import aiosqlite
import httpx
from fastapi import FastAPI
from shared.async_sqlite_base import AsyncSqliteBase

from dashboard.config import DashboardConfig
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    collect_snapshot,
    downsample,
    ensure_snapshot_columns,
)
from dashboard.data.db import DbPool
from dashboard.data.metrics import (
    METRICS_SCHEMA,
    collect_metrics_snapshot,
    downsample_metrics,
)
from dashboard.project_dbs import _project_scoped_dbs_labeled

logger = logging.getLogger(__name__)


_SAMPLE_INTERVAL_SECONDS = 600  # 10 minutes
_DOWNSAMPLE_INTERVAL_SECONDS = 3600  # 1 hour
_CHECKPOINT_INTERVAL_SECONDS = 3600  # 1 hour


class _BurndownStore(AsyncSqliteBase):
    """Writable WAL-mode store for the burndown snapshot collector.

    Subclasses AsyncSqliteBase so that open() applies the full Phase-3
    durability pragma triad (synchronous=FULL, wal_autocheckpoint=100,
    journal_size_limit=64 MiB) and checkpoint() is available for periodic
    use by _burndown_loop.
    """

    @property
    def _schema(self) -> str:
        return BURNDOWN_SCHEMA

    @property
    def connection(self) -> aiosqlite.Connection:
        """Public accessor for the open connection; raises RuntimeError if not opened."""
        return self._require_conn()

    async def open(self) -> None:
        """Open, then bring an existing DB up to the current column set.

        ``BURNDOWN_SCHEMA`` is applied with ``CREATE TABLE IF NOT EXISTS``, so a
        burndown.db created before a column was added never gains it from the
        DDL alone.  Migrating here — before ``_burndown_loop`` can run — is what
        keeps the collector from ever meeting an un-migrated table.  Closes the
        connection on failure so a half-open store is never left behind.
        """
        await super().open()
        try:
            await ensure_snapshot_columns(self.connection)
            await self.connection.commit()
        except BaseException:
            await self.close()
            raise


class _MetricsStore(AsyncSqliteBase):
    """Writable WAL-mode store for the metrics snapshot collector.

    Mirrors _BurndownStore — subclasses AsyncSqliteBase so that open()
    applies the full Phase-3 durability pragma triad and checkpoint() is
    available for periodic use by _metrics_loop.
    """

    @property
    def _schema(self) -> str:
        return METRICS_SCHEMA

    @property
    def connection(self) -> aiosqlite.Connection:
        """Public accessor for the open connection; raises RuntimeError if not opened."""
        return self._require_conn()


async def _sleep_to_aligned_tick(interval: int) -> None:
    """Sleep until the next wall-clock-aligned interval boundary.

    Avoids drift across long uptimes — without alignment, a sleep(600)
    loop slowly desynchronises from minute boundaries because each
    iteration's wakeup latency accumulates.
    """
    now = time.time()
    target = (int(now) // interval + 1) * interval
    await asyncio.sleep(max(0.0, target - now))


async def _burndown_loop(
    store: _BurndownStore,
    config: DashboardConfig,
    client: httpx.AsyncClient,
) -> None:
    """Periodically snapshot task status counts into the burndown DB."""
    conn = store.connection
    try:
        await collect_snapshot(conn, config, client)
    except Exception:
        logger.warning('Initial burndown snapshot failed', exc_info=True)
    last_downsample = 0.0
    last_checkpoint = 0.0
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        conn = store.connection
        try:
            await collect_snapshot(conn, config, client)
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample(conn)
                last_downsample = now
            if now - last_checkpoint > _CHECKPOINT_INTERVAL_SECONDS:
                try:
                    await store.checkpoint()
                except Exception:
                    logger.warning('Periodic WAL checkpoint failed (burndown)', exc_info=True)
                last_checkpoint = now
        except Exception:
            logger.warning('Burndown snapshot error', exc_info=True)


async def _metrics_loop(
    store: _MetricsStore,
    app: FastAPI,
    *,
    pool: DbPool,
    http_client: httpx.AsyncClient,
) -> None:
    """Periodically snapshot ephemeral system metrics into metrics.db.

    Uses fresh per-cycle handles for the recon DB and per-project runs.db
    files so a stale connection cannot strand the loop. Each sampler in
    collect_metrics_snapshot has its own try/except, so one failed source
    does not poison the others.

    **The binding invariant (task 3771): handles bind to ARGUMENTS, config
    binds to ``app.state``.** The asymmetry is deliberate, not an oversight.

    ``pool`` and ``http_client`` belong to the caller — the lifespan that
    created them — and are never re-read from ``app.state``. ``app.state`` is
    one mutable namespace shared by every lifespan over this ``app``, and an
    inner lifespan (starlette runs a full lifespan per ``TestClient`` context,
    and ~15 module-scoped ``TestClient(app)`` fixtures overlap the
    function-scoped ``client`` fixture) installs its own handles there and does
    not restore the outer's on exit. A loop re-reading ``app.state`` therefore
    spends the rest of the outer lifespan polling the inner's *closed* pool and
    client — silently, since a closed ``DbPool.get()`` returns ``None`` and a
    closed ``httpx`` client raises into ``_run_once``'s ``except Exception``,
    surfacing only as a generic 'Metrics snapshot error'. Both are keyword-only
    and required: an ``app.state`` fallback default would silently reinstate
    exactly that cross-talk, so a stale call site fails loudly instead.

    ``config``, by contrast, IS re-read from ``app.state`` every cycle **on
    purpose** — ~25 tests swap ``client.app.state.config`` mid-test and depend
    on the swap being picked up.

    Accepted residual: an outer lifespan's loop still reads the INNER's config
    *object* once the inner has run. Both come from ``DashboardConfig.from_env()``
    under one environment, so they are value-equivalent and carry no
    closed-handle hazard; removing it would break the swap-ability above.

    Both halves are pinned by ``dashboard/tests/test_lifespan_resource_binding.py``
    — change either one and a test there fails.
    """

    async def _run_once() -> None:
        conn = store.connection
        config: DashboardConfig = app.state.config
        recon_db = await pool.get(config.reconciliation_db)
        tickets_db = await pool.get(config.tickets_db)
        merge_dbs = await _project_scoped_dbs_labeled(
            config,
            pool,
            Path('data/orchestrator/runs.db'),
        )
        await collect_metrics_snapshot(
            conn=conn,
            config=config,
            http_client=http_client,
            recon_db=recon_db,
            merge_dbs=merge_dbs,
            tickets_db=tickets_db,
        )

    try:
        await _run_once()
    except Exception:
        logger.warning('Initial metrics snapshot failed', exc_info=True)
    last_downsample = 0.0
    last_checkpoint = 0.0
    while True:
        await _sleep_to_aligned_tick(_SAMPLE_INTERVAL_SECONDS)
        try:
            await _run_once()
            conn = store.connection
            now = time.monotonic()
            if now - last_downsample > _DOWNSAMPLE_INTERVAL_SECONDS:
                await downsample_metrics(conn)
                last_downsample = now
            if now - last_checkpoint > _CHECKPOINT_INTERVAL_SECONDS:
                try:
                    await store.checkpoint()
                except Exception:
                    logger.warning('Periodic WAL checkpoint failed (metrics)', exc_info=True)
                last_checkpoint = now
        except Exception:
            logger.warning('Metrics snapshot error', exc_info=True)
