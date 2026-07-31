"""Regression + characterization tests for /healthz's whole-handler deadline.

``/healthz`` probes three SQLite databases per request (``dashboard/src/
dashboard/app.py``). Before task 3309, only ``cursor.fetchone()`` carried a
timeout — the connection acquire (``pool.get``), the query execute itself,
and cursor cleanup on cancellation were all unbounded, so a single wedged
connection could hang the whole endpoint indefinitely (measured: a 503
arriving at 50.6s against a ``curl --max-time 5`` caller — see
``plans/dashboard-availability-prd.md`` task epsilon).

Every test in this module calls ``dashboard.app.healthz(request)`` directly
through :func:`_call_healthz`, which wraps the call in ``asyncio.wait_for``
and turns expiry into an explicit ``pytest.fail``. This is deliberate: the
house route-test pattern (the synchronous ``TestClient`` fixture in
conftest.py) cannot be wrapped in ``asyncio.wait_for``, so a RED test against
a hanging handler would wedge the whole suite instead of failing fast.

The fakes below (``_BlockingConn``, ``_BlockingCursorCtx``, ``_BlockingPool``)
model aiosqlite 0.22.1's actual shape, verified via ``inspect``:
``Connection.execute`` is a *synchronous* method (wrapped in aiosqlite's own
``@contextmanager`` helper) returning an object that is both directly
awaitable and usable as an async context manager — the real query execution
happens inside ``__aenter__`` (or the awaited coroutine), not at call time.
Exiting the ``async with`` form always closes the cursor, including when an
exception (such as a cancellation) propagates through the block — that close
is a *new* ``await``, which is why a deadline wrapped around the ``async
with`` block alone does not make it hang-free (see steps 7/8 below).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from dashboard.app import healthz
from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool

# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------

# How long the fakes below block for when configured to hang. Bounded (so
# nothing leaks a truly-eternal task) but far beyond any budget under test —
# real interruption must come from cancellation, not from this ever elapsing.
_BLOCK_SECONDS = 60.0

# Sentinel for _BlockingPool behaviors: get() itself blocks for
# _BLOCK_SECONDS, modelling a stalled aiosqlite.connect() or per-path
# open-lock contention (DbPool.get, dashboard/src/dashboard/data/db.py:66-119).
_BLOCK_IN_GET = object()


class _BlockingCursor:
    """Fake aiosqlite.Cursor — fetchone() and close() each independently
    blockable via *block_on*."""

    def __init__(self, block_on: str | None) -> None:
        self._block_on = block_on

    async def fetchone(self) -> tuple[int]:
        if self._block_on in ('fetch', 'fetch_and_close'):
            await asyncio.sleep(_BLOCK_SECONDS)
        return (1,)

    async def close(self) -> None:
        if self._block_on in ('close', 'fetch_and_close'):
            await asyncio.sleep(_BLOCK_SECONDS)


class _BlockingCursorCtx:
    """Fake for aiosqlite's ``Connection.execute(...)`` return value.

    Real aiosqlite's equivalent (``aiosqlite.context.Result``) supports two
    protocols, and this fake must too since step-8's fix uses the direct-await
    form to sidestep auto-close entirely:

    - ``async with conn.execute(...) as cursor:`` — blocks in ``__aenter__``
      when ``block_on == 'execute'``; always closes the cursor on exit
      (``__aexit__``), which blocks when ``block_on`` is ``'close'`` or
      ``'fetch_and_close'`` — *including* when the block is being exited due
      to a cancellation, which is precisely the trap steps 7/8 exist to
      remove.
    - ``cursor = await conn.execute(...)`` — same blocking-on-open behaviour,
      but never auto-closes (mirrors ``Result.__await__``, which just returns
      the cursor).
    """

    def __init__(self, block_on: str | None) -> None:
        self._block_on = block_on
        self._cursor: _BlockingCursor | None = None

    async def _open(self) -> _BlockingCursor:
        if self._block_on == 'execute':
            await asyncio.sleep(_BLOCK_SECONDS)
        return _BlockingCursor(self._block_on)

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _BlockingCursor:
        self._cursor = await self._open()
        return self._cursor

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._cursor is not None
        await self._cursor.close()


class _BlockingConn:
    """Fake aiosqlite.Connection. ``execute()`` is a plain sync method
    returning a ``_BlockingCursorCtx`` — matching aiosqlite's own
    ``@contextmanager``-wrapped shape (verified via
    ``inspect.getsource(aiosqlite.Connection.execute)``)."""

    def __init__(self, block_on: str | None = None) -> None:
        self._block_on = block_on

    def execute(self, sql: str, parameters: object = None) -> _BlockingCursorCtx:
        return _BlockingCursorCtx(self._block_on)


class _BlockingPool:
    """Stands in for ``DbPool`` with per-db-path-configurable ``get()``.

    *behaviors* maps a db_path to one of:
      - a ``_BlockingConn`` instance   -> get() returns it immediately
      - ``None``                       -> get() returns None ('unavailable')
      - ``_BLOCK_IN_GET``               -> get() itself blocks for _BLOCK_SECONDS
      - a real ``aiosqlite.Connection`` -> returned as-is (step-9's hybrid pool)

    A db_path absent from *behaviors* defaults to 'unavailable' (None), same
    as the real DbPool.get() for a nonexistent file.
    """

    def __init__(self, behaviors: dict[Path, object] | None = None, *, open_count: int = 0) -> None:
        self._behaviors = behaviors or {}
        self.open_count = open_count

    async def get(self, db_path: Path) -> object:
        behavior = self._behaviors.get(db_path)
        if behavior is _BLOCK_IN_GET:
            await asyncio.sleep(_BLOCK_SECONDS)
            return None  # pragma: no cover — a real caller cap always fires first
        return behavior


def _make_healthz_request(
    config: DashboardConfig,
    pool: object,
    *,
    start_time: float | None = None,
) -> Request:
    """Build a minimal stand-in for the Request ``healthz()`` receives.

    ``healthz`` touches exactly three attributes: ``request.app.state.db``,
    ``request.app.state.config``, ``request.app.state.start_time``.
    """
    state = SimpleNamespace(
        db=pool,
        config=config,
        start_time=start_time if start_time is not None else time.monotonic(),
    )
    app = SimpleNamespace(state=state)
    return cast(Request, SimpleNamespace(app=app))


async def _call_healthz(request: Request, *, hard_cap: float) -> tuple[JSONResponse, float]:
    """Call ``healthz(request)`` under a hard external cap; fail fast+loud on hang.

    Every test in this module goes through this wrapper rather than awaiting
    ``healthz()`` directly, so a RED test against a hung handler fails within
    *hard_cap* seconds with an explicit diagnosis instead of wedging the suite.

    Returns ``(response, elapsed_seconds)`` so tests can assert the budget was
    honoured, not just that a response eventually came back.
    """
    start = time.monotonic()
    try:
        resp = await asyncio.wait_for(healthz(request), timeout=hard_cap)
    except TimeoutError:
        pytest.fail(
            f'/healthz did not return within {hard_cap}s — the handler has no '
            'whole-handler deadline and hung instead of returning a verdict'
        )
    return resp, time.monotonic() - start


def _make_real_dbs(config: DashboardConfig) -> None:
    """Create real on-disk sqlite files for every /healthz probe target.

    A real ``DbPool()`` handed these paths returns responsive read-only
    connections, which is what the happy-path characterization test needs.
    """
    for db_path in (config.reconciliation_db, config.write_journal_db, config.runs_db):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute('SELECT 1')
        conn.commit()
        conn.close()


def _body(resp: JSONResponse) -> dict:
    return json.loads(bytes(resp.body))


# ---------------------------------------------------------------------------
# pre-2: happy-path characterization (must stay green through every step below)
# ---------------------------------------------------------------------------


async def test_healthz_reports_healthy_when_all_dbs_respond(dashboard_config):
    """Baseline characterization: all three DBs responsive -> 200 healthy.

    Pins the response contract the deadline refactor must not break: the
    per-DB ``checks`` keys, the ``threads``/``connections``/``uptime_seconds``
    shape, and the 200/``healthy`` mapping. Not a RED — this must stay green
    through every impl step in this plan.
    """
    config = dashboard_config
    _make_real_dbs(config)
    pool = DbPool()
    request = _make_healthz_request(config, pool)

    resp, _elapsed = await _call_healthz(request, hard_cap=8.0)
    body = _body(resp)

    assert resp.status_code == 200
    assert body['status'] == 'healthy'
    checks = body['checks']
    assert checks['db_reconciliation'] == 'ok'
    assert checks['db_write_journal'] == 'ok'
    assert checks['db_runs'] == 'ok'
    assert set(checks['threads']) == {'count', 'limit', 'ok'}
    assert 'open' in checks['connections']
    assert 'uptime_seconds' in checks

    await pool.close_all()
