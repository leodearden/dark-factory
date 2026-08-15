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
import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

import dashboard.app as app_module
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


class _RaisingCursorCtx:
    """Fake for ``Connection.execute(...)`` that RAISES instead of blocking.

    Mirrors :class:`_BlockingCursorCtx`'s dual protocol (directly awaitable
    *and* usable as an async context manager), but ``_open()`` raises the
    configured exception. Raising from ``_open`` rather than from
    ``execute()`` matches aiosqlite: a sqlite error surfaces from the worker
    thread at the AWAIT point, not at call time.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def _open(self) -> _BlockingCursor:
        raise self._exc

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _BlockingCursor:
        return await self._open()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _RaisingConn:
    """Fake aiosqlite.Connection whose queries fail FAST by raising.

    Models a corrupt database (``DatabaseError: database disk image is
    malformed``), a connection closed under us (``ProgrammingError``), or a
    disk I/O error — failures that return in ~0ms and so are emphatically
    *not* the handler exceeding its budget.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def execute(self, sql: str, parameters: object = None) -> _RaisingCursorCtx:
        return _RaisingCursorCtx(self._exc)


class _BlockingPool:
    """Stands in for ``DbPool`` with per-db-path-configurable ``get()``.

    *behaviors* maps a db_path to one of:
      - a ``_BlockingConn`` instance   -> get() returns it immediately
      - ``None``                       -> get() returns None ('unavailable')
      - ``_BLOCK_IN_GET``               -> get() itself blocks for _BLOCK_SECONDS
      - a ``BaseException`` instance   -> get() raises it, modelling a failing
        acquire (``aiosqlite.connect`` raising ``OperationalError: unable to
        open database file``)
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
        if isinstance(behavior, BaseException):
            raise behavior
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
    try:
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
        assert checks['budget_seconds'] == app_module._HEALTHZ_TOTAL_BUDGET
        assert checks['deadline_exceeded'] is False
    finally:
        # A real DbPool's aiosqlite connections each own a non-daemon worker
        # thread — without this in a `finally`, an assertion failure above
        # (e.g. the new-key RED before step-10 lands) leaks those threads and
        # hangs process/session teardown rather than just failing the test.
        await pool.close_all()


# ---------------------------------------------------------------------------
# step-1 / step-2: budget must be structurally deliverable to the tightest
# real caller (`curl -sf --max-time 5`, dark-factory-dashboard-watchdog.service:6)
# ---------------------------------------------------------------------------


def test_healthz_budget_is_structurally_deliverable(tmp_path):
    """The shipped budget constants must fit under the tightest real caller.

    dark-factory-dashboard-watchdog.service:6 calls `curl -sf --max-time 5`.
    _HEALTHZ_TOTAL_BUDGET must stay strictly below that ceiling, and the SUM
    of per-DB budgets (_DB_PROBE_TIMEOUT * probe count) must fit inside the
    whole-handler budget — otherwise the 15s-behind-a-5s-caller arithmetic
    bug (measured: 503 delivered at 50.6s) can silently return. The probe
    count is derived from _healthz_db_targets(), not hard-coded, so adding a
    4th database without raising the budget fails here too.
    """
    config = DashboardConfig(project_root=tmp_path)
    targets = app_module._healthz_db_targets(config)

    assert app_module._HEALTHZ_TOTAL_BUDGET < 5.0, (
        'the whole-handler budget must stay strictly below the tightest real '
        "caller's ceiling (curl -sf --max-time 5, "
        'dark-factory-dashboard-watchdog.service:6) or its degraded verdict '
        'is undeliverable'
    )
    assert app_module._DB_PROBE_TIMEOUT * len(targets) <= app_module._HEALTHZ_TOTAL_BUDGET, (
        f'{app_module._DB_PROBE_TIMEOUT} * {len(targets)} probes exceeds the '
        f'whole-handler budget of {app_module._HEALTHZ_TOTAL_BUDGET}s'
    )


# ---------------------------------------------------------------------------
# step-3 / step-4: TIMEOUT SCOPE BUG — conn.execute(...) itself blocking (not
# just cursor.fetchone()) must not hang the handler
# ---------------------------------------------------------------------------


async def test_healthz_returns_degraded_when_execute_blocks(dashboard_config):
    """conn.execute('SELECT 1') blocking in __aenter__ must not hang /healthz.

    Today's code wraps only `cursor.fetchone()` in `asyncio.wait_for` — the
    `async with conn.execute(...)` statement's own `__aenter__` (where
    aiosqlite's real execute-the-query work happens) carries no deadline at
    all. All three DB targets block there. Uses the DEFAULT shipped
    constants (no monkeypatching) so this asserts the real user-observable
    signal from the PRD, not an artificially shrunk one.
    """
    config = dashboard_config
    pool = _BlockingPool(
        {
            config.reconciliation_db: _BlockingConn(block_on='execute'),
            config.write_journal_db: _BlockingConn(block_on='execute'),
            config.runs_db: _BlockingConn(block_on='execute'),
        }
    )
    request = _make_healthz_request(config, pool)

    resp, elapsed = await _call_healthz(request, hard_cap=8.0)
    body = _body(resp)

    assert resp.status_code == 503
    assert body['status'] == 'degraded'
    # The handler is bounded at _HEALTHZ_TOTAL_BUDGET (3.0s); ~2.0s of slack
    # below the 5.0s curl --max-time ceiling absorbs event-loop scheduling
    # and JSON serialisation (same convention as test_metrics_curator.py:924).
    assert elapsed < 5.0, (
        f'elapsed {elapsed:.3f}s >= 5.0s — verdict undeliverable to curl --max-time 5'
    )
    checks = body['checks']
    assert checks['db_reconciliation'] == 'timeout'
    assert checks['db_write_journal'] == 'timeout'
    assert checks['db_runs'] == 'timeout'


# ---------------------------------------------------------------------------
# step-5 / step-6: the deadline must also cover the connection-acquire step
# (pool.get), not just execute + fetch
# ---------------------------------------------------------------------------


async def test_healthz_deadline_covers_the_connection_acquire_step(dashboard_config):
    """A stalled pool.get(db_path) must not hang the handler either.

    DbPool.get (dashboard/src/dashboard/data/db.py:66-119) has two unbounded
    awaits of its own: the per-path open-lock, and aiosqlite.connect(). Under
    contention or a stalled filesystem, acquire alone hangs the handler even
    though step-4 already bounds every query. Only the `reconciliation`
    target's acquire blocks; the other two targets are promptly-responsive,
    so one wedged acquire must not starve its neighbours of a real verdict.
    """
    config = dashboard_config
    pool = _BlockingPool(
        {
            config.reconciliation_db: _BLOCK_IN_GET,
            config.write_journal_db: _BlockingConn(),
            config.runs_db: _BlockingConn(),
        }
    )
    request = _make_healthz_request(config, pool)

    resp, elapsed = await _call_healthz(request, hard_cap=8.0)
    body = _body(resp)

    assert resp.status_code == 503
    assert body['status'] == 'degraded'
    assert elapsed < 5.0, (
        f'elapsed {elapsed:.3f}s >= 5.0s — verdict undeliverable to curl --max-time 5'
    )
    checks = body['checks']
    assert checks['db_reconciliation'] == 'timeout'
    assert checks['db_write_journal'] == 'ok'
    assert checks['db_runs'] == 'ok'


# ---------------------------------------------------------------------------
# step-7 / step-8: the deepest trap — asyncio.timeout around `async with
# conn.execute(...)` is not hang-free, because cancellation-triggered cursor
# cleanup issues a NEW, uncancelled await
# ---------------------------------------------------------------------------


async def test_healthz_returns_verdict_when_cursor_cleanup_blocks(dashboard_config):
    """Cursor cleanup blocking on cancellation must not hang /healthz.

    When asyncio.timeout's deadline fires, CancelledError is raised at
    whatever await is in progress. If that's `fetchone()`, the exception
    unwinds through the `async with conn.execute(...) as cursor:` block's
    `__aexit__`, which issues a NEW `await cursor.close()` — a fresh await
    that the ALREADY-FIRED (one-shot) timeout will not cancel again. Using
    `block_on='fetch_and_close'` reproduces this precisely: fetchone() blocks
    long enough for the deadline to fire *during* the with-block (consuming
    the one cancellation), and close() then blocks with nothing left to
    interrupt it — hanging step-6's implementation despite its deadline.
    """
    config = dashboard_config
    pool = _BlockingPool(
        {
            config.reconciliation_db: _BlockingConn(block_on='fetch_and_close'),
            config.write_journal_db: _BlockingConn(block_on='fetch_and_close'),
            config.runs_db: _BlockingConn(block_on='fetch_and_close'),
        }
    )
    request = _make_healthz_request(config, pool)

    resp, elapsed = await _call_healthz(request, hard_cap=8.0)
    body = _body(resp)

    assert resp.status_code == 503
    assert body['status'] == 'degraded'
    assert elapsed < 5.0, (
        f'elapsed {elapsed:.3f}s >= 5.0s — verdict undeliverable to curl --max-time 5'
    )
    checks = body['checks']
    assert checks['db_reconciliation'] == 'timeout'
    assert checks['db_write_journal'] == 'timeout'
    assert checks['db_runs'] == 'timeout'


# ---------------------------------------------------------------------------
# step-9 / step-10: /healthz must STATE its budget and flag deadline expiry,
# not just enforce it silently
# ---------------------------------------------------------------------------


async def test_healthz_states_its_budget_and_flags_deadline_expiry(dashboard_config):
    """An operator reading a 503 must be able to see the budget, not just infer it.

    `checks['db_x'] == 'timeout'` alone cannot distinguish "this DB is slow"
    from "the handler ran out of its own budget", nor reveal what that
    budget was — a fact the handler already has in a variable. Mixed
    scenario (one blocking DB, two real responsive temp DBs via a hybrid
    pool) also re-confirms a blown probe does not poison its neighbours.
    """
    config = dashboard_config
    _make_real_dbs(config)
    real_pool = DbPool()
    try:
        write_journal_conn = await real_pool.get(config.write_journal_db)
        runs_conn = await real_pool.get(config.runs_db)
        pool = _BlockingPool(
            {
                config.reconciliation_db: _BlockingConn(block_on='execute'),
                config.write_journal_db: write_journal_conn,
                config.runs_db: runs_conn,
            }
        )
        request = _make_healthz_request(config, pool)

        resp, elapsed = await _call_healthz(request, hard_cap=8.0)
        body = _body(resp)

        assert resp.status_code == 503
        assert body['status'] == 'degraded'
        assert elapsed < 5.0, (
            f'elapsed {elapsed:.3f}s >= 5.0s — verdict undeliverable to curl --max-time 5'
        )
        checks = body['checks']
        assert checks['db_reconciliation'] == 'timeout'
        assert checks['db_write_journal'] == 'ok'
        assert checks['db_runs'] == 'ok'
        assert checks['budget_seconds'] == app_module._HEALTHZ_TOTAL_BUDGET
        assert checks['deadline_exceeded'] is True
    finally:
        # Same rationale as the pre-2 characterization test above: real_pool
        # owns two real aiosqlite connections, each a non-daemon worker
        # thread. Skipping close_all() on assertion failure (e.g. the
        # pre-step-10 RED) leaks those threads and hangs process teardown.
        await real_pool.close_all()


# ---------------------------------------------------------------------------
# step-11 / step-12: a probe that fails FAST by raising is an 'error', not a
# 'timeout' — and must not claim the handler blew its budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('inject_at', ['acquire', 'execute'])
async def test_healthz_reports_error_not_timeout_when_probe_raises(
    dashboard_config, caplog, inject_at
):
    """A probe that RAISES must report 'error', not a budget expiry.

    ``_probe_db`` maps every exception to ``'timeout'``, and the handler keys
    ``checks['deadline_exceeded']`` on exactly that string. So a corrupt DB
    (``sqlite3.DatabaseError: database disk image is malformed``), a
    connection closed under us, or a disk I/O error — all of which return in
    ~0ms — are reported to the operator as the handler having blown its
    deadline. That points them at latency and budgets when the fact in hand
    was an exception, inverting the very purpose ``deadline_exceeded`` was
    added for in step-9/step-10. The exception is also swallowed with no
    logging, so the real cause is unrecoverable anywhere (INV-2
    ``structured-facts-at-failure``, docs/legibility/design-invariants.md).

    Both failure-injection sites are covered: raising during the connection
    acquire (``aiosqlite.connect`` failing) and raising during the query
    execute (the worker thread surfacing a sqlite error at the await point).

    The ``elapsed`` bound is chosen structurally, not guessed: it is strictly
    below ONE per-DB budget, so if it holds, no probe can possibly have
    reached its deadline — which is what makes ``deadline_exceeded is False``
    a fact about the code rather than a coincidence. Three probes that raise
    immediately return in ~0ms, so the margin is ~3 orders of magnitude,
    ample for event-loop scheduling and JSON serialisation (the same headroom
    convention as the ``elapsed < 5.0`` assertions above).
    """
    config = dashboard_config
    targets = (config.reconciliation_db, config.write_journal_db, config.runs_db)

    def _make_exc() -> sqlite3.DatabaseError:
        # A fresh instance per target: re-raising one shared instance would
        # chain three tracebacks onto the same object.
        return sqlite3.DatabaseError('database disk image is malformed')

    behaviors: dict[Path, object]
    if inject_at == 'acquire':
        behaviors = {path: _make_exc() for path in targets}
    else:
        behaviors = {path: _RaisingConn(_make_exc()) for path in targets}

    pool = _BlockingPool(behaviors)
    request = _make_healthz_request(config, pool)

    with caplog.at_level(logging.WARNING, logger='dashboard.app'):
        resp, elapsed = await _call_healthz(request, hard_cap=8.0)
    body = _body(resp)

    checks = body['checks']
    assert checks['db_reconciliation'] == 'error'
    assert checks['db_write_journal'] == 'error'
    assert checks['db_runs'] == 'error'
    assert checks['deadline_exceeded'] is False, (
        'a probe that failed fast by raising must not be reported as the '
        'handler having exceeded its budget — that sends the operator to look '
        'at latency when the fact in hand was an exception'
    )
    # An errored probe must STILL flip healthy: a corrupt DB is not healthy.
    # Guards against the fix over-correcting into a fail-soft.
    assert resp.status_code == 503
    assert body['status'] == 'degraded'
    assert elapsed < app_module._DB_PROBE_TIMEOUT, (
        f'elapsed {elapsed:.3f}s >= one per-DB budget '
        f'({app_module._DB_PROBE_TIMEOUT}s) — a raising probe returns '
        'immediately, so deadline_exceeded being False must be provable'
    )

    warning_records = [
        r for r in caplog.records if r.levelno == logging.WARNING and r.name == 'dashboard.app'
    ]
    assert len(warning_records) == len(targets), (
        f'expected one WARNING per failing probe ({len(targets)}), got '
        f'{len(warning_records)} — the payload carries only a status string, '
        'so an unlogged exception is unrecoverable everywhere'
    )
    for db_path, rec in zip(targets, warning_records, strict=True):
        assert str(db_path) in rec.getMessage(), (
            f'WARNING must name the failing db path: {rec.getMessage()!r}'
        )
        assert rec.exc_info is not None, (
            'the payload carries only the status string "error", so the '
            'exception TYPE must survive in the logs'
        )


# ---------------------------------------------------------------------------
# step-13 / step-14: caller/handler cancellation must not strand an
# untracked probe task (task 4089 — reopens the task-leak class task 3466
# already closed once, this time via CALLER cancellation rather than budget
# expiry)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def _drained_probe_registry():
    """Yield ``_ABANDONED_PROBES``; on teardown, cancel + drain whatever remains.

    Teardown-ONLY: deliberately does not ``.clear()`` the registry at setup.
    Clearing would drop the strong reference to a task left in-flight by an
    earlier test, which would GC it mid-flight — exactly the hazard this
    registry exists to prevent (see the module comment at app.py:540-543).
    Same hygiene rationale as the ``finally: await pool.close_all()`` blocks
    above (lines 298-303 / 509-514): a RED assertion failure here must fail
    loudly, not leak a pending task into the rest of the suite.
    """
    registry = app_module._ABANDONED_PROBES
    yield registry
    pending = set(registry)
    if pending:
        for task in pending:
            task.cancel()
        await asyncio.wait(pending, timeout=5.0)


@pytest.mark.parametrize('trigger', ['budget_expiry', 'caller_cancelled'])
async def test_probe_db_abandons_and_tracks_on_every_exit_path(
    dashboard_config, _drained_probe_registry, trigger
):
    """No exit path out of ``_probe_db`` may leave its inner task untracked.

    ``budget_expiry`` (budget=0.05) characterizes the pre-existing task-3466
    fix: the inner task blocks past the budget, so ``_probe_db`` abandons +
    tracks it on the ``if task not in done`` branch. This must stay green
    through this change.

    ``caller_cancelled`` uses budget=60.0 — chosen STRUCTURALLY, not tuned.
    60.0s is far beyond any wall-clock this test can reach (and beyond the
    60s pytest-timeout), so the outer ``probe`` task's own ``.cancel()`` is
    provably the ONLY thing that can end the probe; the budget itself cannot
    have expired. Before this change, cancelling the caller propagates
    ``CancelledError`` straight out of ``_probe_db``'s ``asyncio.wait``,
    jumping over the abandon+track call entirely — the inner task is then
    referenced only by the event loop's WEAK set, the exact GC hazard
    ``_ABANDONED_PROBES`` exists to prevent.

    The inner task is identified by diffing ``asyncio.all_tasks()`` across
    task creation, not by reading ``_ABANDONED_PROBES`` membership — the
    registry is a module-level global that other tests in this module also
    populate, so counting its members would be order- and timing-dependent.
    """
    config = dashboard_config
    registry = _drained_probe_registry
    pool = _BlockingPool({config.reconciliation_db: _BlockingConn(block_on='execute')})
    budget = 0.05 if trigger == 'budget_expiry' else 60.0

    before = asyncio.all_tasks()
    probe = asyncio.create_task(
        app_module._probe_db(cast(DbPool, pool), config.reconciliation_db, budget)
    )
    await asyncio.sleep(0.01)
    new_tasks = (asyncio.all_tasks() - before) - {probe}
    assert len(new_tasks) == 1, f'expected exactly one new inner task, got {new_tasks}'
    inner = next(iter(new_tasks))

    if trigger == 'budget_expiry':
        assert await probe == 'timeout'
    else:
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe

    assert inner in registry, 'inner probe task was not tracked in _ABANDONED_PROBES'
    await asyncio.wait({inner}, timeout=5.0)
    assert inner.cancelled(), 'inner probe task was tracked but never actually cancelled'
    assert inner not in registry, (
        'inner probe task leaked: still in _ABANDONED_PROBES after it finished'
    )


async def test_healthz_handler_cancellation_strands_no_probe_task(
    dashboard_config, _drained_probe_registry
):
    """Cancelling the whole ``/healthz`` handler must not strand a probe task.

    The production trigger: uvicorn's ``--timeout-graceful-shutdown`` force-
    cancels remaining request tasks at every restart (the dashboard watchdog
    restarts this unit routinely), and a client disconnect reaches the same
    path. All three DB targets block in ``execute()`` so the handler is
    guaranteed to still be parked inside ``_probe_db``'s ``asyncio.wait``
    (on the FIRST target) when cancellation lands — only one inner task is
    ever in flight at a time because ``healthz`` awaits ``_probe_db``
    sequentially per DB target.
    """
    config = dashboard_config
    registry = _drained_probe_registry
    pool = _BlockingPool(
        {
            config.reconciliation_db: _BlockingConn(block_on='execute'),
            config.write_journal_db: _BlockingConn(block_on='execute'),
            config.runs_db: _BlockingConn(block_on='execute'),
        }
    )
    request = _make_healthz_request(config, pool)

    before = asyncio.all_tasks()
    handler = asyncio.create_task(healthz(request))
    await asyncio.sleep(0.01)
    new_tasks = (asyncio.all_tasks() - before) - {handler}
    assert len(new_tasks) == 1, (
        f'expected exactly one in-flight inner probe task, got {new_tasks}'
    )
    inner = next(iter(new_tasks))

    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler

    assert inner in registry, 'handler cancellation stranded an untracked probe task'
    await asyncio.wait({inner}, timeout=5.0)
    assert inner.cancelled(), 'stranded probe task was tracked but never actually cancelled'
