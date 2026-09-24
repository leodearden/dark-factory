"""The app lifespan must reap detached bypass refreshes at shutdown (task 5185).

``TTLCache._start_bypass`` fires a bare ``asyncio.create_task`` that the
cache's abandon-don't-cancel policy deliberately never cancels — correct while
the process runs, since a late store still heals the key for the next caller.
``lifespan``'s shutdown, though, cancels exactly the tasks it started itself —
``collector_task``, ``metrics_task`` and the ``http_pool`` orphan reaper — so a
bypass started under one app outlives it: the caches are
module-level and so process-global, and it keeps pinning a connection on an
``httpx.AsyncClient`` that is about to be closed.

In this suite that is not merely a leak. Every ``TestClient(app)`` runs a fresh
event loop in its own thread, so a task surviving one lifespan is still pending
when the next test file starts — and a task bound to a by-then-closed loop is
the ``RuntimeError: Event loop is closed`` class ``lifespan``'s own docstring
records for task 3466.

Its own module rather than a section of ``test_mcp_fanout.py`` (~2900 lines) or
``test_durability.py``: the single subject here is one clause of ``lifespan``'s
shutdown contract, which makes sense read in isolation.

The second clause pinned here is the teardown's ORDERING contract: the reap
sits above the store, pool and client closes, so anything escaping it would
skip all four and strand exactly the writable WAL connections whose ``__del__``
later queues work onto a closed loop. The hook added to prevent that failure
would otherwise be capable of causing it.

``lifespan`` is driven DIRECTLY as an async context manager, the shape
``test_app_http_limits.py``'s pool-sizing test uses, rather than through a
``TestClient``. That keeps the assertions on the same event loop as the test,
which matters precisely because the loop-identity filter in
``TTLCache.cancel_live_bypasses`` is what is being exercised: through a
``TestClient`` the lifespan would run on a different loop and the bypass task
built here would be the foreign-loop case, which is deliberately NOT reaped.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _dashboard_helpers import apply_isolated_env, drain, wedge_one_bypass
from fastapi import FastAPI

from dashboard.app import lifespan
from dashboard.data.mcp_fanout import TTLCache
from dashboard.loops import _BurndownStore


class TestLifespanReapsDetachedBypassRefreshes:
    """A bypass refresh must not survive the lifespan that spawned it."""

    @staticmethod
    async def _run_lifespan_around_one_detached_bypass(tmp_path, monkeypatch):
        """Run one full lifespan with a wedged bypass in flight inside it.

        Returns ``(bypass_task, observed)``. ``observed`` records what was true
        of the bypass at the instant the shared client's ``aclose()`` was
        entered — an observation of the running system, not a reading of
        ``lifespan``'s source, so the ordering claim survives a refactor of it.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        apply_isolated_env(monkeypatch, tmp_path)
        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)

        wedged: dict[str, asyncio.Task] = {}
        observed: dict[str, bool] = {}
        real_async_client = httpx.AsyncClient

        def _observing_async_client(*args, **kwargs):
            # A REAL client, so aclose() and the pool behave normally; only
            # the moment of closing is instrumented.
            client = real_async_client(*args, **kwargs)
            real_aclose = client.aclose

            async def _observing_aclose():
                task = wedged.get('bypass')
                observed['bypass_done'] = task is not None and task.done()
                observed['bypass_cancelled'] = task is not None and task.cancelled()
                return await real_aclose()

            client.aclose = _observing_aclose
            return client

        with (
            patch('dashboard.app.httpx.AsyncClient', _observing_async_client),
            # The background loops' own fan-out is not what this module is
            # about, and un-stubbed it would dial the configured fused-memory
            # endpoint on every run.
            patch('dashboard.loops.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.loops.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
        ):
            cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
            async with lifespan(FastAPI(lifespan=lifespan)):
                wedged['bypass'], caller = await wedge_one_bypass(cache)
                assert not wedged['bypass'].done(), (
                    'precondition: the bypass is genuinely in flight when the '
                    'lifespan begins shutting down'
                )

        await drain(caller)
        assert observed, 'lifespan never closed the client it constructed'
        return wedged['bypass'], observed

    async def test_a_detached_bypass_does_not_outlive_the_app(
        self, tmp_path, monkeypatch
    ):
        bypass, _observed = await self._run_lifespan_around_one_detached_bypass(
            tmp_path, monkeypatch
        )

        assert bypass.cancelled(), (
            'a bypass refresh still in flight at shutdown must be cancelled by '
            'the lifespan that spawned it. Left running it pins a connection on '
            'a client about to close, and — since TTLCaches are process-global '
            'while event loops are not — survives into the next test file, '
            'where its loop is already closed'
        )

    async def test_the_reap_lands_before_the_shared_client_is_closed(
        self, tmp_path, monkeypatch
    ):
        """Ordering, observed rather than read off the source.

        A reap after ``aclose()`` would still end the task, but it would unwind
        against a closed pool instead of returning its connection to a live
        one. The check is what was true of the bypass at the instant ``aclose``
        was entered.
        """
        _bypass, observed = await self._run_lifespan_around_one_detached_bypass(
            tmp_path, monkeypatch
        )

        assert observed['bypass_cancelled'], (
            'the bypass must already be cancelled when http_client.aclose() is '
            f'entered, so it releases its pooled connection to a client that is '
            f'still open; observed at aclose: {observed}'
        )


class TestLifespanClosesItsResourcesEvenIfTheReapFails:
    """A failing teardown step must not take the closes below it with it.

    ``lifespan``'s teardown is a flat sequence: the reap, then
    ``burndown_store.close()``, ``metrics_store.close()``, ``pool.close_all()``
    and ``http_client.aclose()``. Anything escaping the reap skips all four,
    stranding two writable WAL connections and a ``DbPool`` — the handles
    whose finalisers queue work onto a by-then-closed loop, which is the
    ``RuntimeError: Event loop is closed`` failure this same lifespan's
    docstring records for task 3466. The shutdown hook this task ADDED to
    prevent stranded work would be capable of causing it.

    The hazard is structural, so it is closed structurally rather than by an
    argument that nothing above the closes can raise any more.
    """

    _BOOM = 'the reap itself blew up'

    async def test_a_failing_reap_still_closes_what_the_lifespan_opened(
        self, tmp_path, monkeypatch
    ):
        apply_isolated_env(monkeypatch, tmp_path)
        # Held, not inlined into the `async with`: app.state is the observable
        # and it has to outlive the context.
        app = FastAPI(lifespan=lifespan)

        with (
            patch(
                'dashboard.app.reap_detached_refreshes',
                new=AsyncMock(side_effect=RuntimeError(self._BOOM)),
            ),
            patch('dashboard.loops.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.loops.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
            # The fix must not convert a shutdown bug into silence: a reap that
            # fails is a real defect and its only route to an operator is out
            # of shutdown.
            pytest.raises(RuntimeError, match=self._BOOM),
        ):
            async with lifespan(app):
                pass

        assert app.state.http_client.is_closed, (
            'the resources this lifespan opened must be closed on every exit '
            'path, including one where a teardown step above them raised. '
            'aclose() is LAST in that sequence, so this one observable stands '
            'for the burndown store, the metrics store and the DB pool too'
        )


class TestLifespanClosesTheRestWhenOneCloseFails:
    """A handle that refuses to close must not take its neighbours with it.

    The ``finally`` above protects the closes from a failing REAP; it does not
    protect them from EACH OTHER. Run as a flat sequence, a raising
    ``burndown_store.close()`` skips the metrics store, the ``DbPool`` and the
    shared client — and that is not a hypothetical shape:
    ``AsyncSqliteBase.close()`` awaits ``self._conn.close()``, so a failure
    there is the same aiosqlite/loop incident class this lifespan documents.
    The invariant its docstring states — every resource this lifespan opened
    is closed on every exit path — has to hold for that exit path too, or it
    is a claim broader than the code behind it.
    """

    _CLOSE_BOOM = 'the burndown store refused to close'

    async def test_a_failing_close_does_not_skip_the_closes_behind_it(
        self, tmp_path, monkeypatch
    ):
        apply_isolated_env(monkeypatch, tmp_path)
        app = FastAPI(lifespan=lifespan)

        with (
            # The FIRST close in the sequence, so every other one is behind it
            # and a flat sequence would skip all three.
            patch.object(
                _BurndownStore,
                'close',
                autospec=True,
                side_effect=RuntimeError(self._CLOSE_BOOM),
            ),
            patch('dashboard.loops.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.loops.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
            # Still a real defect, so it still reaches an operator.
            pytest.raises(RuntimeError, match=self._CLOSE_BOOM),
        ):
            async with lifespan(app):
                pass

        assert app.state.http_client.is_closed, (
            'a close that raises must not skip the closes behind it. aclose() '
            'is LAST in that sequence, so this one observable stands for the '
            'metrics store and the DB pool too — both of which a flat '
            'sequence would have stranded along with it'
        )
        # The patched-out close genuinely did not run, so this test owns the
        # handle it left open: dropped instead, its finaliser is exactly the
        # stranded writable WAL connection this module exists to keep out of
        # the suite.
        await app.state.burndown_store.close()

