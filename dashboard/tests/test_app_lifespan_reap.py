"""The app lifespan must reap detached bypass refreshes at shutdown (task 5185).

``TTLCache._start_bypass`` fires a bare ``asyncio.create_task`` that the
cache's abandon-don't-cancel policy deliberately never cancels — correct while
the process runs, since a late store still heals the key for the next caller.
``lifespan``'s shutdown, though, cancels exactly ``collector_task`` and
``metrics_task``, so a bypass started under one app outlives it: the caches are
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
from _dashboard_helpers import apply_isolated_env
from fastapi import FastAPI

from dashboard.app import lifespan
from dashboard.data.mcp_fanout import TTLCache


class TestLifespanReapsDetachedBypassRefreshes:
    """A bypass refresh must not survive the lifespan that spawned it."""

    @staticmethod
    def _never_resolving_refresh():
        """Refresh stub that enters, signals, and then never resolves.

        The idiom ``test_mcp_fanout.py``'s bounded-acquisition classes use: a
        genuinely unresolved ``asyncio.Event``, never a sleep — a sleeping stub
        would eventually finish on its own and prove nothing about reaping.
        """
        entered = asyncio.Event()
        wedged = asyncio.Event()

        async def _refresh():
            entered.set()
            await wedged.wait()  # never set
            raise AssertionError('unreachable: the wedged event is never set')

        return _refresh, entered

    @classmethod
    async def _wedge_one_bypass(cls, cache, key='k'):
        """Put one genuinely in-flight bypass on *cache*, via the real public path.

        Holds *key*'s lock so the caller's bounded acquisition times out into
        the bypass path, and waits until the refresh has actually been ENTERED
        before returning. Returns ``(bypass_task, caller_task)``; the caller is
        parked on the shielded bypass and never returns on its own.
        """
        refresh, entered = cls._never_resolving_refresh()
        lock = cache._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        try:
            caller = asyncio.create_task(cache.get_or_refresh(key, refresh))
            await asyncio.wait_for(entered.wait(), timeout=5.0)
        finally:
            lock.release()
        return cache._bypass_tasks[key][1], caller

    @classmethod
    async def _run_lifespan_around_one_detached_bypass(cls, tmp_path, monkeypatch):
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
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
        ):
            cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
            async with lifespan(FastAPI(lifespan=lifespan)):
                wedged['bypass'], caller = await cls._wedge_one_bypass(cache)
                assert not wedged['bypass'].done(), (
                    'precondition: the bypass is genuinely in flight when the '
                    'lifespan begins shutting down'
                )

        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
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
