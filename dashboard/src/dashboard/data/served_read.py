"""A read served as a ``Datum``: cached, single-flight, waited on for a bounded time, and
falling back to its last good value (the pattern of ``plans/dashboard-prd-lanes-prd.md`` §4.2).

No clock read: every instant is the caller's injected *now*. The TTL is
:class:`~dashboard.data.mcp_fanout.TTLCache`'s own monotonic one.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable, Coroutine, Hashable
from datetime import datetime
from typing import Any, Generic, TypeVar

from shared.asyncio_tasks import track_task

from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum
from dashboard.data.mcp_fanout import TTLCache

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')
T = TypeVar('T')

_OUTLIVED_WAITS: set[asyncio.Task[Any]] = set()


async def wait_shielded(
    work: Coroutine[Any, Any, T], wait_seconds: float | None, *, what: str,
) -> T:
    """*work*'s outcome, waited on at most *wait_seconds*; past that ``TimeoutError``.

    Work its caller stops waiting on runs on, held until it ends, so it still
    lands for the next caller. Nobody is left to receive a failure it then
    ends in, so that failure is logged as a BUG naming *what*.
    """
    task = asyncio.ensure_future(work)
    try:
        return await asyncio.wait_for(asyncio.shield(task), wait_seconds)
    finally:
        if not task.done():
            track_task(task, _OUTLIVED_WAITS, on_done=functools.partial(_log_late_failure, what))


def _log_late_failure(what: str, task: asyncio.Task[Any]) -> None:
    if task.cancelled() or (failure := task.exception()) is None:
        return
    logger.warning(
        '%s failed after its caller stopped waiting on it; this is a BUG, not an outage',
        what, exc_info=failure,
    )


class ServedRead(Generic[K, V]):
    """Serves each key's read through one cache and one last good value.

    A read returns a ``Datum`` and never raises; one that raises anyway is a
    bug, logged and served as a failed read. A read that answers with a value
    (fresh, lower bound or stale) is cached for the TTL and becomes the key's
    last good value. A read that answers unknown is not cached, and the key
    is served its last good value as ``stale`` with the read's reason, or the
    read's unknown before any success.

    With a *wait_seconds*, a request waits that long and is then served the
    last good value as ``stale`` (unknown before any success). The read is
    shielded, so it still lands for the next request.

    A last good value is kept per *last_good_slot* of the key, the key itself
    by default. Keys sharing a slot share one value, the latest success among
    them, so a slot that many short-lived keys pass through holds one value
    rather than one per key.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float | Callable[[], float],
        freshness_bound_seconds: int,
        last_good_slot: Callable[[K], Hashable] = lambda key: key,
    ) -> None:
        self._cache: TTLCache[Datum[V], K] = TTLCache(ttl_seconds)
        self._last_good: dict[Hashable, Datum[V]] = {}
        self._last_good_slot = last_good_slot
        self._freshness_bound_seconds = freshness_bound_seconds
        self._generation = 0

    async def serve(
        self,
        key: K,
        read: Callable[[], Awaitable[Datum[V]]],
        *,
        now: datetime,
        wait_seconds: float | None = None,
    ) -> Datum[V]:
        """*key*'s read as it reads at *now*."""
        cached = self._cache.get_fresh(key)
        if cached is not None:
            return aged_at(cached, now)
        try:
            outcome = await wait_shielded(
                self._refresh(key, read, self._generation), wait_seconds,
                what=f'served read for {key!r}',
            )
        except TimeoutError:
            return self._fallback(key, f'read still in flight after {wait_seconds}s')
        if outcome.value is None:
            return self._fallback(key, outcome.reason or 'read answered no value')
        return aged_at(outcome, now)

    def clear(self) -> None:
        """Forget every key's cached and last good value; a read in flight lands nowhere."""
        self._cache.clear()
        self._last_good.clear()
        self._generation += 1

    async def _refresh(
        self, key: K, read: Callable[[], Awaitable[Datum[V]]], generation: int,
    ) -> Datum[V]:
        def current(datum: Datum[V]) -> bool:
            return datum.value is not None and generation == self._generation

        outcome = await self._cache.get_or_refresh(
            key, lambda: self._guarded(key, read), cache_ok=current,
        )
        if current(outcome):
            self._last_good[self._last_good_slot(key)] = outcome
        return outcome

    async def _guarded(self, key: K, read: Callable[[], Awaitable[Datum[V]]]) -> Datum[V]:
        try:
            return await read()
        except Exception as exc:
            logger.warning(
                'served read for %r raised instead of answering a Datum; this is a BUG, '
                'not an outage', key, exc_info=True,
            )
            return unknown_datum(
                f'read raised {type(exc).__name__}: {exc}', self._freshness_bound_seconds,
            )

    def _fallback(self, key: K, reason: str) -> Datum[V]:
        good = self._last_good.get(self._last_good_slot(key))
        if good is None:
            return unknown_datum(reason, self._freshness_bound_seconds)
        return Datum(
            value=good.value,
            as_of=good.as_of,
            state=DatumState.STALE,
            reason=reason,
            freshness_bound_seconds=self._freshness_bound_seconds,
        )
