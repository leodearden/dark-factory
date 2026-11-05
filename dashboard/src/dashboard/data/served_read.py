"""A read served as a ``Datum``: cached, single-flight, waited on for a bounded time, and
falling back to its last good value (the pattern of ``plans/dashboard-prd-lanes-prd.md`` §4.2).

No clock read: every instant is the caller's injected *now*. The TTL is
:class:`~dashboard.data.mcp_fanout.TTLCache`'s own monotonic one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Hashable
from datetime import datetime
from typing import Generic, TypeVar

from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum
from dashboard.data.mcp_fanout import TTLCache

logger = logging.getLogger(__name__)

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


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
    """

    def __init__(
        self, *, ttl_seconds: float | Callable[[], float], freshness_bound_seconds: int,
    ) -> None:
        self._cache: TTLCache[Datum[V], K] = TTLCache(ttl_seconds)
        self._last_good: dict[K, Datum[V]] = {}
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
        refresh = asyncio.ensure_future(self._refresh(key, read, self._generation))
        try:
            outcome = await asyncio.wait_for(asyncio.shield(refresh), wait_seconds)
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
            self._last_good[key] = outcome
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
        good = self._last_good.get(key)
        if good is None:
            return unknown_datum(reason, self._freshness_bound_seconds)
        return Datum(
            value=good.value,
            as_of=good.as_of,
            state=DatumState.STALE,
            reason=reason,
            freshness_bound_seconds=self._freshness_bound_seconds,
        )
