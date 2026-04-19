"""Watchdog for the WP-B EventQueue drainer.

Periodically inspects ``EventQueue.stats()``. If the drainer hasn't committed
in ``stall_threshold_seconds`` AND there are events still in the queue, emits
a structured ERROR log with diagnostics so the SQLite-lock condition is
visible (instead of rotting silently as it did before WP-A's incident).

WP-D will subscribe to the optional callback to escalate at L1.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

WedgeCallback = Callable[[dict], Awaitable[None]] | Callable[[dict], None]


@runtime_checkable
class WatchdogObservable(Protocol):
    """Minimal interface required by :class:`SqliteWatchdog`.

    Both :class:`~fused_memory.reconciliation.event_queue.EventQueue` and
    test doubles satisfy this protocol.
    """

    def stats(self) -> dict: ...
    def recent_ops(self) -> list[dict]: ...


class SqliteWatchdog:
    """Background watchdog over an :class:`~fused_memory.reconciliation.event_queue.EventQueue` drainer.

    Wedge condition (both must hold):
      ``now - last_commit_ts > stall_threshold_seconds`` AND
      ``queue_depth + retry_in_flight > 0``

    The second clause is "there is outstanding work the drainer hasn't
    committed yet". It covers both queue-depth backlog and the common case
    where the drainer has popped one event off the queue and is stuck
    retrying it (``queue_depth`` goes to 0 but the event is *in flight*).

    If ``last_commit_ts`` is ``None`` (drainer hasn't committed anything yet),
    the wall-clock anchor is the watchdog start time — so a never-committing
    drainer is correctly flagged once the threshold elapses.
    """

    def __init__(
        self,
        event_queue: WatchdogObservable,
        *,
        check_interval_seconds: float = 30.0,
        stall_threshold_seconds: float = 120.0,
        wedge_callback: WedgeCallback | None = None,
        rearm_after_seconds: float = 600.0,
    ):
        self._event_queue = event_queue
        self._check_interval = check_interval_seconds
        self._stall_threshold = stall_threshold_seconds
        self._wedge_callback = wedge_callback
        # Don't spam: once we've fired, wait at least rearm_after_seconds
        # before logging the same wedge again. Reset when the drainer recovers
        # (commits something) so a future wedge fires immediately.
        self._rearm_after = rearm_after_seconds
        self._task: asyncio.Task | None = None
        self._closed = False
        self._wedge_active = False
        self._last_wedge_log_ts: float = 0.0
        self._anchor_ts = time.time()

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError('SqliteWatchdog already started')
        self._anchor_ts = time.time()
        self._task = asyncio.create_task(self._loop(), name='sqlite-watchdog')
        logger.info(
            'SqliteWatchdog started (check_interval=%.1fs, stall_threshold=%.1fs)',
            self._check_interval, self._stall_threshold,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                'SqliteWatchdog: unexpected exception awaiting cancelled task',
                exc_info=True,
            )
            raise
        self._task = None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._check_interval)
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception('SqliteWatchdog: tick failed')
        except asyncio.CancelledError:
            return

    async def _tick(self) -> None:
        stats = self._event_queue.stats()
        queue_depth = stats.get('queue_depth') or 0
        retry_in_flight = stats.get('retry_in_flight') or 0
        outstanding = queue_depth + retry_in_flight
        last_commit_ts = stats.get('last_commit_ts')
        anchor = last_commit_ts if last_commit_ts is not None else self._anchor_ts
        now = time.time()
        stale_for = now - anchor

        wedged = outstanding > 0 and stale_for > self._stall_threshold

        if not wedged:
            # Recovery — clear the latch so the next wedge fires immediately.
            if self._wedge_active and last_commit_ts is not None:
                logger.info(
                    'SqliteWatchdog: drainer recovered '
                    '(queue_depth=%d, last_commit %.1fs ago)',
                    queue_depth, stale_for,
                )
            self._wedge_active = False
            return

        # Wedged. Rate-limit ERROR log to once per rearm_after window.
        should_log = (
            not self._wedge_active
            or (now - self._last_wedge_log_ts) >= self._rearm_after
        )
        self._wedge_active = True
        if not should_log:
            return
        self._last_wedge_log_ts = now

        diagnostic = self._build_diagnostic(stats, stale_for)
        logger.error(
            'SqliteWatchdog: drainer wedged — no commit for %.1fs with '
            'queue_depth=%d retry_in_flight=%d. Diagnostic: %s',
            stale_for, queue_depth, retry_in_flight, diagnostic,
        )
        if self._wedge_callback is not None:
            try:
                result = self._wedge_callback(diagnostic)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception('SqliteWatchdog: wedge_callback raised')

    def _build_diagnostic(self, stats: dict, stale_for: float) -> dict:
        try:
            recent_ops = self._event_queue.recent_ops()
        except Exception:
            recent_ops = []

        try:
            tasks = asyncio.all_tasks()
            task_names = sorted({t.get_name() for t in tasks})
            task_count = len(tasks)
        except Exception:
            task_names = []
            task_count = 0

        return {
            'stale_for_seconds': round(stale_for, 1),
            'queue_depth': stats.get('queue_depth'),
            'queue_capacity': stats.get('queue_capacity'),
            'retry_in_flight': stats.get('retry_in_flight'),
            'events_committed': stats.get('events_committed'),
            'overflow_drops': stats.get('overflow_drops'),
            'dead_letters': stats.get('dead_letters'),
            'drainer_running': stats.get('drainer_running'),
            'last_commit_ts': stats.get('last_commit_ts'),
            'recent_ops': recent_ops,
            'asyncio_task_count': task_count,
            'asyncio_task_names': task_names[:30],
        }
