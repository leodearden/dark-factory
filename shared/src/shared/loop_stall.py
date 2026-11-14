"""Report an event loop that has stopped running callbacks, naming what blocks it.

An on-loop beat can only measure a stall once it is over, when the blocking call
has already returned, so it cannot say what blocked. A watchdog thread therefore
reads the loop thread's live stack while the stall is still happening, which is
the evidence py-spy gave for the task 7366 escalation-MCP stall. It only logs:
it never emits events and never files escalations.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

DEFAULT_WARN_AFTER_SECS = 2.0
DEFAULT_BEAT_INTERVAL_SECS = 0.25
DEFAULT_RESTACK_INTERVAL_SECS = 30.0

_WATCHDOG_THREAD_NAME = 'loop-stall-watchdog'

#: (the beat stamp a stall was reported for, when it was last reported)
_Reported = tuple[float, float]


def _require_positive(name: str, value: float) -> float:
    if not value > 0:
        raise ValueError(f'{name} must be > 0, got {value!r}')
    return value


class LoopStallMonitor:
    """A beat on the loop plus a watchdog thread that logs the loop's stack when it stalls."""

    def __init__(
        self,
        *,
        warn_after_secs: float = DEFAULT_WARN_AFTER_SECS,
        beat_interval_secs: float = DEFAULT_BEAT_INTERVAL_SECS,
        restack_interval_secs: float = DEFAULT_RESTACK_INTERVAL_SECS,
    ) -> None:
        self._warn_after = _require_positive('warn_after_secs', warn_after_secs)
        self._beat_interval = _require_positive('beat_interval_secs', beat_interval_secs)
        self._restack_interval = _require_positive('restack_interval_secs', restack_interval_secs)
        self._last_beat = time.monotonic()
        self._stopping = threading.Event()
        self._beat_task: asyncio.Task[None] | None = None
        self._watchdog: threading.Thread | None = None

    def start(self) -> None:
        """Begin watching the running loop. Call on the loop thread; a no-op while watching."""
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        loop = asyncio.get_running_loop()
        self._stopping = threading.Event()
        self._last_beat = time.monotonic()
        self._beat_task = loop.create_task(self._beat(self._stopping), name='loop-stall-beat')
        self._watchdog = threading.Thread(
            target=self._watch,
            args=(loop, threading.get_ident(), self._stopping),
            name=_WATCHDOG_THREAD_NAME,
            daemon=True,
        )
        self._watchdog.start()

    async def stop(self) -> None:
        """Stop watching and wait for the watchdog thread to end. Idempotent."""
        beat_task, watchdog = self._beat_task, self._watchdog
        self._beat_task = self._watchdog = None
        self._stopping.set()
        if beat_task is not None:
            beat_task.cancel()
            await asyncio.wait((beat_task,))
        if watchdog is not None:
            await asyncio.to_thread(watchdog.join, self._beat_interval * 4)

    async def _beat(self, stopping: threading.Event) -> None:
        try:
            while True:
                now = time.monotonic()
                overshoot = now - self._last_beat - self._beat_interval
                if overshoot >= self._warn_after:
                    logger.warning(
                        'loop_stall: event loop resumed after a %.1fs stall', overshoot,
                        extra={'stall_secs': overshoot},
                    )
                self._last_beat = now
                await asyncio.sleep(self._beat_interval)
        finally:
            # Without a beat the watchdog would read every later moment as a stall.
            stopping.set()

    def _watch(
        self, loop: asyncio.AbstractEventLoop, loop_thread_id: int, stopping: threading.Event,
    ) -> None:
        reported: _Reported | None = None
        while not stopping.wait(self._beat_interval):
            if loop.is_closed() or not loop.is_running():
                return
            try:
                reported = self._report_if_stalled(loop_thread_id, reported)
            except Exception:
                logger.warning('loop_stall: could not inspect the loop thread', exc_info=True)

    def _report_if_stalled(
        self, loop_thread_id: int, reported: _Reported | None,
    ) -> _Reported | None:
        beat = self._last_beat
        now = time.monotonic()
        stalled = now - beat - self._beat_interval
        if stalled < self._warn_after:
            return reported
        if reported is not None and reported[0] == beat and now - reported[1] < self._restack_interval:
            return reported
        frame = sys._current_frames().get(loop_thread_id)
        stack = ''.join(traceback.format_stack(frame)) if frame is not None else '<no frame>'
        logger.warning(
            'loop_stall: event loop blocked for %.1fs so far; loop thread stack:\n%s',
            stalled, stack,
            extra={'stall_secs': stalled, 'loop_stack': stack},
        )
        return beat, now
