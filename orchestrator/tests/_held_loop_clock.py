"""Hold the running loop's clock so an asyncio deadline counts from a state the test reached.

While held, I/O and ``call_soon`` callbacks still run, so a subprocess keeps
being driven; only timers stop coming due. Wait for the precondition off-loop
(``asyncio.to_thread``): ``asyncio.sleep`` is itself a timer and never completes
while held. A deadline set inside the block must be awaited inside it.

Contrast ``shared/src/shared/testing_virtual_clock.py``, which JUMPS the clock
whenever the loop idles instead of holding it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator


class HeldClock:
    """Reads the value *base* had at construction until released, then resumes from it."""

    def __init__(self, base: Callable[[], float]) -> None:
        self._base = base
        self._held_at = base()
        self._lag: float | None = None

    def __call__(self) -> float:
        if self._lag is None:
            return self._held_at
        return self._base() - self._lag

    def release(self) -> None:
        if self._lag is None:
            self._lag = self._base() - self._held_at


@contextlib.contextmanager
def held_loop_clock() -> Iterator[Callable[[], None]]:
    """Hold the running loop's ``time()`` for the block; yields the release callable."""
    loop = asyncio.get_running_loop()
    clock = HeldClock(loop.time)
    loop.time = clock
    try:
        yield clock.release
    finally:
        clock.release()
        del loop.time
