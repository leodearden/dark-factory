"""A clock that moves only when a test tells it to.

Inject it wherever the code under test reads or spends time: it is a
``Callable[[], float]`` time provider, a ``sleep`` callable and an awaitable
``async_sleep``.  It is the in-process alternative to a real sleep, which the
timing-dependence lint rejects.  To run a whole asyncio loop on virtual time,
use shared/src/shared/testing_virtual_clock.py instead.

Pure stdlib, and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import asyncio

__all__ = ['FakeClock']


class FakeClock:
    """Readings that start at *start* and only ever move forward, by ``advance`` or a sleep."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._sleeps: list[float] = []

    @property
    def now(self) -> float:
        return self._now

    @property
    def sleeps(self) -> tuple[float, ...]:
        """Every duration slept, in order."""
        return tuple(self._sleeps)

    def __call__(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._now

    def time(self) -> float:
        return self._now

    def _require_forward(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(
                f'a FakeClock only moves forward: seconds={seconds!r} at now={self._now!r}'
            )

    def advance(self, seconds: float) -> None:
        self._require_forward(seconds)
        self._now += seconds

    def sleep(self, seconds: float) -> None:
        self._require_forward(seconds)
        self._sleeps.append(seconds)
        self._now += seconds

    async def async_sleep(self, seconds: float) -> None:
        """``sleep``, then one yield to the loop so a ready task runs before it returns."""
        self.sleep(seconds)
        await asyncio.sleep(0)
