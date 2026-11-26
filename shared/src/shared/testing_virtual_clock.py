"""Run an asyncio scenario on a virtual clock that host stalls cannot move.

The loop clock advances only when the loop is idle, and by exactly the timeout
it would otherwise have slept. A host stall (CPU starvation under xdist load, a
GC pause) therefore cannot expire a deadline, while timers still fire in exact
loop-time order.

Scope: scenarios whose every wait is in-loop (httpx.MockTransport handlers,
asyncio primitives). Real sockets and threads race the jump; executor work
(run_in_executor, asyncio.to_thread, getaddrinfo) is rejected with RuntimeError.

The dashboard and orchestrator suites both run scenarios on this clock, and
neither can import the other's test tree, so it lives HERE, in the shipped
``shared`` package — the same convention as :mod:`shared.testing_stdin` and
:mod:`shared.testing_streams` — rather than being copied into each suite (INV-5).

Pure stdlib, and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import asyncio
import functools
import selectors
from collections.abc import Callable, Coroutine
from typing import Any, NoReturn, ParamSpec, TypeVar

__all__ = ['run_on_virtual_clock', 'virtual_clock_test']

T = TypeVar('T')
P = ParamSpec('P')


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0


class _IdleJumpSelector(selectors.DefaultSelector):
    def __init__(self, clock: _VirtualClock) -> None:
        super().__init__()
        self._clock = clock

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        if timeout is not None and timeout > 0:
            self._clock.now += timeout
            return super().select(0)
        return super().select(timeout)


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._clock = _VirtualClock()
        super().__init__(selector=_IdleJumpSelector(self._clock))

    def time(self) -> float:
        return self._clock.now

    def run_in_executor(self, executor: Any, func: Any, *args: Any) -> NoReturn:
        raise RuntimeError(
            'virtual-clock loop: executor work runs on a real thread that the idle '
            'jump races, so its deadlines would expire spuriously; keep the '
            'scenario in-loop (e.g. httpx.MockTransport)'
        )


def run_on_virtual_clock(scenario: Coroutine[Any, Any, T]) -> T:
    """Run *scenario* to completion on a fresh virtual-clock loop and return its result.

    Tasks still pending when it returns are cancelled before this returns.
    """
    with asyncio.Runner(loop_factory=_VirtualClockLoop) as runner:
        return runner.run(scenario)


def virtual_clock_test(test: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    """Turn an ``async def`` test into a sync test whose body runs on :func:`run_on_virtual_clock`.

    pytest still resolves fixtures from the test's own signature, and
    pytest-asyncio sees no coroutine function to claim, so the test needs no
    ``@pytest.mark.asyncio``.
    """

    @functools.wraps(test)
    def run_test(*args: P.args, **kwargs: P.kwargs) -> None:
        run_on_virtual_clock(test(*args, **kwargs))

    return run_test
