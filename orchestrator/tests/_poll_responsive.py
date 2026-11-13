"""Poll a predicate to truthiness under wait_responsive's loop-responsive deadline.

This is Rule C's named remedy for a poll written as an ``asyncio.timeout``
block or a clock-deadline loop (fused-memory/scripts/check_bare_magicmock_config.py,
Rule C section).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TypeVar

from _orch_helpers import MERGE_RESULT_TIMEOUT, wait_responsive

_PollT = TypeVar('_PollT')


async def poll_responsive(
    predicate: Callable[[], _PollT],
    /,
    *,
    timeout: float = MERGE_RESULT_TIMEOUT,
    label: str,
    interval: float = 0.01,
    max_wall_s: float | None = None,
) -> _PollT:
    """Return *predicate*'s first truthy value, checking it before each sleep.

    The deadline accounting, the wall-clock bound and the COUPLED
    timeout-mark obligation are wait_responsive's: see
    ``_orch_helpers.py::wait_responsive``.

    *predicate* must be synchronous: an awaitable result raises TypeError
    naming *label*, because a coroutine object is always truthy and would
    end the poll at once with the condition unchecked.
    """

    def _checked(value: _PollT) -> _PollT:
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise TypeError(
                f'poll_responsive predicate must be synchronous; {label!r} got an '
                f'awaitable ({type(value).__name__})'
            )
        return value

    async def _first_truthy() -> _PollT:
        while not (value := _checked(predicate())):
            await asyncio.sleep(interval)
        return value

    return await wait_responsive(
        _first_truthy(), timeout=timeout, label=label, max_wall_s=max_wall_s
    )
