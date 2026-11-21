"""Land an interruption at a known point inside an awaited call.

A test waits for a marker the call produces once it reaches that point (a
hung seed, a stalled merge), then interrupts it there, instead of guessing a
duration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

from _poll_responsive import poll_responsive

T = TypeVar('T')


async def expire_deadline_once(call: Awaitable[T], started: Awaitable[object]) -> T:
    """Await *call* under an ``asyncio.timeout`` that expires as soon as *started* completes.

    Raises ``TimeoutError`` once the deadline lands.  If *started* itself
    raises, *call* is cancelled and awaited before that error propagates, so
    no task outlives the helper.
    """
    deadline = asyncio.timeout(None)

    async def under_deadline() -> T:
        async with deadline:
            return await call

    task = asyncio.ensure_future(under_deadline())
    try:
        await started
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    deadline.reschedule(asyncio.get_running_loop().time())
    return await task


async def wait_until_exists(path: Path) -> None:
    await poll_responsive(path.exists, label=f'{path} to exist', interval=0.02)
