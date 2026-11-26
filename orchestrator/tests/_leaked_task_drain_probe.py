"""The test that test_leaked_task_drain.py runs in a nested pytest session.

Not collected by the suite (no ``test_`` prefix): it deliberately leaves a task
behind that survives ONE cancellation, the shape asyncio's subprocess transport
takes when it is cancelled between spawning a child and connecting its pipes.
"""
import asyncio

import pytest

_KEEP: list[asyncio.Task] = []


async def _survives_one_cancel() -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_leaves_a_task_that_survives_one_cancel() -> None:
    _KEEP.append(asyncio.get_running_loop().create_task(
        _survives_one_cancel(), name='survives-one-cancel',
    ))
    await asyncio.sleep(0)
