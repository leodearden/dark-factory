"""Park a coroutine inside one awaited call, so a test can act while it waits there."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

ENTERED_TIMEOUT_SECONDS = 30.0


def block_first_call(monkeypatch: pytest.MonkeyPatch, obj: object, name: str) -> asyncio.Event:
    """Make the first call of ``obj.<name>`` set the returned event and never return."""
    original = getattr(obj, name)
    entered = asyncio.Event()

    async def blocking_first_call(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(obj, name, blocking_first_call)
    return entered


async def wait_until_entered(
    entered: asyncio.Event,
    task: asyncio.Future[Any],
    *,
    timeout: float = ENTERED_TIMEOUT_SECONDS,
) -> None:
    """Return once ``entered`` is set; fail if ``task`` ends or ``timeout`` passes first."""
    entered_wait = asyncio.ensure_future(entered.wait())
    try:
        await asyncio.wait(
            {entered_wait, task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        entered_wait.cancel()
    if not entered.is_set():
        pytest.fail(f'the blocked call was not reached within {timeout}s; task: {task!r}')
