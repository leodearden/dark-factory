"""Shared helpers for the remote verify host-health tests (task 6670)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from _orch_helpers import wait_responsive

from orchestrator.verify_runner import RunnerUnavailable


def transport_down(
    reason: str = 'ssh: connect to host laptop.invalid: Connection timed out',
) -> RunnerUnavailable:
    """The RunnerUnavailable an unreachable host's ssh transport raises."""
    return RunnerUnavailable(reason, kind='transport')


async def until(predicate: Callable[[], bool], *, what: str) -> None:
    """Poll *predicate* until it holds, failing by *what* on the responsive-time deadline."""
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await wait_responsive(poll(), label=what)
