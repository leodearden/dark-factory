"""Primitives that carry a teardown to a confirmed end, despite a cancelled
caller and without mistaking an unknown for done.

First consumer: git_ops.py::GitOps.ephemeral_worktree.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar('T')


def path_confirmed_absent(path: Path) -> bool:
    """True only when *path* is known not to exist; an lstat that fails for any other reason is not absence."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


async def complete_despite_cancellation(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* to completion even if the awaiting task is cancelled.

    The caller's exit waits for *coro*; its cancellation is then re-raised,
    chained to *coro*'s exception if it had one. A cancellation of *coro*
    itself propagates as-is.
    """
    task = asyncio.ensure_future(coro)
    deferred: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            deferred = exc
    if deferred is not None:
        if not task.cancelled() and task.exception() is not None:
            raise deferred from task.exception()
        raise deferred
    return task.result()
