"""What every quality tick does per project root: pick the git checkouts, run each root's work on a
thread of its own under a per-root budget, say why a root produced nothing, and commit each root's
write on its own.

The landing-sample tick (dashboard/src/dashboard/data/quality_sampler.py) and the marker tick
(dashboard/src/dashboard/data/quality_marker_sampler.py) are both built on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import aiosqlite
from shared.tasks_db_readonly import TaskDbUnreadable

__all__ = [
    'describe_root_failure',
    'describe_unreadable',
    'git_checkouts',
    'run_in_root_threads',
    'store_root',
]

logger = logging.getLogger(__name__)


def git_checkouts(roots: Iterable[Path]) -> list[Path]:
    """The roots that are git checkouts, in order: the only roots a quality tick reads."""
    return [root for root in roots if (root / '.git').exists()]


_Result = TypeVar('_Result')


async def run_in_root_threads(
    work: Callable[..., _Result],
    per_root_args: Sequence[tuple[object, ...]],
    per_root_budget_s: float,
    *,
    thread_name_prefix: str,
) -> list[_Result | BaseException]:
    """``work(*args)`` for every root's args at once, each on a thread of its own, in order.

    Neither git nor a runs.db scan ever holds a request's thread. A root that raises gets its
    exception; one that overruns *per_root_budget_s* gets a TimeoutError and is abandoned to
    finish on its own thread.
    """
    if not per_root_args:
        return []
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=len(per_root_args), thread_name_prefix=thread_name_prefix)
    try:
        return await asyncio.gather(
            *(
                asyncio.wait_for(loop.run_in_executor(pool, work, *args), timeout=per_root_budget_s)
                for args in per_root_args
            ),
            return_exceptions=True,
        )
    finally:
        pool.shutdown(wait=False)


def describe_unreadable(exc: TaskDbUnreadable) -> str:
    """From its fields: the message's remedy is written for tasks.db, not runs.db."""
    return f'{type(exc).__name__}: {exc.path}: {exc.reason.value}'


def describe_root_failure(exc: BaseException, work: str, per_root_budget_s: float) -> str:
    """Why one root's *work* produced nothing, for a run_in_root_threads failure."""
    if isinstance(exc, TimeoutError):
        return f'{work} exceeded the {per_root_budget_s}s per-root budget'
    if isinstance(exc, TaskDbUnreadable):
        return describe_unreadable(exc)
    return f'{type(exc).__name__}: {exc}'


async def _commit_or_roll_back(
    conn: aiosqlite.Connection, write: Awaitable[None], what: str, root: object,
) -> Exception | None:
    """Await *write* and commit it; on failure log, roll back, and hand back the exception."""
    try:
        await write
        await conn.commit()
    except Exception as exc:
        logger.warning('Quality %s write failed for %s', what, root, exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()
        return exc
    return None


async def store_root(
    conn: aiosqlite.Connection,
    root: object,
    write: Awaitable[None],
    record_failure: Callable[[str], Awaitable[None]],
    *,
    what: str,
) -> None:
    """Commit one root's *write* on its own. If it fails, roll it back and commit
    ``record_failure(reason)`` instead, the reason starting ``'<what> write failed: '``."""
    failure = await _commit_or_roll_back(conn, write, what, root)
    if failure is not None:
        reason = f'{what} write failed: {type(failure).__name__}: {failure}'
        await _commit_or_roll_back(conn, record_failure(reason), what, root)
