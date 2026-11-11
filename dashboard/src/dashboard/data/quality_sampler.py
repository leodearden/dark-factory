"""One quality tick: per git-checkout project root, measure new landings and write one value-or-gap sample.

git runs only here, inside a worker thread; nothing on a request path calls this module.
Design: plans/quality-dashboard-prd.md decision 14 and open question 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

import aiosqlite
from shared.project_id import project_id_for_root
from shared.project_layout import runs_db
from shared.quality_series.landings import GitCrossCheck, landings
from shared.quality_series.rows import LandingRow, rows_for
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

from dashboard.data.quality_store import (
    QualitySample,
    insert_sample,
    stored_cursor,
    upsert_landing_rows,
)
from dashboard.data.utils import resolve_now

__all__ = [
    'DISPATCH_ROW_SETTLE_SECONDS',
    'QUALITY_GIT_BUDGET_SECONDS',
    'QUALITY_PER_ROOT_BUDGET_SECONDS',
    'QUALITY_REV_LIST_TIMEOUT_SECONDS',
    'QUALITY_SAMPLE_INTERVAL_SECONDS',
    'collect_quality',
    'describe_root_failure',
    'describe_unreadable',
    'git_checkouts',
    'run_in_root_threads',
]

logger = logging.getLogger(__name__)

QUALITY_SAMPLE_INTERVAL_SECONDS = 600

# Set from the measurement in plans/quality-dashboard-prd.md open question 3.
QUALITY_GIT_BUDGET_SECONDS = 30.0
QUALITY_REV_LIST_TIMEOUT_SECONDS = 30.0
QUALITY_PER_ROOT_BUDGET_SECONDS = 180.0

# How long a landing's own task_results row may trail it: decision 14's correction in the PRD.
DISPATCH_ROW_SETTLE_SECONDS = 1800


@dataclass(frozen=True)
class _RootMeasurement:
    rows: tuple[LandingRow, ...]
    git_check: GitCrossCheck
    deferred: int


def _measure_root(
    root: Path, project_id: str, resume_cursor: str | None, git_budget_s: float,
) -> _RootMeasurement:
    """The landings after *resume_cursor*, measured as far as the git budget reaches."""
    conn = connect_ro(runs_db(root))
    try:
        batch = landings(
            conn, root, resume_cursor, project_id=project_id,
            git_timeout_s=QUALITY_REV_LIST_TIMEOUT_SECONDS,
        )
        rows = rows_for(batch.landings, conn, root, git_budget_s)
    finally:
        conn.close()
    return _RootMeasurement(tuple(rows), batch.git_check, len(batch.landings) - len(rows))


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


async def _write_sample(
    conn: aiosqlite.Connection,
    root: Path,
    project_id: str,
    now: datetime,
    result: _RootMeasurement | BaseException,
    per_root_budget_s: float,
) -> None:
    if isinstance(result, BaseException):
        logger.warning('Quality sampling failed for %s', root, exc_info=result)
        reason = describe_root_failure(result, 'quality sampling', per_root_budget_s)
        await insert_sample(conn, QualitySample.gap(project_id, now, reason))
        return
    await upsert_landing_rows(conn, result.rows, sampled_at=now)
    await insert_sample(conn, QualitySample.value(
        project_id, now,
        cursor_sha=await stored_cursor(conn, project_id),
        landings_sampled=len(result.rows),
        landings_deferred=result.deferred,
        git_check=result.git_check,
    ))


async def _commit_or_roll_back(
    conn: aiosqlite.Connection, root: Path, write: Awaitable[None],
) -> Exception | None:
    """Await *write* and commit it; on failure log, roll back, and hand back the exception."""
    try:
        await write
        await conn.commit()
    except Exception as exc:
        logger.warning('Failed to write the quality sample for %s', root, exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()
        return exc
    return None


async def collect_quality(
    conn: aiosqlite.Connection,
    roots: Sequence[Path],
    *,
    now: datetime | None = None,
    git_budget_s: float = QUALITY_GIT_BUDGET_SECONDS,
    per_root_budget_s: float = QUALITY_PER_ROOT_BUDGET_SECONDS,
) -> None:
    """Write one sample per root in *roots* that is a git checkout, and upsert its new rows.

    Each root resumes from its newest stored landing that has settled
    (DISPATCH_ROW_SETTLE_SECONDS before *now*), so a younger one is re-measured in place
    and a budget-truncated backfill resumes exactly where it stopped. The sample's cursor
    is the newest stored landing. A root whose measurement raises or overruns
    *per_root_budget_s* gets a gap naming why. Each root commits on its own, and a root
    whose write fails gets a gap naming that failure instead.
    """
    try:
        now_dt = resolve_now(now)
        checkouts = git_checkouts(roots)
        project_ids = [project_id_for_root(root) for root in checkouts]
        settled_by = now_dt - timedelta(seconds=DISPATCH_ROW_SETTLE_SECONDS)
        resume = [await stored_cursor(conn, pid, landed_by=settled_by) for pid in project_ids]

        results: list[_RootMeasurement | BaseException] = await run_in_root_threads(
            _measure_root,
            [(root, pid, cursor, git_budget_s)
             for root, pid, cursor in zip(checkouts, project_ids, resume, strict=True)],
            per_root_budget_s,
            thread_name_prefix='quality',
        )

        for root, pid, result in zip(checkouts, project_ids, results, strict=True):
            failure = await _commit_or_roll_back(
                conn, root, _write_sample(conn, root, pid, now_dt, result, per_root_budget_s),
            )
            if failure is not None:
                reason = f'sample write failed: {type(failure).__name__}: {failure}'
                await _commit_or_roll_back(
                    conn, root, insert_sample(conn, QualitySample.gap(pid, now_dt, reason)),
                )
    except Exception:
        with contextlib.suppress(Exception):
            await conn.rollback()
        raise
