"""One snapshot tick per git-checkout project root: scan both of its snapshot homes through the
instrument, and store the scan in quality.db.

The instrument and git run only here, in worker threads; nothing on a request path calls this
module. Each root's stored scan is the next scan's cache, and a scan works newest input first under
a work deadline inside its per-root budget, so a cold start progresses under any budget and what it
leaves undone is stored as pending. Design: plans/quality-dashboard-prd.md decision 20.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import Path

import aiosqlite
from shared.project_id import project_id_for_root

from dashboard.data.quality_root_tick import (
    describe_root_failure,
    git_checkouts,
    run_in_root_threads,
    store_root,
)
from dashboard.data.quality_snapshots import (
    INSTRUMENT_SCRIPT,
    INSTRUMENT_TIMEOUT_SECONDS,
    RootScan,
    SnapshotCache,
    scan_root,
)
from dashboard.data.quality_store import (
    read_snapshot_cache,
    record_snapshot_failure,
    write_snapshot_scan,
)
from dashboard.data.utils import resolve_now

__all__ = [
    'SNAPSHOT_PER_ROOT_BUDGET_SECONDS',
    'SNAPSHOT_WORK_BUDGET_SECONDS',
    'refresh_snapshots',
]

logger = logging.getLogger(__name__)


def _work_budget_s(per_root_budget_s: float) -> float:
    """How long a scan may keep starting instrument runs: one run's timeout short of its per-root
    budget, so the last run it starts still ends inside that budget."""
    return max(per_root_budget_s - INSTRUMENT_TIMEOUT_SECONDS, 0.0)


SNAPSHOT_PER_ROOT_BUDGET_SECONDS = 180.0
SNAPSHOT_WORK_BUDGET_SECONDS = _work_budget_s(SNAPSHOT_PER_ROOT_BUDGET_SECONDS)
"""Measured 2026-11-14: a --summary takes 0.25 s and a --diff about 2.2 s, so a cold df start of
about 10 points, about 25 s of instrument runs, finishes in one tick."""


async def _resume_cache(conn: aiosqlite.Connection, project_id: str) -> SnapshotCache:
    """The stored scan to reuse, else an empty cache, including when the stored scan no longer
    decodes: every snapshot table is derived only, so scanning cold loses nothing, and this tick's
    write replaces the rows that failed."""
    try:
        return await read_snapshot_cache(conn, project_id)
    except Exception:
        logger.warning('The stored snapshot scan of %s does not decode; scanning it cold', project_id,
                       exc_info=True)
        return SnapshotCache.empty()


async def _record_failure(
    conn: aiosqlite.Connection, project_id: str, now: datetime, failure: str,
) -> None:
    await record_snapshot_failure(conn, project_id, now, failure)


async def _store(
    conn: aiosqlite.Connection, project_id: str, now: datetime,
    result: RootScan | BaseException, per_root_budget_s: float,
) -> None:
    """The root's new scan, or a failure keeping its previous one; the caller commits."""
    if isinstance(result, BaseException):
        logger.warning('Quality snapshot scan failed for %s', project_id, exc_info=result)
        await _record_failure(conn, project_id, now,
                              describe_root_failure(result, 'snapshot scan', per_root_budget_s))
        return
    await write_snapshot_scan(conn, project_id, result, now)


async def refresh_snapshots(
    conn: aiosqlite.Connection,
    roots: Sequence[Path],
    *,
    nightly_dir: Path,
    now: datetime | None = None,
    instrument: Path = INSTRUMENT_SCRIPT,
    per_root_budget_s: float = SNAPSHOT_PER_ROOT_BUDGET_SECONDS,
) -> None:
    """Scan and store both snapshot homes of every root in *roots* that is a git checkout.

    The homes are ``<root>/plans/quality-metrics/`` and ``<nightly_dir>/<project_id>/``. A root
    that raises or overruns *per_root_budget_s* records a failure naming why and keeps its previous
    scan; each root commits on its own, and one whose write fails records that failure instead.
    """
    now_dt = resolve_now(now)
    checkouts = git_checkouts(roots)
    project_ids = [project_id_for_root(root) for root in checkouts]
    caches = [await _resume_cache(conn, pid) for pid in project_ids]
    deadline = time.monotonic() + _work_budget_s(per_root_budget_s)

    results: list[RootScan | BaseException] = await run_in_root_threads(
        partial(scan_root, deadline=deadline, instrument=instrument),
        [(root, pid, nightly_dir, cache)
         for root, pid, cache in zip(checkouts, project_ids, caches, strict=True)],
        per_root_budget_s,
        thread_name_prefix='quality-snapshots',
    )

    for pid, result in zip(project_ids, results, strict=True):
        await store_root(
            conn, pid, _store(conn, pid, now_dt, result, per_root_budget_s),
            partial(_record_failure, conn, pid, now_dt), what='snapshot',
        )
