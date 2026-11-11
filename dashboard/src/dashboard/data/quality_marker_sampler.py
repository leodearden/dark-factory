"""One marker tick per git-checkout project root: fold its new routing decisions, re-derive its
markers, and store the derivation in quality.db.

runs.db and git are read only here, in worker threads; nothing on a request path calls this
module. Each root resumes its stored routing fold from the fold's cursor, so the routing scan runs
once, and each tick folds at most MARKER_ROUTING_BATCH rows so a cold start progresses under any
budget. Design: plans/quality-dashboard-prd.md decisions 10-12 and 14.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from shared.project_id import project_id_for_root
from shared.project_layout import runs_db
from shared.quality_series.git_lines import GitTimeout, GitUnavailable, git_output
from shared.tasks_db_readonly import connect_ro

from dashboard.data.quality_markers import (
    CuratedMarkers,
    DerivedState,
    MarkerKind,
    PromptCoverage,
    RoutingFold,
    derive,
    fold_routing_decisions,
    in_scope,
    load_curated_markers,
)
from dashboard.data.quality_sampler import (
    describe_root_failure,
    git_checkouts,
    run_in_root_threads,
)
from dashboard.data.quality_store import (
    read_marker_state,
    record_marker_failure,
    write_marker_state,
)
from dashboard.data.utils import resolve_now

__all__ = [
    'FACTORY_REPO',
    'MARKER_GIT_TIMEOUT_SECONDS',
    'MARKER_PER_ROOT_BUDGET_SECONDS',
    'MARKER_ROUTING_BATCH',
    'ROLES_PY',
    'prompt_coverage',
    'refresh_markers',
]

logger = logging.getLogger(__name__)

MARKER_ROUTING_BATCH = 20_000
MARKER_PER_ROOT_BUDGET_SECONDS = 120.0
MARKER_GIT_TIMEOUT_SECONDS = 30.0

ROLES_PY = 'orchestrator/src/orchestrator/agents/roles.py'
"""Where every role's prompt lives in the factory repo."""

FACTORY_REPO = Path(__file__).resolve().parents[4]
"""This module's own checkout: the factory whose roles.py every project's orchestrator runs."""


def _counted_since(fold: RoutingFold, curated: CuratedMarkers) -> datetime | None:
    """The newest curated prompt marker in the project's scope, else its first routing decision."""
    prompts = [marker.at for marker in curated.markers
               if marker.kind is MarkerKind.PROMPT and in_scope(marker, fold.project_id)]
    return max(prompts, default=fold.first_at)


def _roles_py_landings(factory_repo: Path, since: datetime) -> int:
    stamp = since.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S +0000')
    stdout = git_output(
        factory_repo,
        ['rev-list', '--count', '--first-parent', f'--since={stamp}', 'main', '--', ROLES_PY],
        MARKER_GIT_TIMEOUT_SECONDS,
    )
    return int(stdout.strip())


def prompt_coverage(fold: RoutingFold, curated: CuratedMarkers, factory_repo: Path) -> PromptCoverage:
    """Written once a decision carries a prompt digest; until then, how many roles.py landings
    the curated prompt markers may have missed. git runs only in the second case."""
    if fold.digest_since is not None:
        return PromptCoverage(digest_since=fold.digest_since, roles_py_landings=None,
                              counted_since=None, reason=None)
    since = _counted_since(fold, curated)
    if since is None:
        return PromptCoverage(
            digest_since=None, roles_py_landings=None, counted_since=None,
            reason='no routing decision is folded yet to count roles.py landings from',
        )
    try:
        landings = _roles_py_landings(factory_repo, since)
    except (GitUnavailable, GitTimeout) as exc:
        return PromptCoverage(digest_since=None, roles_py_landings=None, counted_since=None,
                              reason=f'{type(exc).__name__}: {exc}')
    return PromptCoverage(digest_since=None, roles_py_landings=landings, counted_since=since,
                          reason=None)


def _refresh_root(
    root: Path, previous: RoutingFold, curated: CuratedMarkers, factory_repo: Path, batch: int,
    now: datetime,
) -> DerivedState:
    """*previous* advanced over the root's next routing rows, and every marker re-derived from it."""
    conn = connect_ro(runs_db(root))
    try:
        fold = fold_routing_decisions(conn, previous, batch=batch)
        derived = derive(fold, conn, snapshots=())
    finally:
        conn.close()
    return DerivedState(
        fold=fold, markers=derived.markers, skipped=derived.skipped,
        prompt_coverage=prompt_coverage(fold, curated, factory_repo), refreshed_at=now,
    )


async def _stored_fold(conn: aiosqlite.Connection, project_id: str) -> RoutingFold:
    stored = await read_marker_state(conn, project_id)
    if stored is None or stored.derived is None:
        return RoutingFold.empty(project_id)
    return stored.derived.fold


async def _store(
    conn: aiosqlite.Connection, project_id: str, now: datetime,
    result: DerivedState | BaseException, per_root_budget_s: float,
) -> None:
    """The root's new derivation, or a failure keeping its previous one; the caller commits."""
    if isinstance(result, BaseException):
        logger.warning('Quality marker refresh failed for %s', project_id, exc_info=result)
        await record_marker_failure(
            conn, project_id, attempted_at=now,
            failure=describe_root_failure(result, 'marker refresh', per_root_budget_s),
        )
        return
    await write_marker_state(conn, result)


async def _commit_or_roll_back(
    conn: aiosqlite.Connection, project_id: str, write: Awaitable[None],
) -> Exception | None:
    """Await *write* and commit it; on failure log, roll back, and hand back the exception."""
    try:
        await write
        await conn.commit()
    except Exception as exc:
        logger.warning('Failed to store the quality markers of %s', project_id, exc_info=True)
        with contextlib.suppress(Exception):
            await conn.rollback()
        return exc
    return None


async def refresh_markers(
    conn: aiosqlite.Connection,
    roots: Sequence[Path],
    *,
    now: datetime | None = None,
    curated_path: Path | None = None,
    factory_repo: Path = FACTORY_REPO,
    batch: int = MARKER_ROUTING_BATCH,
    per_root_budget_s: float = MARKER_PER_ROOT_BUDGET_SECONDS,
) -> None:
    """Advance and store the marker derivation of every root in *roots* that is a git checkout.

    Each root folds at most *batch* routing rows after its stored cursor and re-derives every
    marker. A root that raises or overruns *per_root_budget_s* records a failure naming why and
    keeps its previous derivation; each root commits on its own, and one whose write fails
    records that failure instead.
    """
    now_dt = resolve_now(now)
    checkouts = git_checkouts(roots)
    project_ids = [project_id_for_root(root) for root in checkouts]
    curated = await asyncio.to_thread(load_curated_markers, curated_path)
    previous = [await _stored_fold(conn, pid) for pid in project_ids]

    results: list[DerivedState | BaseException] = await run_in_root_threads(
        _refresh_root,
        [(root, fold, curated, factory_repo, batch, now_dt)
         for root, fold in zip(checkouts, previous, strict=True)],
        per_root_budget_s,
        thread_name_prefix='quality-markers',
    )

    for pid, result in zip(project_ids, results, strict=True):
        failure = await _commit_or_roll_back(
            conn, pid, _store(conn, pid, now_dt, result, per_root_budget_s),
        )
        if failure is not None:
            await _commit_or_roll_back(conn, pid, record_marker_failure(
                conn, pid, attempted_at=now_dt,
                failure=f'marker write failed: {type(failure).__name__}: {failure}',
            ))
