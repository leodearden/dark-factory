"""One marker tick per git-checkout project root: fold its new routing decisions, re-derive its
markers, and store the derivation in quality.db.

runs.db and git are read only here, in worker threads; nothing on a request path calls this
module. Each root resumes its stored routing fold from the fold's cursor, so the routing scan runs
once, and each tick folds at most MARKER_ROUTING_BATCH rows so a cold start progresses under any
budget. Design: plans/quality-dashboard-prd.md decisions 10-12 and 14.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
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
    load_curated_markers,
)
from dashboard.data.quality_root_tick import (
    describe_root_failure,
    git_checkouts,
    run_in_root_threads,
    store_root,
)
from dashboard.data.quality_store import (
    read_routing_fold,
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

MARKER_ROUTING_BATCH = 500_000
"""Bounds one tick's fold so it stays inside the per-root budget, yet folds a cold start in one tick.
Measured 2026-11-12: the live df runs.db's 27,152 routing rows fold in 0.32 s, so 500,000 take
about 6 s."""
MARKER_PER_ROOT_BUDGET_SECONDS = 120.0
MARKER_GIT_TIMEOUT_SECONDS = 30.0

ROLES_PY = 'orchestrator/src/orchestrator/agents/roles.py'
"""Where every role's prompt lives in the factory repo."""

FACTORY_REPO = Path(__file__).resolve().parents[4]
"""This module's own checkout: the factory whose roles.py every project's orchestrator runs."""


def _counted_since(fold: RoutingFold, curated: CuratedMarkers) -> datetime | None:
    """The newest curated prompt marker in the project's scope, else its first routing decision."""
    prompts = [marker.at for marker in curated.for_project(fold.project_id)
               if marker.kind is MarkerKind.PROMPT]
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


async def _resume_fold(conn: aiosqlite.Connection, project_id: str) -> RoutingFold:
    """The stored fold to continue, else an empty one, including when the stored fold no longer
    decodes: every marker table is derived only, so folding from the start again loses nothing,
    and this tick's write replaces the rows that failed."""
    try:
        stored = await read_routing_fold(conn, project_id)
    except Exception:
        logger.warning('The stored routing fold of %s does not decode; folding it from the start',
                       project_id, exc_info=True)
        return RoutingFold.empty(project_id)
    return RoutingFold.empty(project_id) if stored is None else stored


async def _record_failure(
    conn: aiosqlite.Connection, project_id: str, now: datetime, failure: str,
) -> None:
    await record_marker_failure(conn, project_id, attempted_at=now, failure=failure)


async def _store(
    conn: aiosqlite.Connection, project_id: str, now: datetime,
    result: DerivedState | BaseException, per_root_budget_s: float,
) -> None:
    """The root's new derivation, or a failure keeping its previous one; the caller commits."""
    if isinstance(result, BaseException):
        logger.warning('Quality marker refresh failed for %s', project_id, exc_info=result)
        await _record_failure(conn, project_id, now,
                              describe_root_failure(result, 'marker refresh', per_root_budget_s))
        return
    await write_marker_state(conn, result)


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
    records that failure instead. A stored fold that no longer decodes is folded again from the
    start.
    """
    now_dt = resolve_now(now)
    checkouts = git_checkouts(roots)
    project_ids = [project_id_for_root(root) for root in checkouts]
    curated = await asyncio.to_thread(load_curated_markers, curated_path)
    previous = [await _resume_fold(conn, pid) for pid in project_ids]

    results: list[DerivedState | BaseException] = await run_in_root_threads(
        _refresh_root,
        [(root, fold, curated, factory_repo, batch, now_dt)
         for root, fold in zip(checkouts, previous, strict=True)],
        per_root_budget_s,
        thread_name_prefix='quality-markers',
    )

    for pid, result in zip(project_ids, results, strict=True):
        await store_root(
            conn, pid, _store(conn, pid, now_dt, result, per_root_budget_s),
            partial(_record_failure, conn, pid, now_dt), what='marker',
        )
