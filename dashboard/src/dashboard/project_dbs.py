"""Resolve the per-project SQLite connections a consumer needs.

The dashboard reads the same relative DB path out of every project it knows
about: ``config.project_root`` first, then each of
``config.known_project_roots``, de-duplicated so a root named twice is
opened once. These helpers are that walk, plus the two concrete paths —
``runs.db`` and ``burndown.db`` — that more than one consumer asks for.

This is its own module rather than part of ``app.py`` because its callers
land on both sides of the route/loop boundary:
``_project_scoped_dbs_labeled`` is called by ``loops.py::_metrics_loop``
*and* ``api/merge_queue.py::api_merge_queue``, and ``_project_scoped_dbs``
by ``_cost_dbs`` (whose callers stay in ``app.py``) *and* ``_burndown_dbs``
(whose caller moved to ``api/burndown.py``). Importing back into ``app.py``
for these would be a circular import.

A missing DB file is not an error here: ``DbPool.get`` yields ``None`` for
it, so a consumer fans out over whatever exists and a project that has not
produced a given DB yet simply contributes nothing.

All four names carry a leading underscore from when they were private to
``app.py``, and all four are now this module's entire interface. Read the
underscore as vestigial, not as a private-use signal — the move that
created this module was a pure extraction, with renaming outside its scope.

This is the canonical home for that primary-root-first, de-duplicated walk
but not yet its only copy: ``app.py::_performance_resources``,
``app.py::_analytics_project_dirs``,
``dashboard/data/escalations.py::build_escalation_queues`` and
``dashboard/data/active_tasks.py::_all_project_roots`` each still hand-roll
the same iteration, differing only in what they derive per root. Rebuilding
them on one shared root iterator is a logic change the extraction
deliberately did not make; it is filed as a follow-up rather than dropped.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool


async def _project_scoped_dbs(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[aiosqlite.Connection | None]:
    """Return DB connections for a project-scoped file across all known roots."""
    seen: set[Path] = {config.project_root}
    paths: list[Path] = [config.project_root / rel_path]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            paths.append(root / rel_path)
    return [await pool.get(p) for p in paths]


async def _project_scoped_dbs_labeled(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[tuple[str, aiosqlite.Connection | None]]:
    """Return labeled (str(root), connection|None) pairs across all known project roots."""
    seen: set[Path] = {config.project_root}
    roots: list[Path] = [config.project_root]
    for root in config.known_project_roots:
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return [(str(root), await pool.get(root / rel_path)) for root in roots]


async def _cost_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project runs.db files (costs and performance)."""
    return await _project_scoped_dbs(config, pool, Path('data/orchestrator/runs.db'))


async def _burndown_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project burndown.db files."""
    return await _project_scoped_dbs(config, pool, Path('data/burndown/burndown.db'))
