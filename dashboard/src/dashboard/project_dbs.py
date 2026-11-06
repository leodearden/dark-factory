"""Resolve the per-project SQLite connections a consumer needs.

The dashboard reads the same relative DB path out of every project it knows
about. These helpers map ``DashboardConfig.all_project_roots`` to per-root
paths — ``runs.db``, ``burndown.db`` and the task store's ``tasks.db`` — and
``_cost_sources`` adds the fused-memory curator's own invocation ledger, which
lives outside any project tree, to the cost view's runs.db list.

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

These six names are this module's entire interface: ``TASKS_DB_REL_PATH``
(read by ``api/invariants.py``) and five connection helpers. The helpers
carry a leading underscore from when they were private to ``app.py``. Read
their underscore as vestigial, not as a private-use signal — the move that
created this module was a pure extraction, with renaming outside its scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import aiosqlite

from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool

# The doubled `tasks` directory is load-bearing: a bare `.taskmaster/tasks.db` is a 0-byte decoy.
TASKS_DB_REL_PATH: Final = Path('.taskmaster/tasks/tasks.db')


async def _project_scoped_dbs(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[aiosqlite.Connection | None]:
    """Return DB connections for a project-scoped file across all known roots."""
    return [await pool.get(root / rel_path) for root in config.all_project_roots]


async def _project_scoped_dbs_labeled(
    config: DashboardConfig,
    pool: DbPool,
    rel_path: Path,
) -> list[tuple[str, aiosqlite.Connection | None]]:
    """Return labeled (str(root), connection|None) pairs across all known project roots."""
    return [(str(root), await pool.get(root / rel_path)) for root in config.all_project_roots]


async def _cost_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project runs.db files (costs and performance)."""
    return await _project_scoped_dbs(config, pool, Path('data/orchestrator/runs.db'))


async def _cost_sources(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Every invocation ledger the cost view reads: each project's runs.db plus the fused-memory curator's."""
    return [*await _cost_dbs(config, pool), await pool.get(config.curator_events_db)]


async def _burndown_dbs(
    config: DashboardConfig,
    pool: DbPool,
) -> list[aiosqlite.Connection | None]:
    """Connections for all known project burndown.db files."""
    return await _project_scoped_dbs(config, pool, Path('data/burndown/burndown.db'))
