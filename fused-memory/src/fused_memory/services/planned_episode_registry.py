"""SQLite-backed registry for tracking planned/aspirational episode UUIDs.

When content is ingested via add_episode(temporal_context='planning'), the
resulting episode UUID is registered here.  During search, edges whose entire
provenance (all contributing episodes) is from the planned registry are excluded
by default — preventing PRD/aspirational facts from appearing as current truth.

Promotion removes an episode from the planned registry, making its derived edges
visible in normal searches.  This happens when a task is marked done.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS planned_episodes (
    episode_uuid TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (episode_uuid)
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_pe_project
    ON planned_episodes (project_id);
"""


class PlannedEpisodeRegistry:
    """Tracks episode UUIDs that were ingested with temporal_context='planning'.

    Lifecycle::

        registry = PlannedEpisodeRegistry(data_dir=cfg.queue.data_dir)
        await registry.initialize()
        ...
        await registry.close()
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the SQLite database and tables."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._data_dir / 'planned_episodes.db'
        self._db = await aiosqlite.connect(str(db_path))
        await self._db.execute('PRAGMA journal_mode=WAL')
        await self._db.execute('PRAGMA busy_timeout=5000')
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()
        logger.info('PlannedEpisodeRegistry initialized at %s', db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        logger.info('PlannedEpisodeRegistry closed')

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def register(self, episode_uuid: str, project_id: str) -> None:
        """Register an episode as planned (idempotent)."""
        raise NotImplementedError

    async def is_planned(self, episode_uuid: str) -> bool:
        """Return True if the episode is registered as planned."""
        raise NotImplementedError

    async def get_planned_uuids(self, project_id: str) -> set[str]:
        """Return the set of planned episode UUIDs for a project."""
        raise NotImplementedError

    async def promote(self, episode_uuid: str) -> None:
        """Remove an episode from the planned registry (promote to real)."""
        raise NotImplementedError

    async def are_all_planned(self, episode_uuids: list[str]) -> bool:
        """Return True iff the list is non-empty and ALL uuids are planned."""
        raise NotImplementedError
