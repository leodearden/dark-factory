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
from datetime import UTC, datetime
from pathlib import Path

from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)

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
        self._access: AtomicConnection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the SQLite database and tables.

        Idempotent — safe to call multiple times; subsequent calls are no-ops.
        """
        if self._access is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._data_dir / 'planned_episodes.db'
        conn = await connect_daemon(str(db_path))
        await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
        self._access = AtomicConnection(conn)
        async with self._access.write() as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_CREATE_INDEX)
        logger.info('PlannedEpisodeRegistry initialized at %s', db_path)

    def _require_access(self) -> AtomicConnection:
        if self._access is None:
            raise RuntimeError('PlannedEpisodeRegistry not initialized — call initialize() first')
        return self._access

    async def close(self) -> None:
        """Close the database connection."""
        if self._access is not None:
            await self._access.close()
            self._access = None
        logger.info('PlannedEpisodeRegistry closed')

    async def checkpoint(self) -> CheckpointResult:
        """``PRAGMA wal_checkpoint(TRUNCATE)`` → ``(busy, log, checkpointed)``."""
        if self._access is None:
            return CheckpointResult.unavailable()
        return await self._access.checkpoint()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def register(self, episode_uuid: str, project_id: str) -> None:
        """Register an episode as planned (idempotent).

        Uses INSERT OR IGNORE to handle duplicate calls without raising.
        """
        created_at = datetime.now(UTC).isoformat()
        async with self._require_access().write() as db:
            await db.execute(
                'INSERT OR IGNORE INTO planned_episodes (episode_uuid, project_id, created_at) '
                'VALUES (?, ?, ?)',
                (episode_uuid, project_id, created_at),
            )
        logger.debug('Registered planned episode %s for project %s', episode_uuid, project_id)

    async def is_planned(self, episode_uuid: str) -> bool:
        """Return True if the episode is registered as planned."""
        row = await self._require_access().read_one(
            'SELECT 1 FROM planned_episodes WHERE episode_uuid = ? LIMIT 1',
            (episode_uuid,),
        )
        return row is not None

    async def get_planned_uuids(self, project_id: str) -> set[str]:
        """Return the set of planned episode UUIDs for a project."""
        rows = await self._require_access().read_all(
            'SELECT episode_uuid FROM planned_episodes WHERE project_id = ?',
            (project_id,),
        )
        return {row[0] for row in rows}

    async def promote(self, episode_uuid: str) -> None:
        """Remove an episode from the planned registry (promote to real)."""
        async with self._require_access().write() as db:
            await db.execute(
                'DELETE FROM planned_episodes WHERE episode_uuid = ?',
                (episode_uuid,),
            )
        logger.debug('Promoted episode %s (removed from planned registry)', episode_uuid)

    async def are_all_planned(self, episode_uuids: list[str]) -> bool:
        """Return True iff the list is non-empty and ALL uuids are planned.

        An empty list returns False — there are no episodes to call planned.
        A mixed list (some planned, some not) returns False.
        """
        if not episode_uuids:
            return False
        placeholders = ','.join('?' * len(episode_uuids))
        row = await self._require_access().read_one(
            f'SELECT COUNT(*) FROM planned_episodes WHERE episode_uuid IN ({placeholders})',
            episode_uuids,
        )
        count = row[0] if row else 0
        return count == len(episode_uuids)
