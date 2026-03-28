"""Maintenance utility for deleting stale Graphiti edges via FalkorDB temporal-range query."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from fused_memory.maintenance._utils import maintenance_service

logger = logging.getLogger(__name__)

# Default stale window (2026-03-22 reconciliation incident)
_DEFAULT_START = '2026-03-22T17:50:00+00:00'
_DEFAULT_END = '2026-03-22T18:15:00+00:00'


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""

    edges_found: int = 0
    edges_deleted: int = 0
    edge_details: list[dict] = field(default_factory=list)


class CleanupManager:
    """Finds and deletes stale Graphiti edges using temporal-range Cypher queries.

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    async def find_stale_edges(self, start: str, end: str) -> list[dict]:
        """Return edges whose valid_at falls within [start, end].

        Args:
            start: ISO 8601 lower bound (inclusive).
            end: ISO 8601 upper bound (inclusive).

        Returns:
            List of edge dicts (uuid, fact, name, valid_at, invalid_at).
        """
        edges = await self.backend.query_edges_by_time_range(start=start, end=end)
        logger.info(f'Found {len(edges)} edge(s) in range [{start}, {end}]')
        return edges

    async def cleanup(
        self,
        start: str,
        end: str,
        dry_run: bool = False,
    ) -> CleanupResult:
        """Find stale edges and optionally delete them.

        Args:
            start: ISO 8601 lower bound (inclusive).
            end: ISO 8601 upper bound (inclusive).
            dry_run: When True, find but do not delete. Defaults to False.

        Returns:
            CleanupResult with edges_found, edges_deleted, and edge_details.
        """
        edges = await self.find_stale_edges(start=start, end=end)
        result = CleanupResult(
            edges_found=len(edges),
            edge_details=edges,
        )

        if dry_run:
            logger.info(
                f'Dry run — found {len(edges)} edge(s), skipping deletion. '
                f'Re-run without --dry-run to delete.'
            )
            return result

        if not edges:
            logger.info('No stale edges found; nothing to delete.')
            return result

        uuids = [e['uuid'] for e in edges]
        deleted = await self.backend.bulk_remove_edges(uuids)
        result.edges_deleted = deleted
        logger.info(f'Deleted {deleted} edge(s)')
        return result


async def run_cleanup(
    start: str = _DEFAULT_START,
    end: str = _DEFAULT_END,
    config_path: str | None = None,
    dry_run: bool = False,
) -> CleanupResult:
    """Load config, initialize service, run cleanup, close resources.

    This is the CLI-callable entrypoint.  It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates and initialises a MemoryService (which owns GraphitiBackend).
    3. Creates a CleanupManager and runs cleanup().
    4. Closes all resources in a finally block.

    Args:
        start: ISO 8601 lower bound for the stale window (inclusive).
        end: ISO 8601 upper bound for the stale window (inclusive).
        config_path: Optional path to the YAML config file.  When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
        dry_run: When True, find but do not delete edges. Defaults to False.

    Returns:
        CleanupResult with edges_found, edges_deleted, and edge_details.
    """
    async with maintenance_service(config_path) as (config, service):
        manager = CleanupManager(backend=service.graphiti)
        result = await manager.cleanup(start=start, end=end, dry_run=dry_run)
        logger.info(
            f'run_cleanup complete: found={result.edges_found}, '
            f'deleted={result.edges_deleted}, dry_run={dry_run}'
        )
        return result


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Delete stale Graphiti edges via FalkorDB temporal-range query'
    )
    parser.add_argument(
        '--start',
        default=_DEFAULT_START,
        help=(
            f'ISO 8601 lower bound for the stale window (default: {_DEFAULT_START})'
        ),
    )
    parser.add_argument(
        '--end',
        default=_DEFAULT_END,
        help=(
            f'ISO 8601 upper bound for the stale window (default: {_DEFAULT_END})'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Find but do not delete edges. Review output before re-running without this flag.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    args = parser.parse_args()
    asyncio.run(
        run_cleanup(
            start=args.start,
            end=args.end,
            config_path=args.config,
            dry_run=args.dry_run,
        )
    )
