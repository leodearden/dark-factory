"""Maintenance utility for batch-rebuilding Graphiti entity summaries."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from fused_memory.maintenance._utils import maintenance_service

logger = logging.getLogger(__name__)


@dataclass
class RebuildResult:
    """Result of a rebuild_entity_summaries operation."""

    total_entities: int = 0
    stale_entities: int = 0
    rebuilt: int = 0
    skipped: int = 0
    errors: int = 0
    details: list[dict] = field(default_factory=list)


class RebuildSummariesManager:
    """Discovers and rebuilds stale Graphiti entity summaries.

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
        group_id: FalkorDB graph (project) to operate on.
    """

    def __init__(self, backend, group_id: str = 'dark_factory') -> None:
        self.backend = backend
        self.group_id = group_id

    async def run(
        self,
        force: bool = False,
        dry_run: bool = False,
    ) -> RebuildResult:
        """Detect stale entity summaries and optionally rebuild them.

        Args:
            force: When True, rebuilds all entities regardless of staleness.
            dry_run: When True, detects stale entities but does not rebuild.
                     Defaults to False.

        Returns:
            RebuildResult with aggregate counts and per-entity details.
        """
        raw = await self.backend.rebuild_entity_summaries(
            group_id=self.group_id,
            force=force,
            dry_run=dry_run,
        )
        return RebuildResult(
            total_entities=raw.get('total_entities', 0),
            stale_entities=raw.get('stale_entities', 0),
            rebuilt=raw.get('rebuilt', 0),
            skipped=raw.get('skipped', 0),
            errors=raw.get('errors', 0),
            details=raw.get('details', []),
        )


async def run_rebuild_summaries(
    config_path: str | None = None,
    group_id: str = 'dark_factory',
    force: bool = False,
    dry_run: bool = False,
) -> RebuildResult:
    """Load config, initialize service, run rebuild, close resources.

    This is the CLI-callable entrypoint.  It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates and initialises a MemoryService (which owns GraphitiBackend).
    3. Creates a RebuildSummariesManager and runs it.
    4. Closes all resources in a finally block.

    Args:
        config_path: Optional path to the YAML config file.  When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
        group_id: FalkorDB graph (project) to operate on.
        force: Rebuild every entity regardless of staleness.
        dry_run: Detect but do not rebuild. Defaults to False.

    Returns:
        RebuildResult with total_entities, stale_entities, rebuilt, skipped,
        errors, and details.
    """
    async with maintenance_service(config_path) as (config, service):
        manager = RebuildSummariesManager(backend=service.graphiti, group_id=group_id)
        result = await manager.run(force=force, dry_run=dry_run)
        logger.info(
            'run_rebuild_summaries complete: total=%d stale=%d rebuilt=%d '
            'skipped=%d errors=%d dry_run=%s',
            result.total_entities, result.stale_entities, result.rebuilt,
            result.skipped, result.errors, dry_run,
        )
        return result


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Batch-rebuild Graphiti entity summaries from valid edges'
    )
    parser.add_argument(
        '--group-id',
        default='dark_factory',
        help='FalkorDB graph (project) to operate on (default: dark_factory)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Rebuild every entity regardless of staleness.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Detect stale entities but do not rebuild. Review output before re-running without this flag.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    args = parser.parse_args()
    asyncio.run(
        run_rebuild_summaries(
            config_path=args.config,
            group_id=args.group_id,
            force=args.force,
            dry_run=args.dry_run,
        )
    )
