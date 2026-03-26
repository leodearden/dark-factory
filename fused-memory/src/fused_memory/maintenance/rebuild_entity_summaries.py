"""Maintenance utility for rebuilding entity summaries via Graphiti's build_communities().

Use this after deleting stale task-count/status-distribution edges to force Graphiti
to regenerate entity summaries based on the remaining (non-stale) edges.

The 11 affected entities accumulated stale count data because Graphiti only rebuilds
summaries on new edge addition, not on edge deletion.  Running build_communities()
triggers the full LLM-based summary regeneration pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from fused_memory.backends.graphiti_client import NodeNotFoundError
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


@dataclass
class RebuildResult:
    """Result of a summary rebuild operation."""

    inspected: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    """List of (uuid, name, summary) tuples from inspect_summaries()."""

    rebuilt: bool = False
    """True if build_communities() was called (False when inspect_only=True)."""


class SummaryRebuilder:
    """Inspect and rebuild Graphiti entity summaries.

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    async def inspect_summaries(
        self,
        uuids: list[str],
    ) -> list[tuple[str, str | None, str | None]]:
        """Return (uuid, name, summary) for each UUID.

        Gracefully handles missing entities by returning (uuid, None, None).

        Args:
            uuids: List of Entity node UUIDs to inspect.

        Returns:
            List of (uuid, name, summary) tuples in the same order as input.
        """
        results: list[tuple[str, str | None, str | None]] = []
        for uuid in uuids:
            try:
                name, summary = await self.backend.get_node_text(uuid)
                results.append((uuid, name, summary))
            except NodeNotFoundError:
                logger.warning(f'Entity not found: {uuid}')
                results.append((uuid, None, None))
        return results

    async def rebuild(self, project_id: str) -> None:
        """Rebuild community summaries for the given project.

        Calls Graphiti's build_communities() which triggers LLM-based summary
        regeneration based on remaining edges.

        Args:
            project_id: The project group_id to rebuild summaries for.
        """
        logger.info(f'Running build_communities for project_id={project_id!r}')
        await self.backend.build_communities(group_ids=[project_id])
        logger.info('build_communities complete')


async def run_rebuild(
    project_id: str,
    uuids: list[str] | None = None,
    inspect_only: bool = False,
    config_path: str | None = None,
) -> dict:
    """Load config, initialize service, run rebuild + inspect, close resources.

    This is the CLI-callable entrypoint.  It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates and initialises a MemoryService (which owns GraphitiBackend).
    3. Creates a SummaryRebuilder.
    4. If not inspect_only, calls rebuild(project_id) to run build_communities().
    5. If uuids are provided, calls inspect_summaries(uuids) for diagnostics.
    6. Closes all resources in a finally block.

    Args:
        project_id: The project group_id for community rebuild.
        uuids: Optional list of entity UUIDs to inspect before/after rebuild.
        inspect_only: When True, skip rebuild and only run inspect.
        config_path: Optional path to the YAML config file.

    Returns:
        dict with keys:
          - 'inspected': list of (uuid, name, summary) tuples
          - 'rebuilt': bool — True if build_communities was called
    """
    import os

    old_config_path = os.environ.get('CONFIG_PATH')
    service = None
    try:
        if config_path is not None:
            os.environ['CONFIG_PATH'] = config_path

        config = FusedMemoryConfig()
        service = MemoryService(config)
        await service.initialize()

        rebuilder = SummaryRebuilder(backend=service.graphiti)

        rebuilt = False
        if not inspect_only:
            await rebuilder.rebuild(project_id=project_id)
            rebuilt = True

        inspected: list[tuple[str, str | None, str | None]] = []
        if uuids:
            inspected = await rebuilder.inspect_summaries(uuids)
            for uuid, name, summary in inspected:
                if name is None:
                    logger.info(f'  {uuid}: NOT FOUND')
                else:
                    # Truncate long summaries for log readability
                    preview = (summary or '')[:120].replace('\n', ' ')
                    logger.info(f'  {uuid}: {name!r} — {preview!r}')

        return {'inspected': inspected, 'rebuilt': rebuilt}

    finally:
        if service is not None:
            try:
                await service.close()
            except Exception:
                logger.warning('Error closing service during run_rebuild', exc_info=True)
        if config_path is not None:
            if old_config_path is None:
                os.environ.pop('CONFIG_PATH', None)
            else:
                os.environ['CONFIG_PATH'] = old_config_path


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Rebuild Graphiti entity summaries via build_communities()'
    )
    parser.add_argument(
        '--project-id',
        required=True,
        help='Project group_id for community rebuild (e.g. dark_factory)',
    )
    parser.add_argument(
        '--uuids',
        default=None,
        help='Comma-separated list of entity UUIDs to inspect (diagnostic only)',
    )
    parser.add_argument(
        '--inspect-only',
        action='store_true',
        default=False,
        help='Only inspect entity summaries — skip build_communities()',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var)',
    )
    args = parser.parse_args()

    uuid_list = [u.strip() for u in args.uuids.split(',')] if args.uuids else None

    result = asyncio.run(
        run_rebuild(
            project_id=args.project_id,
            uuids=uuid_list,
            inspect_only=args.inspect_only,
            config_path=args.config,
        )
    )
    rebuilt_label = 'yes' if result['rebuilt'] else 'no (inspect-only)'
    print(f'Rebuilt: {rebuilt_label}')
    print(f'Inspected {len(result["inspected"])} entities:')
    for uuid, name, summary in result['inspected']:
        if name is None:
            print(f'  {uuid}: NOT FOUND')
        else:
            preview = (summary or '')[:80].replace('\n', ' ')
            print(f'  {uuid}: {name!r} — {preview!r}')
