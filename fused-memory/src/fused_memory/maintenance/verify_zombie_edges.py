"""Maintenance utility for verifying and deleting specific zombie Graphiti edges by UUID.

This script targets the 19 zombie edges identified in Task 111
('Delete 19 confirmed FalkorDB zombie Graphiti edges') whose deletion was
marked done but never confirmed. Run in dry-run mode first to verify which
edges still exist, then re-run without --dry-run to delete them.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.maintenance._utils import override_config_path
from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 19 zombie edge UUIDs identified in Task 111.
# These are Graphiti short-UUID (8-char hex) identifiers.
# ---------------------------------------------------------------------------

TASK_111_ZOMBIE_UUIDS: list[str] = [
    # The real UUIDs from Task 111 were never confirmed.
    # This constant is intentionally empty to prevent accidental use of placeholder values.
    #
    # To use this script:
    #   1. Extract the real zombie edge UUIDs from FalkorDB first, OR
    #   2. Pass them via the --uuids CLI flag:
    #      python -m fused_memory.maintenance.verify_zombie_edges --dry-run --uuids <uuid1> <uuid2> ...
]


@dataclass
class VerifyResult:
    """Result of a zombie-edge verification (and optional deletion) operation."""

    found: list[str] = field(default_factory=list)
    """UUIDs that exist as edges in FalkorDB."""

    missing: list[str] = field(default_factory=list)
    """UUIDs that were NOT found as edges in FalkorDB."""

    deleted: int = 0
    """Number of edges actually deleted (0 when dry_run=True or none found)."""


class ZombieEdgeVerifier:
    """Verifies and optionally deletes zombie Graphiti edges by UUID.

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
    """

    def __init__(self, backend) -> None:
        self.backend = backend

    async def verify(self, uuids: list[str]) -> VerifyResult:
        """Check which UUIDs exist as edges in FalkorDB (read-only).

        Args:
            uuids: List of edge UUIDs to check.

        Returns:
            VerifyResult with found/missing UUID lists and deleted=0.
        """
        found = await self.backend.check_edges_by_uuid(uuids)
        found_set = set(found)
        missing = [u for u in uuids if u not in found_set]
        logger.info(
            'verify: %d found, %d missing out of %d checked',
            len(found),
            len(missing),
            len(uuids),
        )
        return VerifyResult(found=found, missing=missing, deleted=0)

    async def cleanup(
        self,
        uuids: list[str],
        dry_run: bool = False,
    ) -> VerifyResult:
        """Verify which edges exist, then optionally delete the found ones.

        Args:
            uuids: List of edge UUIDs to verify and (if not dry_run) delete.
            dry_run: When True, only verify — do not delete. Defaults to False.

        Returns:
            VerifyResult with found/missing lists and deleted count.
        """
        result = await self.verify(uuids)

        if dry_run:
            logger.info(
                'Dry run — found %d edge(s), skipping deletion. '
                'Re-run without --dry-run to delete.',
                len(result.found),
            )
            return result

        if not result.found:
            logger.info('No zombie edges found in FalkorDB; nothing to delete.')
            return result

        deleted = await self.backend.bulk_remove_edges(result.found)
        result.deleted = deleted
        logger.info('Deleted %d zombie edge(s)', deleted)
        if deleted < len(result.found):
            logger.warning(
                '%d edge(s) were found by verify but could not be deleted by bulk_remove_edges '
                '— possible edge type mismatch or concurrent removal',
                len(result.found) - deleted,
            )
        return result


async def run_verify_zombie_edges(
    uuids: list[str] | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> VerifyResult:
    """Load config, initialize service, run cleanup, close resources.

    This is the CLI-callable entrypoint. It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates and initialises a MemoryService (which owns GraphitiBackend).
    3. Creates a ZombieEdgeVerifier and runs cleanup().
    4. Closes all resources in a finally block.

    Args:
        uuids: List of edge UUIDs to process. Defaults to TASK_111_ZOMBIE_UUIDS.
        config_path: Optional path to the YAML config file. When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
        dry_run: When True, verify but do not delete edges. Defaults to False.

    Returns:
        VerifyResult with found, missing, and deleted counts.
    """
    if uuids is None:
        uuids = TASK_111_ZOMBIE_UUIDS

    if not uuids:
        raise ValueError(
            'No UUIDs to process. TASK_111_ZOMBIE_UUIDS is empty because the real UUIDs '
            'were never confirmed. Provide them via --uuids <uuid1> <uuid2> ...'
        )

    service = None
    with override_config_path(config_path):
        try:
            config = FusedMemoryConfig()
            service = MemoryService(config)
            verifier = ZombieEdgeVerifier(backend=service.graphiti)

            await service.initialize()
            result = await verifier.cleanup(uuids=uuids, dry_run=dry_run)
            logger.info(
                'run_verify_zombie_edges complete: found=%d, missing=%d, deleted=%d, dry_run=%s',
                len(result.found),
                len(result.missing),
                result.deleted,
                dry_run,
            )
            return result
        finally:
            if service is not None:
                # Catch close() errors so CONFIG_PATH restoration below always runs.
                try:
                    await service.close()
                except Exception:
                    logger.warning(
                        'Error closing service during run_verify_zombie_edges cleanup',
                        exc_info=True,
                    )


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Verify and optionally delete zombie Graphiti edges by UUID (Task 111)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Verify which edges exist but do not delete. Review before re-running without this flag.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    parser.add_argument(
        '--uuids',
        nargs='*',
        default=None,
        help=(
            'Space-separated list of edge UUIDs to process. '
            'Defaults to the 19 Task 111 zombie UUIDs.'
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        run_verify_zombie_edges(
            uuids=args.uuids,
            config_path=args.config,
            dry_run=args.dry_run,
        )
    )
