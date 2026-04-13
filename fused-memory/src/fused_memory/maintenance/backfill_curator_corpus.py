"""Maintenance utility: backfill the task_curator Qdrant corpus from the full task tree.

The TaskCurator maintains a per-project Qdrant collection ``task_curator_{project_id}``
for vector-similarity deduplication.  Tasks are recorded individually via
``record_task()`` after each create, but the collection starts empty for existing
projects — making the curator blind to all pre-existing work.

This script fetches the full task tree from Taskmaster, flattens it, and calls
``TaskCurator.backfill_corpus()`` to populate the collection in one pass.

The operation is idempotent: re-running just re-upserts the same deterministic
point IDs, leaving nothing duplicated.

Usage::

    uv run python -m fused_memory.maintenance.backfill_curator_corpus \\
        --project-root /path/to/project

    # With explicit config:
    uv run python -m fused_memory.maintenance.backfill_curator_corpus \\
        --config /path/to/config.yaml \\
        --project-root /path/to/project
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.maintenance._utils import maintenance_service
from fused_memory.middleware.task_curator import BackfillResult, TaskCurator, _flatten_task_tree
from fused_memory.models.scope import resolve_project_id

logger = logging.getLogger(__name__)


@dataclass
class BackfillReport:
    """Aggregate report from a backfill run."""

    project_root: str
    project_id: str
    tasks_found: int = 0
    upserted: int = 0
    skipped: int = 0
    errors: int = 0


class BackfillManager:
    """Orchestrates fetching the task tree and backfilling the curator corpus.

    Args:
        config: FusedMemoryConfig (for curator + embedder settings).
        taskmaster: An already-configured TaskmasterBackend.
        curator: A TaskCurator instance (uses its backfill_corpus method).
    """

    def __init__(self, config, taskmaster: TaskmasterBackend, curator: TaskCurator) -> None:
        self.config = config
        self.taskmaster = taskmaster
        self.curator = curator

    async def backfill(self, project_root: str) -> BackfillReport:
        """Fetch the full task tree and upsert all tasks into the curator corpus.

        Args:
            project_root: Absolute path to the project root (used to derive project_id
                          and to call get_tasks).

        Returns:
            BackfillReport with task counts and outcome.
        """
        project_id = resolve_project_id(project_root)
        report = BackfillReport(project_root=project_root, project_id=project_id)

        logger.info('backfill_curator_corpus: fetching task tree for %s', project_root)
        tasks_result = await self.taskmaster.get_tasks(project_root)
        flat_tasks = _flatten_task_tree(tasks_result)
        report.tasks_found = len(flat_tasks)

        logger.info(
            'backfill_curator_corpus: found %d tasks; starting backfill into task_curator_%s',
            report.tasks_found, project_id,
        )

        result: BackfillResult = await self.curator.backfill_corpus(flat_tasks, project_id)
        report.upserted = result.upserted
        report.skipped = result.skipped
        report.errors = result.errors

        logger.info(
            'backfill_curator_corpus: complete — upserted=%d skipped=%d errors=%d',
            report.upserted, report.skipped, report.errors,
        )
        return report


async def run_backfill(
    config_path: str | None = None,
    project_root: str = '.',
) -> BackfillResult:
    """Load config, connect to Taskmaster, run backfill, and close resources.

    This is the CLI-callable entrypoint.  It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates a TaskmasterBackend from config.taskmaster and connects it.
    3. Creates a TaskCurator and runs BackfillManager.backfill().
    4. Returns a BackfillResult with upserted/skipped/errors counts.

    Args:
        config_path: Optional path to the YAML config file.  When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
        project_root: Absolute path to the project root directory.

    Returns:
        BackfillResult with upserted, skipped, and errors counts.
    """
    # maintenance_service is used here for consistent CONFIG_PATH resolution and
    # lifecycle management (override_config_path context), even though this
    # script only uses `config` from the yielded tuple and does not require the
    # full MemoryService (Graphiti + Mem0).  The unused `service` variable is
    # intentional — it keeps parity with other maintenance scripts and ensures
    # the backing stores are healthy before the backfill begins.
    async with maintenance_service(config_path) as (config, _service):
        # Build a Taskmaster client.  If no [taskmaster] section exists in the
        # config we fall back to a minimal TaskmasterConfig so the script still
        # works when project_root is supplied explicitly on the CLI.
        if config.taskmaster is not None:
            tm_config = config.taskmaster.model_copy(update={'project_root': project_root})
        else:
            tm_config = TaskmasterConfig(project_root=project_root)
        taskmaster = TaskmasterBackend(config=tm_config)

        # Build a TaskCurator (lazy — connects on first use).
        curator = TaskCurator(config=config, taskmaster=taskmaster)

        manager = BackfillManager(config=config, taskmaster=taskmaster, curator=curator)
        report = await manager.backfill(project_root=project_root)

        return BackfillResult(
            upserted=report.upserted,
            skipped=report.skipped,
            errors=report.errors,
        )


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Backfill the task_curator Qdrant corpus from the full Taskmaster task tree.',
    )
    parser.add_argument(
        '--project-root',
        required=True,
        help='Absolute path to the project root.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    args = parser.parse_args()

    result = asyncio.run(run_backfill(config_path=args.config, project_root=args.project_root))
    print(
        f'Backfill complete: upserted={result.upserted} skipped={result.skipped} '
        f'errors={result.errors}'
    )
