#!/usr/bin/env python3
"""One-shot deletion of the stored reconciliation task-count snapshot records.

Reconciliation used to keep one Mem0 record per project with
``metadata.kind='task_count_snapshot'``, a periodic count of its tasks by
status. Task 10081 stopped writing it for every project, so the records still
stored are stale counts that nothing refreshes. This script deletes them.

It enumerates each selected project with a metadata-filtered scroll, then
re-checks every returned record's ``metadata.kind`` before acting, so a backend
that ignored the filter cannot turn ``--apply`` into a wider delete. A failed
delete is reported for its id and the rest still run.

Run once per registered project (plans/live-task-facts-at-recall-prd.md, C6).

Usage
-----
  # Dry run (default): print the matching record ids per project, touch nothing.
  python scripts/delete_task_count_snapshots.py

  # Delete them.
  python scripts/delete_task_count_snapshots.py --apply

  # Limit to a single project.
  python scripts/delete_task_count_snapshots.py --apply --project-id reify
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from fused_memory.maintenance._utils import override_config_path
from fused_memory.maintenance.project_selection import select_projects
from fused_memory.utils.store_mutation_preflight import assert_store_mutation_allowed

logger = logging.getLogger('delete_task_count_snapshots')

TASK_COUNT_SNAPSHOT_KIND = 'task_count_snapshot'


def _registered_projects() -> dict[str, str]:
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.models.scope import build_known_projects_map

    config = FusedMemoryConfig()
    return build_known_projects_map(config.taskmaster.project_root if config.taskmaster else '')


async def _snapshot_ids(memory: Any, project_id: str) -> list[str]:
    records = await memory.get_memories_by_metadata(
        project_id, {'kind': TASK_COUNT_SNAPSHOT_KIND},
    )
    return [
        record['id'] for record in records
        if (record.get('metadata') or {}).get('kind') == TASK_COUNT_SNAPSHOT_KIND
    ]


async def _delete_each(
    memory: Any, project_id: str, memory_ids: list[str], causation_id: str,
) -> tuple[list[str], list[str]]:
    deleted: list[str] = []
    failed: list[str] = []
    for memory_id in memory_ids:
        try:
            await memory.delete_memory(
                memory_id=memory_id,
                store='mem0',
                project_id=project_id,
                causation_id=causation_id,
                _source='delete_task_count_snapshots',
            )
        except Exception as exc:
            logger.warning(
                'delete_task_count_snapshots: failed to delete %s in %s: %s',
                memory_id, project_id, exc,
            )
            failed.append(memory_id)
        else:
            deleted.append(memory_id)
    return deleted, failed


async def run(
    args: argparse.Namespace,
    *,
    memory: Any,
    known_projects_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Find, and under ``--apply`` delete, each selected project's snapshot records.

    Returns a JSON-serialisable report: ``dry_run``, ``aborted``, and per
    project the ``matched``, ``deleted`` and ``failed`` record ids. An unknown
    ``--project-id`` returns an aborted report before any read.
    """
    started_at = datetime.now(UTC).isoformat()
    if args.apply:
        assert_store_mutation_allowed(
            operation='delete_task_count_snapshots --apply',
            logger=logger,
            extra_detail=(
                'a delete would remove the Qdrant point and then fail to write its '
                'history. No project was enumerated and nothing was mutated.'
            ),
        )

    if known_projects_map is None:
        known_projects_map = _registered_projects()
    try:
        project_ids = select_projects(known_projects_map, args.project_id)
    except ValueError as exc:
        return {
            'started_at': started_at,
            'dry_run': not args.apply,
            'aborted': True,
            'error': str(exc),
        }

    causation_id = f'delete_task_count_snapshots-{started_at}'
    projects: dict[str, dict[str, list[str]]] = {}
    for project_id in project_ids:
        matched = await _snapshot_ids(memory, project_id)
        deleted, failed = (
            await _delete_each(memory, project_id, matched, causation_id)
            if args.apply else ([], [])
        )
        projects[project_id] = {'matched': matched, 'deleted': deleted, 'failed': failed}

    return {
        'started_at': started_at,
        'dry_run': not args.apply,
        'aborted': False,
        'projects': projects,
    }


def resolve_exit_code(report: dict[str, Any]) -> int:
    """1 when the run aborted or any delete failed, else 0."""
    if report.get('aborted'):
        return 1
    if any(project['failed'] for project in report['projects'].values()):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true',
        help='Delete the matched records (default: dry run, report only).',
    )
    parser.add_argument(
        '--project-id', dest='project_id', default=None,
        help='Restrict the run to one registered project_id.',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH for this run).',
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    args = build_parser().parse_args()

    async def _run_live() -> dict[str, Any]:
        from fused_memory.config.schema import FusedMemoryConfig
        from fused_memory.maintenance._utils import maintenance_service

        async with maintenance_service(config=FusedMemoryConfig()) as (_, memory):
            return await run(args, memory=memory)

    with override_config_path(args.config):
        report = asyncio.run(_run_live())
    print(json.dumps(report, indent=2))
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
