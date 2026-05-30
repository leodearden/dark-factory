#!/usr/bin/env python3
"""One-shot repair: clear the ``invalid_at`` on six falsely-superseded dependency edges.

Motivation
----------
Graphiti-core's LLM edge-resolution pipeline (``client.add_episode``) can
falsely supersede existing "X depends on Y" edges when a new dependency fact
is added for a hub entity.  Dependencies are additive — supersession is wrong
for them.

The six edges listed in ``TARGET_EDGE_UUIDS`` were falsely invalidated in the
``know_live`` project.  This script clears their ``invalid_at`` fields via
``update_edge(clear_invalid_at=True)``.

Backend ``update_edge`` automatically rebuilds both endpoint entity summaries
after each save, so the source/target nodes for tasks 562, 563, 561, 560,
557, and 559 are refreshed implicitly — no separate
``refresh_entity_summary`` step is needed.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/clear_false_dependency_invalidations.py

  # Commit the clearings.
  python scripts/clear_false_dependency_invalidations.py --apply

  # Target a non-default project.
  python scripts/clear_false_dependency_invalidations.py \\
      --project know_live --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

logger = logging.getLogger('clear_false_dep_invalidations')

# Six dependency-edge UUIDs that were falsely invalidated by the upstream
# LLM edge-resolution pipeline.  Clearing these restores the dependency edges
# to active status and triggers a summary rebuild for their endpoint nodes.
TARGET_EDGE_UUIDS: list[str] = [
    '001060dd-71d7-43f0-b1cd-6e29e60fb351',
    'f30b8f29-9ff0-4f78-938b-dd600d13e5d1',
    '4bd30efc-7437-45b9-a0e6-08243c52c3c2',
    '5f44fe41-65a9-44a1-b80f-bf3124043151',
    'db0a4a8c-0995-4330-9086-478cebc376fc',
    'dc96732e-f784-476c-a417-d0fe3fe2bf62',
]


async def repair(
    memory: Any,
    project_id: str = 'know_live',
    *,
    apply: bool,
) -> dict[str, Any]:
    """Clear false dependency-edge invalidations.

    For each UUID in ``TARGET_EDGE_UUIDS``, calls
    ``memory.update_edge(edge_uuid=..., project_id=..., clear_invalid_at=True)``
    when *apply* is True; otherwise records the intended action without touching
    the store.

    Args:
        memory: MemoryService instance (or compatible AsyncMock for testing).
        project_id: The project scope to target (default: 'know_live').
        apply: When False (dry run), no mutations are performed.

    Returns:
        A JSON-serialisable report dict with keys:
        ``project``, ``dry_run``, ``edges_total``, ``cleared``, ``edges``.
    """
    edges_report: list[dict[str, Any]] = []
    cleared = 0

    for edge_uuid in TARGET_EDGE_UUIDS:
        entry: dict[str, Any] = {
            'edge_uuid': edge_uuid,
            'project_id': project_id,
            'action': 'clear_invalid_at',
        }
        if apply:
            try:
                result = await memory.update_edge(
                    edge_uuid=edge_uuid,
                    project_id=project_id,
                    clear_invalid_at=True,
                    _source='clear_false_dep_invalidations',
                )
                entry['status'] = 'cleared'
                entry['verified'] = result.get('verified', None)
                cleared += 1
                logger.info('Cleared %s — verified=%s', edge_uuid, entry['verified'])
            except Exception as exc:
                entry['status'] = 'error'
                entry['error'] = f'{type(exc).__name__}: {exc}'
                logger.error('Failed to clear %s: %s', edge_uuid, exc)
        else:
            entry['status'] = 'dry_run'

        edges_report.append(entry)

    return {
        'project': project_id,
        'dry_run': not apply,
        'edges_total': len(TARGET_EDGE_UUIDS),
        'cleared': cleared,
        'edges': edges_report,
    }


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    memory = MemoryService(config)
    await memory.initialize()
    try:
        report = await repair(memory, project_id=args.project, apply=args.apply)
        print(json.dumps(report, indent=2, default=str))

        if not args.apply:
            logger.info('Dry run — no edges were modified. Use --apply to commit.')
        else:
            logger.info(
                'Cleared %d/%d edge(s).', report['cleared'], report['edges_total']
            )
            # Return non-zero when apply mode had partial failures so callers
            # and CI pipelines can detect incomplete repairs.
            if report['cleared'] < report['edges_total']:
                failed = report['edges_total'] - report['cleared']
                logger.error(
                    '%d edge(s) failed to clear — check logs for details.', failed
                )
                return 1
        return 0
    finally:
        if hasattr(memory, 'close'):
            await memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project', default='know_live',
        help='project_id to target (default: know_live)',
    )
    parser.add_argument('--config', help='Path to fused-memory config file')
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit clearings (default: dry-run, report only)',
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
