#!/usr/bin/env python3
"""One-shot backfill: invalidate 'shipped via X' edges that reference files not
shipped by the cited task's merge commit (or where no provenance was recorded).

Motivation: Stage-2 reconciliation historically fabricated temporal-fact edges
of the form "Task N reached status done with its implementation shipped via X"
from ``metadata.modules``, without verifying that X was actually in the merge
diff. The reify task 1746 incident on 2026-04-19 produced four such edges
against files that never existed or were shipped by a sibling task.

The new `done_provenance` gate prevents future fabrications. This script
retires the existing ones.

For each candidate edge:
  1. Parse the referenced task_id and file path from the edge fact.
  2. Load the task via Taskmaster.
  3. If ``metadata.done_provenance.commit`` is set, diff that commit; the edge
     is valid iff the file appears in the diff.
  4. If provenance is absent/note-only, the edge cannot be verified → flagged
     for invalidation (configurable via ``--keep-unverified`` to be conservative).
  5. Edges flagged for invalidation have ``invalid_at`` set to the run start
     time via ``update_edge`` — preserving the audit trail rather than deleting.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/invalidate_fabricated_shipping_edges.py --project reify

  # Commit the invalidations.
  python scripts/invalidate_fabricated_shipping_edges.py --project reify --apply

  # Be conservative: only invalidate edges that have provenance AND fail the
  # commit-diff check. Edges with no provenance at all are left alone.
  python scripts/invalidate_fabricated_shipping_edges.py \
      --project reify --keep-unverified
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger('invalidate_shipping_edges')

# Pattern for the historically-fabricated edge text. The LLM emitted variants
# like "Task 1746 reached status done with its implementation shipped via
# ViewSelector.tsx." — we match any "shipped via <path>" fragment to be lenient.
_SHIPPED_VIA_RE = re.compile(r'shipped via\s+(?P<path>[A-Za-z0-9_./-]+)')
_TASK_ID_RE = re.compile(r'Task\s+(?P<tid>\d+(?:\.\d+)*)\b')


@dataclass
class EdgeCandidate:
    edge_uuid: str
    fact: str
    project_id: str
    task_id: str | None
    file_path: str | None
    reason: str = ''
    files_in_commit: list[str] = field(default_factory=list)
    provenance_commit: str | None = None
    provenance_note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            'edge_uuid': self.edge_uuid,
            'project_id': self.project_id,
            'task_id': self.task_id,
            'file_path': self.file_path,
            'fact': self.fact,
            'reason': self.reason,
            'provenance_commit': self.provenance_commit,
            'provenance_note': self.provenance_note,
        }


async def _find_shipping_edges(graphiti, project_id: str) -> list[EdgeCandidate]:
    """Cypher query for RELATES_TO edges with 'shipped via' in fact and no invalid_at."""
    graph = graphiti._graph_for(project_id)
    cypher = (
        'MATCH ()-[e:RELATES_TO]->() '
        'WHERE e.fact CONTAINS "shipped via" AND e.invalid_at IS NULL '
        'RETURN e.uuid, e.fact'
    )
    result = await graph.ro_query(cypher)
    edges: list[EdgeCandidate] = []
    for row in result.result_set or []:
        uuid, fact = row[0], row[1]
        path_match = _SHIPPED_VIA_RE.search(fact or '')
        tid_match = _TASK_ID_RE.search(fact or '')
        edges.append(EdgeCandidate(
            edge_uuid=uuid,
            fact=fact,
            project_id=project_id,
            task_id=tid_match.group('tid') if tid_match else None,
            file_path=path_match.group('path') if path_match else None,
        ))
    return edges


async def _git_show_files(project_root: str, commit: str) -> list[str]:
    """Return the list of files in a commit diff, or [] on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'show', '--name-only', '--format=', commit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return []
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [
        ln.strip() for ln in stdout.decode('utf-8', errors='replace').splitlines()
        if ln.strip()
    ]


async def _classify(
    candidate: EdgeCandidate, taskmaster, project_root: str,
    *, keep_unverified: bool,
) -> bool:
    """Return True if the edge should be invalidated.

    Populates ``candidate.reason`` + ``provenance_*`` so the report explains
    the decision per-edge.
    """
    if candidate.task_id is None:
        candidate.reason = 'could_not_parse_task_id_from_fact'
        return not keep_unverified
    if candidate.file_path is None:
        candidate.reason = 'could_not_parse_file_path_from_fact'
        return not keep_unverified

    try:
        task_raw = await taskmaster.get_task(candidate.task_id, project_root)
    except Exception as e:
        candidate.reason = f'task_lookup_failed: {type(e).__name__}: {e}'
        return not keep_unverified

    inner = task_raw
    if isinstance(task_raw, dict) and 'data' in task_raw and isinstance(task_raw['data'], dict):
        inner = task_raw['data']
    metadata = inner.get('metadata') if isinstance(inner, dict) else None
    prov = metadata.get('done_provenance') if isinstance(metadata, dict) else None

    if not isinstance(prov, dict):
        candidate.reason = 'task_has_no_done_provenance'
        return not keep_unverified

    commit = prov.get('commit') if isinstance(prov.get('commit'), str) else None
    note = prov.get('note') if isinstance(prov.get('note'), str) else None
    candidate.provenance_commit = commit
    candidate.provenance_note = note

    if not commit:
        candidate.reason = 'note_only_provenance_cannot_verify_file'
        return not keep_unverified

    files = await _git_show_files(project_root, commit)
    candidate.files_in_commit = files
    if not files:
        candidate.reason = f'git_show_returned_no_files_for_{commit[:12]}'
        return not keep_unverified

    if candidate.file_path in files:
        candidate.reason = 'file_in_commit_diff'
        return False

    candidate.reason = f'file_not_in_commit_diff_for_{commit[:12]}'
    return True


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.backends.taskmaster_client import TaskmasterBackend  # noqa: PLC0415
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    memory = MemoryService(config)
    await memory.initialize()
    try:
        candidates = await _find_shipping_edges(memory.graphiti, args.project)
        logger.info('Found %d candidate shipping-edge(s) in %s',
                    len(candidates), args.project)

        if config.taskmaster is None:
            raise RuntimeError('Taskmaster backend not configured — cannot verify provenance')
        tm = TaskmasterBackend(config.taskmaster)
        await tm.initialize()

        to_invalidate: list[EdgeCandidate] = []
        for c in candidates:
            should = await _classify(
                c, tm, args.project_root, keep_unverified=args.keep_unverified,
            )
            if should:
                to_invalidate.append(c)

        report = {
            'project': args.project,
            'project_root': args.project_root,
            'candidates_total': len(candidates),
            'to_invalidate': len(to_invalidate),
            'dry_run': not args.apply,
            'keep_unverified': args.keep_unverified,
            'invalidated_at': datetime.now(UTC).isoformat(),
            'edges': [c.to_json() for c in candidates],
        }
        print(json.dumps(report, indent=2, default=str))

        if not args.apply:
            logger.info('Dry run — no edges were modified. Use --apply to commit.')
            return 0

        invalidation_time = datetime.now(UTC)
        for c in to_invalidate:
            try:
                await memory.update_edge(
                    edge_uuid=c.edge_uuid,
                    project_id=c.project_id,
                    invalid_at=invalidation_time,
                    _source='backfill_shipping_edges',
                )
                logger.info('Invalidated %s (%s)', c.edge_uuid, c.reason)
            except Exception as e:
                logger.error('Failed to invalidate %s: %s', c.edge_uuid, e)
        return 0
    finally:
        if hasattr(memory, 'close'):
            await memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, help='project_id to scan')
    parser.add_argument(
        '--project-root', required=True,
        help='Filesystem path to the project (for git diff and taskmaster lookups)',
    )
    parser.add_argument('--config', help='Path to fused-memory config file')
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit invalidations (default: dry-run, report only)',
    )
    parser.add_argument(
        '--keep-unverified', action='store_true',
        help=('Conservative mode: only invalidate edges that have provenance '
              'AND fail the commit-diff check. Edges with no provenance / '
              'note-only provenance are left alone.'),
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
