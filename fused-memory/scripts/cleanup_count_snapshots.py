#!/usr/bin/env python3
"""One-shot cleanup: invalidate Graphiti edges whose fact text contains legacy
count-snapshot pollution (e.g. "1505 done / 148 cancelled") and refresh the
affected entity summaries.

Two-phase model
---------------
**Phase 1 — Scan + report (default, --dry-run)**:
  Enumerate every known project's Graphiti entity nodes and valid edges.
  Detect edges whose ``fact`` text matches the count-snapshot pattern (reused
  from task_filter.COUNT_SNAPSHOT_RE / is_count_snapshot).  Report the matches
  in structured JSON + a human-readable summary table.  No writes.

**Phase 2 — Invalidate + audit (--apply)**:
  For each matched edge set ``invalid_at`` to the run start time via
  ``memory.update_edge``.  Write one per-edge rollback-audit memory to Mem0
  via ``memory.add_memory`` (category='observations_and_summaries',
  kind='count_snapshot_cleanup_audit').  After all edges for a given entity are
  invalidated, call ``memory.refresh_entity_summary`` to rebuild its summary;
  refresh failures are non-fatal and recorded in the report.

Usage
-----
  # Dry run (default): print JSON audit report, touch nothing.
  python scripts/cleanup_count_snapshots.py

  # Commit the invalidations.
  python scripts/cleanup_count_snapshots.py --apply

  # Limit to a single project.
  python scripts/cleanup_count_snapshots.py --project-id dark_factory

  # Override the per-project entity limit (safety cap).
  python scripts/cleanup_count_snapshots.py --apply --limit-per-project 2000

  # Bypass the safety cap (required when a project has > limit entities).
  python scripts/cleanup_count_snapshots.py --apply --yes-i-am-sure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fused_memory.reconciliation.task_filter import COUNT_SNAPSHOT_RE, is_count_snapshot  # noqa: F401

logger = logging.getLogger('cleanup_count_snapshots')


# ---------------------------------------------------------------------------
# Pure data containers
# ---------------------------------------------------------------------------

@dataclass
class EdgeMatch:
    """One count-snapshot-polluted Graphiti edge, with all endpoint entity uuids."""
    edge_uuid: str
    fact_excerpt: str
    project_id: str
    entity_uuids: list[str] = field(default_factory=list)


@dataclass
class EntityScanResult:
    """Scan result for a single entity node."""
    project_id: str
    entity_uuid: str
    entity_name: str
    edge_matches: list[EdgeMatch] = field(default_factory=list)
    summary_matched: bool = False
    summary_excerpt: str | None = None


# ---------------------------------------------------------------------------
# Pure core: scan
# ---------------------------------------------------------------------------

def scan_entities_for_snapshots(
    project_id: str,
    entities: list[dict[str, Any]],
    edges_by_entity: dict[str, list[dict[str, Any]]],
) -> list[EntityScanResult]:
    """Scan entity nodes and their edges for count-snapshot pollution.

    Parameters
    ----------
    project_id:
        The Graphiti group_id / project identifier.
    entities:
        List of entity dicts from ``memory.graphiti.list_entity_nodes``.
        Each has at minimum ``{'uuid': ..., 'name': ..., 'summary': ...}``.
    edges_by_entity:
        Mapping from entity_uuid to list of edge dicts, as returned by
        ``memory.graphiti.get_all_valid_edges``.  The same edge_uuid may
        appear under multiple entity_uuids (double-attribution); we dedupe.

    Returns
    -------
    List of ``EntityScanResult``, one per entity, sorted by ``entity_uuid``.
    Each result's ``edge_matches`` are sorted by ``edge_uuid``.
    """
    # First pass: build a global edge_uuid -> EdgeMatch dict to dedupe
    # double-attributed edges across all entities.
    edge_map: dict[str, EdgeMatch] = {}

    sorted_entities = sorted(entities, key=lambda e: e['uuid'])

    for entity in sorted_entities:
        euuid = entity['uuid']
        for edge in edges_by_entity.get(euuid, []):
            edge_uuid = edge['uuid']
            fact = edge.get('fact') or ''
            if not is_count_snapshot(fact):
                continue
            if edge_uuid in edge_map:
                # Already seen — just accumulate the endpoint
                if euuid not in edge_map[edge_uuid].entity_uuids:
                    edge_map[edge_uuid].entity_uuids.append(euuid)
            else:
                excerpt = fact[:200]
                edge_map[edge_uuid] = EdgeMatch(
                    edge_uuid=edge_uuid,
                    fact_excerpt=excerpt,
                    project_id=project_id,
                    entity_uuids=[euuid],
                )

    # Second pass: build per-entity results, referencing the deduped EdgeMatch
    # objects so entity_uuids stay consistent.
    results: list[EntityScanResult] = []
    for entity in sorted_entities:
        euuid = entity['uuid']
        summary = entity.get('summary') or ''

        # Collect EdgeMatches where this entity is an endpoint
        entity_edges = [
            m for m in edge_map.values() if euuid in m.entity_uuids
        ]
        entity_edges.sort(key=lambda m: m.edge_uuid)

        # Summary-level snapshot detection (report-only)
        summary_matched = is_count_snapshot(summary)
        summary_excerpt: str | None = None
        if summary_matched:
            # Return the first matching line as excerpt
            for line in summary.splitlines():
                if is_count_snapshot(line):
                    summary_excerpt = line[:200]
                    break
            if summary_excerpt is None:
                summary_excerpt = summary[:200]

        results.append(EntityScanResult(
            project_id=project_id,
            entity_uuid=euuid,
            entity_name=entity.get('name', ''),
            edge_matches=entity_edges,
            summary_matched=summary_matched,
            summary_excerpt=summary_excerpt,
        ))

    return results


# ---------------------------------------------------------------------------
# Pure core: audit memory payload
# ---------------------------------------------------------------------------

def build_audit_memory_payload(
    match: EdgeMatch,
    entity_uuid: str,
    now_iso: str,
) -> dict[str, Any]:
    """Build the kwargs dict for memory.add_memory for one invalidated edge.

    Parameters
    ----------
    match:
        The EdgeMatch whose edge is being invalidated.
    entity_uuid:
        The primary endpoint entity uuid (canonical representative when the
        edge had multiple endpoints).
    now_iso:
        ISO-8601 timestamp string for the ``invalidated_at`` metadata field.

    Returns
    -------
    Dict with keys: content, category, agent_id, project_id, metadata.
    """
    content = (
        f'Count-snapshot cleanup: invalidated edge {match.edge_uuid} '
        f'on entity {entity_uuid} (project={match.project_id}); '
        f'original fact: {match.fact_excerpt}'
    )
    return {
        'content': content,
        'category': 'observations_and_summaries',
        'agent_id': 'cleanup-count-snapshots',
        'project_id': match.project_id,
        'metadata': {
            'kind': 'count_snapshot_cleanup_audit',
            'edge_uuid': match.edge_uuid,
            'entity_uuid': entity_uuid,
            'project_id': match.project_id,
            'fact_text_original': match.fact_excerpt[:500],
            'invalidated_at': now_iso,
        },
    }
