#!/usr/bin/env python3
"""One-shot (idempotently re-runnable) maintenance script: prefix every
CGL-eta ``cross_target_rehome`` Mem0 entry's content with an explicit
source-project scope tag (task 2452).

Background
----------
The CGL-eta leak migration (task 2273) wrote Mem0 entries (metadata
``kind='cgl_eta_cross_target_rehome'``) whose true project scope
(``src_project``/``dst_project``/``src_entity``/``dst_entity``) lives only in
metadata, never inline in the human-readable content. Because a rehomed fact
physically lives in its ``dst_project``'s Mem0 collection but references
``src_project``'s task numbers, a plain/semantic search over the dst
collection cannot distinguish the stale foreign fact from the dst project's
own same-numbered task without inspecting metadata -- risking a future
Stage-2/Stage-3 reconciliation cycle misattributing it.

No deletion
-----------
Unlike ``scripts/prune_recon_cycle_summaries.py`` and
``scripts/sweep_orphan_flag_markers.py`` (which delete rather than retag
because their Mem0 path lacked an in-place payload-update primitive), this
script edits IN PLACE via a metadata-forwarding ``Mem0Backend.update`` --
preserving ``created_at`` and every custom metadata key (``src_project`` /
``dst_project`` / ``kind`` / ``source_migration`` / ...) by passing the full
existing payload back on update. See ``fused_memory.maintenance.rehome_scope_tag``
for the pure tagging helper (:func:`apply_scope_tag`) this script routes
content through -- the same helper any future rehome path should reuse so
new entries are gated from ever landing untagged.

Two-phase model
----------------
**Phase 1 -- Scan + report (default, dry-run)**: enumerate every known
project's ``cgl_eta_cross_target_rehome`` entries, classify each as
tag/skip, and print a structured JSON report + human-readable summary
table. No writes.

**Phase 2 -- Apply (--apply)**: update the classified tag-set in place via
``memory.mem0.update`` (best-effort; a failed update is logged and excluded,
other updates still proceed). Idempotent -- re-running after a successful
apply finds every entry already tagged and updates nothing.

Direct ``memory.mem0.update`` -- no ``MemoryService`` wrapper, no MCP tool,
no reconciliation event -- a cosmetic provenance tag must not trigger a
recon cycle over these very facts.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/tag_cgl_eta_rehome_scope.py

  # Commit the tags.
  python scripts/tag_cgl_eta_rehome_scope.py --apply

  # Limit to a single project.
  python scripts/tag_cgl_eta_rehome_scope.py --project-id dark_factory

  # Widen the scan window (a project has more than the default scan-limit
  # cgl_eta_cross_target_rehome points) -- this raises the scan only.
  python scripts/tag_cgl_eta_rehome_scope.py --apply --scan-limit 20000

  # Point at a specific fused-memory config file.
  python scripts/tag_cgl_eta_rehome_scope.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from fused_memory.maintenance.rehome_scope_tag import (
    CGL_ETA_REHOME_KIND,
    apply_scope_tag,
    scope_tag_for,
)
from fused_memory.models.scope import Scope

logger = logging.getLogger('tag_cgl_eta_rehome_scope')


# ---------------------------------------------------------------------------
# Pure core: classify_rehome_record
# ---------------------------------------------------------------------------

def classify_rehome_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one ``cgl_eta_cross_target_rehome`` record for tagging.

    Reads content from ``record['metadata']['data']`` (Mem0's payload
    content key) and computes ``apply_scope_tag(content, record['metadata'])``.

    Returns a decision dict ``{'id', 'action', 'new_content'}`` where
    ``action`` is one of:

    - ``'skip:no_src_project'`` -- :func:`~fused_memory.maintenance.
      rehome_scope_tag.scope_tag_for` returned ``None`` (no origin project
      to disambiguate against);
    - ``'skip:already_tagged'`` -- a tag was computable but *content*
      already carries it (``apply_scope_tag`` returned it unchanged);
    - ``'tag'`` -- content needs (and gets) the scope tag prepended.

    ``new_content`` always holds ``apply_scope_tag``'s return value, so a
    ``'tag'`` decision can be applied directly and re-applying a ``'skip'``
    decision (if ever done by mistake) is a no-op.

    Pure function -- no I/O.
    """
    metadata = record['metadata']
    content = metadata.get('data') or ''
    new_content = apply_scope_tag(content, metadata)

    if scope_tag_for(metadata) is None:
        action = 'skip:no_src_project'
    elif new_content == content:
        action = 'skip:already_tagged'
    else:
        action = 'tag'

    return {'id': record['id'], 'action': action, 'new_content': new_content}


# ---------------------------------------------------------------------------
# Pure core: audit report
# ---------------------------------------------------------------------------

def build_tag_report(
    decisions_by_project: dict[str, list[dict[str, Any]]],
    applied_ids: set[str],
    dry_run: bool,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the structured JSON-serialisable tag report.

    Mirrors ``prune_recon_cycle_summaries.build_prune_report``'s shape and
    deterministic sorting, adapted to per-record (not per-pool)
    classification.

    Args:
        decisions_by_project: ``{project_id: [classify_rehome_record(...),
            ...]}`` -- one list per project that was scanned.
        applied_ids: Set of memory ids actually updated (empty on dry-run).
        dry_run: True when no writes were made.
        generated_at: ISO timestamp string.

    Returns:
        Dict with keys: dry_run, generated_at, projects, totals, changes.
        ``projects`` nests ``{project_id: {scanned, taggable, tagged,
        skipped}}``. ``changes`` is a deterministically-ordered list of
        ``{project_id, id, action, applied}`` dicts -- one entry per 'tag'
        decision (skips never appear, mirroring build_prune_report's
        deletions-only 'deletions' list).
    """
    projects: dict[str, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []

    for project_id in sorted(decisions_by_project):
        decisions = decisions_by_project[project_id]
        scanned = len(decisions)
        taggable_ids = [d['id'] for d in decisions if d['action'] == 'tag']
        taggable = len(taggable_ids)
        tagged = sum(1 for mid in taggable_ids if mid in applied_ids)
        skipped = scanned - taggable

        projects[project_id] = {
            'scanned': scanned,
            'taggable': taggable,
            'tagged': tagged,
            'skipped': skipped,
        }

        for mid in sorted(taggable_ids):
            changes.append({
                'project_id': project_id,
                'id': mid,
                'action': 'tag',
                'applied': mid in applied_ids,
            })

    totals: dict[str, Any] = {
        'scanned': sum(p['scanned'] for p in projects.values()),
        'taggable': sum(p['taggable'] for p in projects.values()),
        'tagged': sum(p['tagged'] for p in projects.values()),
        'skipped': sum(p['skipped'] for p in projects.values()),
    }

    return {
        'dry_run': dry_run,
        'generated_at': generated_at,
        'projects': projects,
        'totals': totals,
        'changes': changes,
    }
