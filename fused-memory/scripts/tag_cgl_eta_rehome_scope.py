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

from fused_memory.maintenance.rehome_scope_tag import CGL_ETA_REHOME_KIND
from fused_memory.models.scope import Scope
