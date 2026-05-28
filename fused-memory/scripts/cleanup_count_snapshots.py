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
