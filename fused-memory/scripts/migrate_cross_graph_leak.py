#!/usr/bin/env python3
"""Migrate misrouted cross-graph Graphiti nodes to their resolved home graph
(CGL-ζ, task 2272, PRD ``plans/cross-graph-entity-leak-prd.md`` contract seam
S7, Phase 1 LEAF).

Background
----------
``scripts/investigate_cross_graph_duplication.py`` (task 2116) confirmed that
``GraphitiBackend._graph_for``/``_driver_for`` use a node's ``group_id``
verbatim as the FalkorDB graph name, with no canonicalization -- so a node
whose ``group_id`` names a spelling variant, alias, or otherwise-wrong graph
key ends up "foreign" to the graph it actually lives in (``n.group_id <>
$graph_key``). This script census-enumerates every such foreign node across
every populated graph, classifies each one's correct disposition, and (with
human review of the emitted manifest) re-homes it via the CGL-ε primitives.

ζ is a pure CONSUMER of ε (task 2271, ``fused_memory.maintenance.cross_graph_move``,
merged to main): ``move_entity_across_graphs`` (S5) and
``merge_foreign_duplicate`` (S6). PRD decision G4 centralizes all Cypher and
the byte-exact ``vecf32`` raw-transport read in ε to avoid duplicating that
logic (and the file-lock contention a second implementation would invite) --
this script never re-implements a cross-graph move or a raw embedding read.

Environment requirements (standing ops-script lesson -- cf.
``purge_knowlive_namespace.py`` / ``prune_recon_cycle_summaries.py``)
-----------------------------------------------------------------------------
Run this script under the fused-memory SERVICE environment, not a bare
checkout shell: ``source .env`` first, and ensure ``PROJECT_ROOT`` +
``DASHBOARD_KNOWN_PROJECT_ROOTS`` are set (or pass ``--config`` to point at a
specific config file, which sets ``CONFIG_PATH`` before ``FusedMemoryConfig``
loads). Running with an incomplete environment silently narrows which
graphs/projects are visible to the census -- it does not raise -- so a
manifest produced from a bad environment can under-report the true
population.

Contract
--------
  * Dry run is the DEFAULT (no ``--apply``): the census + classification runs
    and a JSON manifest is printed. Nothing is mutated.
  * The emitted manifest IS the recovery record -- there is no other log of
    what was classified and why. Redirect dry-run output to a file and have
    it human-reviewed BEFORE passing it to ``--apply``.
  * ``--apply`` consumes an EXISTING, human-reviewed manifest file (via
    ``--manifest``) -- it does NOT recompute a fresh plan. It refuses to run
    without one.
  * Destructive steps are flagged: every mutation ``--apply`` performs is a
    call into an ε primitive (``move_entity_across_graphs`` /
    ``merge_foreign_duplicate``), both of which are CREATE-before-DELETE
    (crash-safe) and idempotent on re-run.
  * UNRESOLVED nodes (no populated-home and no ``ALIAS_MAP`` entry) are never
    silently dropped or routed to a new/unpopulated graph -- they block
    ``--apply`` for that node only and force a non-zero exit.

Scope note
----------
This module's test suite is mock-only (per project convention) and asserts
correctness at the manifest/classification/routing level (dry-run-touches-
nothing, apply dispatches to the right ε primitive, post-verify catches a
count mismatch). LIVE byte-fidelity (zero foreign nodes remaining, byte-exact
vector transport, against a REAL FalkorDB) is explicitly η's live
throwaway-graph rehearsal, not this task -- see ε's module docstring and PRD
decision 5. Foreign Episodic nodes are classified and listed here too, but
ε's ``move_entity_across_graphs`` is :Entity-scoped (it only reattaches, does
not relocate, Episodic nodes) -- their live relocation is also η's concern.

Usage
-----
  # Dry run (default): print a full JSON manifest, touch nothing. Redirect to
  # a file and have it reviewed BEFORE running --apply.
  python scripts/migrate_cross_graph_leak.py > cgl_migration_manifest.json

  # Commit the migration from a reviewed manifest.
  python scripts/migrate_cross_graph_leak.py --apply --manifest cgl_migration_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.maintenance.cross_graph_move import (
    merge_foreign_duplicate,
    move_entity_across_graphs,
)

logger = logging.getLogger('migrate_cross_graph_leak')

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Explicit, human-reviewable alias map: orphan group_id spellings with no
# populated graph of their own -> their canonical populated home. No silent
# canonicalization beyond this table -- see resolve_target_graph (PRD
# decision 4).
ALIAS_MAP: dict[str, str] = {
    'knowlive': 'know_live',
    'know-live': 'know_live',
    'pump-web-ui': 'pump_web_ui',
    'dark-factory': 'dark_factory',
}

# Node disposition constants (see disposition_for).
MOVE: str = 'MOVE'
MERGE: str = 'MERGE'
UNRESOLVED: str = 'UNRESOLVED'

DEFAULT_PAGE_SIZE: int = 1000
