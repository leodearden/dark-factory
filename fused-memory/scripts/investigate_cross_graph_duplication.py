#!/usr/bin/env python3
"""Read-only diagnostic to reproduce/scope cross-graph node duplication and
GRAPH.LIST name-normalization collisions (task 2116).

Background
----------
Graphiti node f02a32ea-0efd-4865-94b4-97a412d8ffda (name 'orchestrator',
group_id='reify') has been observed present in three separate FalkorDB
graphs: 'reify', 'dark_factory', and 'know_live'. ``GraphitiBackend._driver_for``
/ ``_graph_for`` (fused_memory/backends/graphiti_client.py) use ``group_id``
VERBATIM as the FalkorDB graph name, with NO canonicalization applied. By
contrast, ``resolve_project_id`` (fused_memory/models/scope.py) -- which
lowercases a project root's basename and maps hyphens to underscores -- IS
applied at other request boundaries (e.g. tools.py, sqlite_task_backend.py).
This divergence is the suspected root cause of two GRAPH.LIST collision
families observed in the live FalkorDB instance:

  * dark_factory | dark-factory | -home-leo-src-dark-factory
  * know_live | know-live | knowlive

What this script does
----------------------
Strictly READ-ONLY. It enumerates every FalkorDB graph (GRAPH.LIST), detects
name-normalization collision groups and filesystem-path-shaped graph names,
probes a target node uuid for presence across every graph, and emits a
confirm/deny verdict plus a JSON manifest. It performs NO writes, NO deletes,
NO graph surgery, and NO routing fix -- any remediation is deferred to a
separate, gated follow-up task once this investigation's verdict has been
reviewed.

Usage
-----
  # Investigate the default target node (task 2116's reported orchestrator
  # node) and write the manifest to a file.
  python scripts/investigate_cross_graph_duplication.py > manifest.json

  # Investigate a different suspected cross-graph node.
  python scripts/investigate_cross_graph_duplication.py --uuid <other-uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from fused_memory.models.scope import resolve_project_id  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TARGET_NODE_UUID: str = 'f02a32ea-0efd-4865-94b4-97a412d8ffda'
DEFAULT_LIMIT: int = 100000

logger = logging.getLogger('investigate_cross_graph_duplication')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def is_path_shaped_name(name: str) -> bool:
    """True if *name* looks like a mangled filesystem path rather than a
    clean project key (e.g. '-home-leo-src-dark-factory')."""
    raise NotImplementedError


def detect_collision_groups(graph_names: list[str]) -> dict:
    """Partition *graph_names* into name-normalization collision groups and
    suspected filesystem-path leaks.

    Returns a dict shaped ``{'collisions': [...], 'suspected_path_leaks': [...]}``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Graphiti: read-only probe
# ---------------------------------------------------------------------------

async def probe_node_across_graphs(
    graphiti: Any,
    uuid: str,
    graph_names: list[str],
) -> list[dict]:
    """Read-only probe for *uuid* across every graph in *graph_names*.

    Returns one ``{'graph', 'uuid', 'name', 'group_id'}`` entry per graph in
    which the node was found. Uses ``ro_query`` only -- never ``query``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Verdict + report assembly
# ---------------------------------------------------------------------------

def classify_config_routing(collision_result: dict, presence: list[dict]) -> dict:
    """Confirm/deny the group_id-to-graph-name routing bug from the
    collision + presence evidence.

    Returns a dict shaped ``{'confirmed': bool, 'signals': [...], 'rationale': str}``.
    """
    raise NotImplementedError


def build_investigation_report(
    target_uuid: str,
    all_graphs: list[str],
    presence: list[dict],
    collision_result: dict,
    verdict: dict,
) -> dict:
    """Assemble the final investigation report dict from already-computed
    inputs. No I/O."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(args: Any, memory_service: Any) -> dict:
    """Enumerate GRAPH.LIST, detect collisions, probe the target uuid, and
    build the investigation report. Strictly read-only: never issues a
    mutating graph query, memory delete, or edge update."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the investigation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    parser = argparse.ArgumentParser(
        description=(
            'Read-only diagnostic: reproduce/scope cross-graph node '
            'duplication and GRAPH.LIST normalization collisions.'
        ),
    )
    parser.add_argument(
        '--uuid', default=TARGET_NODE_UUID,
        help=f'Target node uuid to probe across graphs (default: {TARGET_NODE_UUID}).',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    args = parser.parse_args()

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(args, memory)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
