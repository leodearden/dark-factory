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

from fused_memory.models.scope import resolve_project_id

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TARGET_NODE_UUID: str = 'f02a32ea-0efd-4865-94b4-97a412d8ffda'

# Filesystem-ish tokens that, combined with >=4 '-'-segments, mark a name as
# a mangled absolute path (e.g. '-home-leo-src-dark-factory' ->
# ['', 'home', 'leo', 'src', 'dark', 'factory']) rather than a clean project
# key that merely happens to have several hyphen-separated words.
_PATH_TOKENS: frozenset[str] = frozenset({'home', 'src', 'usr', 'opt', 'users', 'var', 'tmp'})

logger = logging.getLogger('investigate_cross_graph_duplication')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def is_path_shaped_name(name: str) -> bool:
    """True if *name* looks like a mangled filesystem path rather than a
    clean project key (e.g. '-home-leo-src-dark-factory')."""
    if '/' in name:
        return True
    if name.startswith('-') or name.startswith('/'):
        return True
    segments = name.split('-')
    return len(segments) >= 4 and any(seg in _PATH_TOKENS for seg in segments)


def detect_collision_groups(graph_names: list[str]) -> dict:
    """Partition *graph_names* into name-normalization collision groups and
    suspected filesystem-path leaks.

    Returns a dict shaped ``{'collisions': [...], 'suspected_path_leaks': [...]}``.
    """
    path_leaks: list[str] = []
    buckets: dict[str, set[str]] = {}
    for name in graph_names:
        if is_path_shaped_name(name):
            path_leaks.append(name)
            continue
        canonical = resolve_project_id(name)
        buckets.setdefault(canonical, set()).add(name)

    collisions = [
        {'canonical': canonical, 'variants': sorted(variants), 'count': len(variants)}
        for canonical, variants in buckets.items()
        if len(variants) > 1
    ]
    collisions.sort(key=lambda c: c['canonical'])

    return {
        'collisions': collisions,
        'suspected_path_leaks': sorted(path_leaks),
    }


# ---------------------------------------------------------------------------
# Graphiti: read-only probe
# ---------------------------------------------------------------------------

async def probe_node_across_graphs(
    graphiti: Any,
    uuid: str,
    graph_names: list[str],
    errors: list[dict] | None = None,
) -> list[dict]:
    """Read-only probe for *uuid* across every graph in *graph_names*.

    Returns one ``{'graph', 'uuid', 'name', 'group_id'}`` entry per graph in
    which the node was found. Uses ``ro_query`` only -- never ``query``.

    A graph whose ``ro_query`` raises (e.g. a malformed/inaccessible
    path-leak graph -- exactly the kind of graph this tool exists to
    surface) is logged and skipped rather than aborting the whole probe, so
    one bad graph cannot sink the investigation. If *errors* is supplied,
    a ``{'graph', 'error'}`` entry is appended to it for each failing graph.
    """
    entries: list[dict] = []
    for name in graph_names:
        graph = graphiti._graph_for(name)
        try:
            result = await graph.ro_query(
                'MATCH (n {uuid: $uuid}) RETURN n.uuid, n.name, n.group_id',
                {'uuid': uuid},
            )
        except Exception as e:
            logger.warning(
                "investigate_cross_graph_duplication: failed to probe graph "
                "'%s' for uuid %s: %s", name, uuid, e,
            )
            if errors is not None:
                errors.append({'graph': name, 'error': str(e)})
            continue
        rows = result.result_set or []
        for row in rows:
            entries.append(
                {'graph': name, 'uuid': row[0], 'name': row[1], 'group_id': row[2]}
            )
    return entries


# ---------------------------------------------------------------------------
# Verdict + report assembly
# ---------------------------------------------------------------------------

def classify_config_routing(collision_result: dict, presence: list[dict]) -> dict:
    """Confirm/deny the group_id-to-graph-name routing bug from the
    collision + presence evidence.

    Returns a dict shaped ``{'confirmed': bool, 'signals': [...], 'rationale': str}``.
    """
    signals: list[str] = []

    if collision_result['collisions']:
        signals.append('name_normalization_collision')
    if collision_result['suspected_path_leaks']:
        signals.append('suspected_path_leak')
    if len(presence) >= 2 or any(
        resolve_project_id(entry['graph']) != entry['group_id'] for entry in presence
    ):
        signals.append('cross_graph_node_leak')

    confirmed = bool(signals)
    if not confirmed:
        rationale = (
            'No routing anomaly detected: no name-normalization collisions, '
            'no suspected path leaks, and the probed node (if found) is '
            "confined to a single graph consistent with its group_id."
        )
    else:
        signal_summaries = {
            'name_normalization_collision': (
                f"{len(collision_result['collisions'])} graph name-normalization "
                'collision group(s) found (distinct raw graph names collapse '
                'to the same canonical project key)'
            ),
            'suspected_path_leak': (
                f"{len(collision_result['suspected_path_leaks'])} filesystem-"
                'path-shaped graph name(s) found (a project_root leaked into '
                'the FalkorDB graph name instead of a project_id)'
            ),
            'cross_graph_node_leak': (
                f'target node present in {len(presence)} graph(s), including '
                "one whose canonical key does not match the node's recorded "
                'group_id'
            ),
        }
        rationale = '; '.join(signal_summaries[signal] for signal in signals) + '.'

    return {'confirmed': confirmed, 'signals': signals, 'rationale': rationale}


def build_investigation_report(
    target_uuid: str,
    all_graphs: list[str],
    presence: list[dict],
    collision_result: dict,
    verdict: dict,
    errors: list[dict] | None = None,
) -> dict:
    """Assemble the final investigation report dict from already-computed
    inputs. No I/O.

    *errors* (if any per-graph probe failures were recorded by
    ``probe_node_across_graphs``) is surfaced verbatim under the
    ``'errors'`` key so a partial probe failure is visible in the manifest
    rather than silently discarded.
    """
    scope = (
        'systemic'
        if collision_result['collisions'] or collision_result['suspected_path_leaks']
        else 'single_node'
    )
    return {
        'target_uuid': target_uuid,
        'all_graphs': all_graphs,
        'graph_count': len(all_graphs),
        'present_in': presence,
        'collision_groups': collision_result['collisions'],
        'suspected_path_leaks': collision_result['suspected_path_leaks'],
        'verdict': verdict,
        'scope': scope,
        'errors': errors or [],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(args: Any, memory_service: Any) -> dict:
    """Enumerate GRAPH.LIST, detect collisions, probe the target uuid, and
    build the investigation report. Strictly read-only: never issues a
    mutating graph query, memory delete, or edge update."""
    graphs = await memory_service.graphiti.list_graphs()
    collision_result = detect_collision_groups(graphs)
    uuid = getattr(args, 'uuid', None) or TARGET_NODE_UUID
    errors: list[dict] = []
    presence = await probe_node_across_graphs(memory_service.graphiti, uuid, graphs, errors=errors)
    verdict = classify_config_routing(collision_result, presence)
    return build_investigation_report(uuid, graphs, presence, collision_result, verdict, errors=errors)


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
