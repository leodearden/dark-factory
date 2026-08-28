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

from fused_memory.reconciliation.task_filter import (  # noqa: F401
    COUNT_SNAPSHOT_RE,
    is_count_snapshot,
)
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

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


# ---------------------------------------------------------------------------
# Pure core: audit report + summary table
# ---------------------------------------------------------------------------

def build_audit_report(
    scan_results_by_project: dict[str, list[EntityScanResult]],
    applied_edges: set[str],
    failed_refreshes: list[dict[str, Any]],
    dry_run: bool,
    limit_per_project: int,
    generated_at: str,
    failed_invalidations: list[dict[str, Any]] | None = None,
    enumeration_by_project: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the structured JSON audit report.

    Parameters
    ----------
    scan_results_by_project:
        ``{project_id: [EntityScanResult, ...]}``
    applied_edges:
        Set of edge_uuids that were actually invalidated (empty on dry-run).
    failed_refreshes:
        List of ``{entity_uuid, project_id, error}`` dicts for refresh errors.
    dry_run:
        True when no writes were made.
    limit_per_project:
        The safety cap that was in effect.
    generated_at:
        ISO timestamp string.
    failed_invalidations:
        List of ``{edge_uuid, project_id, error, phase}`` dicts for edges that
        could not be invalidated or whose audit memory could not be written.
    enumeration_by_project:
        ``{project_id: {entities_complete, entities_incomplete_kind,
        edges_complete, edges_incomplete_kind}}`` — how COMPLETE each
        project's two whole-graph reads were (task 4386).  Every count in this
        report describes whatever those reads returned; without this, a clean
        audit over the WHOLE corpus is indistinguishable in the JSON from an
        equally clean-looking audit over a truncated one.  A project absent
        from the mapping still gets all four keys, valued ``None`` — the shape
        is uniform on every path (including the cap abort, which returns
        before the edge read happens), and ``None`` already means "no corpus
        observed, so nothing is claimed about it".

    Returns
    -------
    Dict with keys: dry_run, generated_at, limit_per_project, projects, matches,
    totals, summaries_matched, failed_invalidations.

    Each ``projects[pid]`` entry carries entities_scanned, edges_matched,
    edges_invalidated, refresh_failures and the four enumeration keys above.
    ``totals`` carries entities_scanned, edges_matched, edges_invalidated,
    refresh_failures and ``incomplete_enumerations`` — the count of PROJECTS
    (not reads) whose corpus was not proven whole.
    """
    # Collect all matches in deterministic order (by edge_uuid globally)
    all_matches: list[dict[str, Any]] = []
    per_project_summary: dict[str, dict[str, Any]] = {}

    # Collect entities whose summary contains a snapshot (report-only; the edge
    # invalidation removes the text from the rebuilt summary, but we surface these
    # so the operator can spot any residual summaries not covered by the edge scan).
    summaries_matched: list[dict[str, Any]] = []

    for pid in sorted(scan_results_by_project.keys()):
        results = scan_results_by_project[pid]
        entities_scanned = len(results)
        # Dedupe edges across entities for this project
        seen_edges: set[str] = set()
        project_edge_matches: list[dict[str, Any]] = []
        for r in results:
            for m in r.edge_matches:
                if m.edge_uuid not in seen_edges:
                    seen_edges.add(m.edge_uuid)
                    project_edge_matches.append({
                        'edge_uuid': m.edge_uuid,
                        'entity_uuids': sorted(m.entity_uuids),
                        'project_id': m.project_id,
                        'fact_excerpt': m.fact_excerpt,
                        'invalidated': m.edge_uuid in applied_edges,
                    })
        project_edge_matches.sort(key=lambda x: x['edge_uuid'])
        all_matches.extend(project_edge_matches)

        edges_invalidated = sum(1 for m in project_edge_matches if m['invalidated'])
        proj_refresh_failures = [
            f for f in failed_refreshes if f.get('project_id') == pid
        ]
        # Read completeness for THIS project's own two reads, defaulting to
        # UNKNOWN (None) rather than omitted, so the entry shape is identical
        # on every path a caller can reach.  (task 4386)
        proj_enumeration = (enumeration_by_project or {}).get(pid) or {}
        per_project_summary[pid] = {
            'entities_scanned': entities_scanned,
            'edges_matched': len(project_edge_matches),
            'edges_invalidated': edges_invalidated,
            'refresh_failures': len(proj_refresh_failures),
            'entities_complete': proj_enumeration.get('entities_complete'),
            'entities_incomplete_kind': proj_enumeration.get('entities_incomplete_kind'),
            'edges_complete': proj_enumeration.get('edges_complete'),
            'edges_incomplete_kind': proj_enumeration.get('edges_incomplete_kind'),
        }

        for r in results:
            if r.summary_matched:
                summaries_matched.append({
                    'entity_uuid': r.entity_uuid,
                    'entity_name': r.entity_name,
                    'project_id': r.project_id,
                    'summary_excerpt': r.summary_excerpt,
                })

    all_matches.sort(key=lambda x: x['edge_uuid'])

    totals: dict[str, Any] = {
        'entities_scanned': sum(v['entities_scanned'] for v in per_project_summary.values()),
        'edges_matched': sum(v['edges_matched'] for v in per_project_summary.values()),
        'edges_invalidated': sum(v['edges_invalidated'] for v in per_project_summary.values()),
        'refresh_failures': len(failed_refreshes),
        # PROJECTS, not reads: a project with BOTH reads partial counts once.
        # This answers "how many projects is this report unreliable for", which
        # is a per-project question -- counting reads would report 2 for a
        # single affected project and read as twice the damage.  `is not True`
        # and not `is False` on purpose: UNKNOWN is not a project known to be
        # whole, and admitting it here would let a run that never looked pass
        # as one that looked and found everything.  (task 4386)
        'incomplete_enumerations': sum(
            1 for v in per_project_summary.values()
            if v['entities_complete'] is not True or v['edges_complete'] is not True
        ),
    }

    return {
        'dry_run': dry_run,
        'generated_at': generated_at,
        'limit_per_project': limit_per_project,
        'projects': per_project_summary,
        'matches': all_matches,
        'totals': totals,
        'failed_refreshes': failed_refreshes,
        'summaries_matched': summaries_matched,
        'failed_invalidations': failed_invalidations or [],
    }


def _corpus_marker(project_summary: dict[str, Any]) -> str:
    """Render one project's two read verdicts as a single operator-legible cell.

    ``ok`` only when BOTH reads were PROVEN whole.  ``PARTIAL`` when either was
    observed and found incomplete.  ``?`` when either is UNKNOWN — deliberately
    NOT ``ok``, because a run that never looked must never read as a run that
    looked and found everything, which is the single failure this whole signal
    exists to prevent.  (task 4386)
    """
    verdicts = (
        project_summary.get('entities_complete'),
        project_summary.get('edges_complete'),
    )
    if any(v is False for v in verdicts):
        return 'PARTIAL'
    if all(v is True for v in verdicts):
        return 'ok'
    return '?'


def format_summary_table(report: dict[str, Any]) -> str:
    """Render a human-readable per-project summary table from an audit report.

    Produces one row per project plus a TOTALS row, and — when any project was
    scanned over a partial corpus — a trailing warning naming those projects
    (task 4386).  This table goes to STDERR while the machine-readable JSON
    goes to stdout, so the warning reaches the person reading the terminal
    without polluting the parseable output.

    The ``Corpus`` column exists because the operator's next move after reading
    this table is to re-run with ``--apply`` and invalidate the matched edges,
    and that is precisely the move that is unsafe over a partial read: the
    counts describe whatever was fetched, so on a truncated corpus a small
    ``Matched`` is not evidence that little is wrong.
    """
    projects = report.get('projects', {})
    totals = report.get('totals', {})

    header = (
        f"{'Project':<30} {'Entities':>9} {'Matched':>9} {'Invalidated':>12} "
        f"{'RefFail':>8} {'Corpus':>8}"
    )
    sep = '-' * len(header)
    rows = [header, sep]

    partial_projects: list[str] = []
    for pid in sorted(projects.keys()):
        p = projects[pid]
        marker = _corpus_marker(p)
        if marker != 'ok':
            partial_projects.append(pid)
        rows.append(
            f"{pid:<30} {p.get('entities_scanned', 0):>9} "
            f"{p.get('edges_matched', 0):>9} {p.get('edges_invalidated', 0):>12} "
            f"{p.get('refresh_failures', 0):>8} {marker:>8}"
        )

    rows.append(sep)
    rows.append(
        f"{'TOTAL':<30} {totals.get('entities_scanned', 0):>9} "
        f"{totals.get('edges_matched', 0):>9} {totals.get('edges_invalidated', 0):>12} "
        f"{totals.get('refresh_failures', 0):>8} "
        f"{totals.get('incomplete_enumerations', 0):>8}"
    )

    dry_tag = ' [DRY RUN]' if report.get('dry_run') else ''
    rows.insert(0, f"Count-snapshot cleanup report — {report.get('generated_at', '')}{dry_tag}")

    if partial_projects:
        rows.append('')
        rows.append(
            'WARNING: these projects were scanned over a PARTIAL or UNVERIFIED '
            f'corpus: {", ".join(partial_projects)}. Their counts above describe '
            'only what could be fetched, so for them an absence of matches is '
            'NOT proof of cleanliness — re-run before concluding a project is '
            'clean, and do not treat this report as a licence to --apply there.'
        )
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Pure helpers: project selection + safety cap
# ---------------------------------------------------------------------------

def select_projects(
    known_map: dict[str, str],
    project_id_filter: str | None,
) -> list[str]:
    """Return the sorted list of project_ids to process.

    Parameters
    ----------
    known_map:
        ``{project_id: project_root}`` from ``build_known_projects_map``.
    project_id_filter:
        When given, restrict to this single project_id.  Raises ValueError
        with the list of known ids if the filter is not recognised.

    Returns
    -------
    Sorted list of project_ids.
    """
    if project_id_filter is None:
        return sorted(known_map.keys())
    if project_id_filter not in known_map:
        known_ids = sorted(known_map.keys())
        raise ValueError(
            f'Unknown project_id {project_id_filter!r}. '
            f'Known project ids: {known_ids}'
        )
    return [project_id_filter]


def check_limit_cap(
    per_project_entity_counts: dict[str, int],
    limit: int,
    yes_i_am_sure: bool,
) -> tuple[list[str], bool]:
    """Check whether any project exceeds the per-project entity safety cap.

    Parameters
    ----------
    per_project_entity_counts:
        ``{project_id: entity_count}``
    limit:
        Maximum allowed entity count before aborting.
    yes_i_am_sure:
        When True, override the abort even if projects exceed the limit.

    Returns
    -------
    ``(exceeding_projects, abort)`` where ``exceeding_projects`` lists
    project_ids whose count > limit and ``abort`` is True when any project
    exceeds AND ``yes_i_am_sure`` is False.
    """
    exceeding = [pid for pid, count in per_project_entity_counts.items() if count > limit]
    abort = bool(exceeding) and not yes_i_am_sure
    return exceeding, abort


# ---------------------------------------------------------------------------
# Live shell: apply + run
# ---------------------------------------------------------------------------

async def apply_cleanup(
    memory: Any,
    scan_results: list[EntityScanResult],
    now: datetime,
) -> dict[str, Any]:
    """Invalidate matched edges, write audit memories, refresh entity summaries.

    Parameters
    ----------
    memory:
        A live (or mock) MemoryService instance.
    scan_results:
        List of EntityScanResult from scan_entities_for_snapshots.
    now:
        Datetime to use as invalid_at (and invalidated_at in audit memory).

    Returns
    -------
    ``{applied_edges: set[str], failed_refreshes: list[dict]}``
    """
    now_iso = now.isoformat()

    # Dedupe edges across all entity results
    all_edges: dict[str, EdgeMatch] = {}
    for r in scan_results:
        for m in r.edge_matches:
            if m.edge_uuid not in all_edges:
                all_edges[m.edge_uuid] = m

    applied_edges: set[str] = set()
    failed_invalidations: list[dict[str, Any]] = []

    # Invalidate each unique edge + write audit memory.
    # Each (update_edge, add_memory) pair is wrapped in its own try/except so
    # a transient error on one edge does not abort the sweep and leave other
    # edges partially invalidated or without their audit record.
    # - update_edge failure: edge skipped entirely (not added to applied_edges).
    # - add_memory failure after a successful update_edge: the edge is already
    #   invalidated (a safe re-run will skip it via get_all_valid_edges); the
    #   failure is surfaced in failed_invalidations so the operator knows the
    #   rollback-audit record is missing for that edge.
    for edge_uuid, match in sorted(all_edges.items()):
        try:
            await memory.update_edge(
                edge_uuid=edge_uuid,
                project_id=match.project_id,
                invalid_at=now,
                _source='cleanup_count_snapshots',
            )
        except Exception as exc:
            logger.warning('Failed to invalidate edge %s: %s', edge_uuid, exc)
            failed_invalidations.append({
                'edge_uuid': edge_uuid,
                'project_id': match.project_id,
                'error': str(exc),
                'phase': 'update_edge',
            })
            continue  # Skip audit memory; edge was not invalidated

        applied_edges.add(edge_uuid)

        # Use the lexicographically-first entity_uuid as canonical representative
        canonical_entity = sorted(match.entity_uuids)[0]
        payload = build_audit_memory_payload(match, canonical_entity, now_iso)
        try:
            await memory.add_memory(**payload, _source='cleanup_count_snapshots')
        except Exception as exc:
            logger.warning('Failed to write audit memory for edge %s: %s', edge_uuid, exc)
            # Edge is already invalidated; record so operator can write the
            # audit memory manually if the rollback guarantee matters.
            failed_invalidations.append({
                'edge_uuid': edge_uuid,
                'project_id': match.project_id,
                'error': str(exc),
                'phase': 'add_memory',
            })

    # Refresh entity summaries for all affected entities (once per entity).
    # NOTE: update_edge already rebuilds BOTH endpoint summaries internally on
    # each call, so by the time we reach here summaries have been rebuilt once
    # per edge invalidation.  This explicit pass is kept because:
    #   1. It consolidates the refresh to once per entity AFTER all its edges
    #      are invalidated — the authoritative final state.
    #   2. It lets the report track refresh failures explicitly (a
    #      NodeNotFoundError must not abort the whole sweep — PRD §2 issue #1).
    #   3. It is robust to any best-effort nature of update_edge's internal
    #      refresh.
    # A refresh failure here is non-fatal; log and continue.
    affected_entities: dict[str, str] = {}  # entity_uuid -> project_id
    for r in scan_results:
        if r.edge_matches:
            affected_entities[r.entity_uuid] = r.project_id
        # Also include any extra entity_uuids from edge endpoints
        for m in r.edge_matches:
            for euuid in m.entity_uuids:
                if euuid not in affected_entities:
                    affected_entities[euuid] = m.project_id

    failed_refreshes: list[dict[str, Any]] = []
    for entity_uuid, project_id in sorted(affected_entities.items()):
        try:
            await memory.refresh_entity_summary(
                entity_uuid=entity_uuid,
                project_id=project_id,
                _source='cleanup_count_snapshots',
            )
        except Exception as exc:
            logger.warning('Failed to refresh entity %s: %s', entity_uuid, exc)
            failed_refreshes.append({
                'entity_uuid': entity_uuid,
                'project_id': project_id,
                'error': str(exc),
            })

    return {
        'applied_edges': applied_edges,
        'failed_refreshes': failed_refreshes,
        'failed_invalidations': failed_invalidations,
    }


async def run(
    args: argparse.Namespace,
    *,
    memory: Any,
    known_projects_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Main async logic: scan, report, and optionally apply.

    Parameters
    ----------
    args:
        Parsed argparse.Namespace (apply, project_id, limit_per_project,
        yes_i_am_sure).
    memory:
        Live or mock MemoryService instance.
    known_projects_map:
        Optional override for the project registry (used in tests to inject
        a fixture map without reading env vars / filesystem).

    Returns
    -------
    The audit report dict.
    """
    generated_at = datetime.now(UTC).isoformat()

    if known_projects_map is None:
        from fused_memory.config.schema import FusedMemoryConfig as _FMC  # noqa: PLC0415
        from fused_memory.models.scope import build_known_projects_map  # noqa: PLC0415
        cfg = _FMC()
        known_projects_map = build_known_projects_map(cfg.taskmaster.project_root if cfg.taskmaster else '')

    try:
        project_ids = select_projects(known_projects_map, getattr(args, 'project_id', None))
    except ValueError as exc:
        abort_payload: dict[str, Any] = {
            'error': str(exc),
            'aborted': True,
            'dry_run': not args.apply,
            'generated_at': generated_at,
        }
        print(json.dumps(abort_payload, indent=2, default=str))
        filter_val = getattr(args, 'project_id', None)
        print(
            f'ABORT: unknown --project-id {filter_val!r}; '
            f'known: {sorted(known_projects_map)}',
            file=sys.stderr,
        )
        return abort_payload

    # Fail-CLOSED capability preflight, one probe per run, BEFORE the scan.
    #
    # Deliberately placed AFTER the unknown---project-id abort above (a bad
    # project id still fails with its own clearer message and never probes)
    # and BEFORE the first store read below. It is NOT at the ``if args.apply:``
    # gate further down, which is downstream of the entire two-pass
    # enumeration, and NOT inside ``apply_cleanup``, whose three mutation loops
    # each carry a best-effort ``except Exception``: since
    # ``StoreMutationUnavailable`` subclasses ``RuntimeError``, a probe there
    # would be absorbed into failed_invalidations/failed_refreshes rows while
    # the writes proceeded -- and ``update_edge`` invalidates an edge BEFORE
    # its audit memory is written, so each such row would be an
    # already-invalidated edge with no rollback record.
    if args.apply:
        try:
            assert_store_mutation_allowed(operation='cleanup_count_snapshots --apply')
        except StoreMutationUnavailable:
            logger.error(
                'cleanup_count_snapshots: --apply NOT started (fail-closed) -- '
                "this process cannot write mem0's history directory, so each "
                'invalidation would supersede its edge and then fail to write '
                'the audit memory that records the rollback, leaving edges '
                'invalidated with no way to reconstruct what they said. No '
                'entity was enumerated, no edge was scanned and nothing was '
                'mutated. Route the cleanup through the fused-memory MCP '
                'server (the unsandboxed owner of the store), or re-run from '
                'an unsandboxed operator shell. To obtain the audit report '
                'safely from anywhere, re-run without --apply.'
            )
            raise

    # Both whole-graph reads below go through the ``enumerate_*`` API rather
    # than the ``list_entity_nodes`` / ``get_all_valid_edges`` shims, which
    # return the collection alone and discard the completeness signal.  The
    # audit report is an OPERATOR-facing artefact whose every count describes
    # whatever the reads returned, so "was this the whole corpus?" belongs in
    # it -- see build_audit_report's enumeration_by_project.  (task 4386)
    #
    # ``enumerate_*`` NEVER raises: it reports incompleteness as a value.  So
    # the fail-closed structural guard the shims applied on this script's
    # behalf has to be re-applied here, through the SAME shared policy, or a
    # page-capped read is taken for the whole corpus and every entity the
    # missing pages carry is scanned as absent.  Deliberately left UNCAUGHT,
    # which preserves today's behaviour exactly: ``run()`` has no try/except
    # around these reads, so a structural incompleteness propagates out and
    # aborts BEFORE apply_cleanup -- precisely what the shim's raise does
    # today.  An EMPIRICAL incompleteness only warns, and the scan proceeds on
    # what was fetched with the report saying so.
    from fused_memory.backends.graphiti_client import (  # noqa: PLC0415
        apply_incompleteness_policy,
    )

    enumeration_by_project: dict[str, dict[str, Any]] = {}

    # First pass: fetch entity counts for the safety cap check.
    # We do this BEFORE enumerating edges / scanning so that an
    # oversized project is rejected cheaply, without enumerating its edges.
    per_project_counts: dict[str, int] = {}
    entities_by_project: dict[str, list[Any]] = {}
    for pid in project_ids:
        entities, paged = await memory.graphiti.enumerate_entity_nodes(group_id=pid)
        # Recorded BEFORE the policy is applied, and the ordering is
        # load-bearing: apply_incompleteness_policy RAISES on a structural
        # incompleteness, so recording after it would leave the aborted run
        # with no stated reason.  The two edge keys are seeded here in the
        # SAME record, so the cap-abort path below -- which returns before the
        # edge loop ever runs -- still yields the uniform four-key shape.
        enumeration_by_project[pid] = {
            'entities_complete': paged.complete,
            'entities_incomplete_kind': paged.incomplete_kind,
            'edges_complete': None,
            'edges_incomplete_kind': None,
        }
        apply_incompleteness_policy(
            paged,
            method='enumerate_entity_nodes',
            group_id=pid,
            returned_count=len(entities),
            noun='nodes',
            consequence='must not drive a staleness verdict or a summary rewrite',
        )
        entities_by_project[pid] = entities
        per_project_counts[pid] = len(entities)

    # Safety cap — abort before issuing enumerate_all_valid_edges / scanning
    exceeding, abort = check_limit_cap(
        per_project_counts,
        args.limit_per_project,
        args.yes_i_am_sure,
    )
    if abort:
        # The node-read completeness travels WITH the abort, because the cap
        # itself was decided on a count from that read: check_limit_cap sees
        # per_project_counts derived from len(entities), so a TRUNCATED node
        # read UNDER-counts and an oversized project slips UNDER the cap --
        # the cap silently stops protecting exactly the projects it exists
        # for. This return is before build_audit_report, so the report path
        # cannot carry it and the payload must. (task 4386)
        cap_payload: dict[str, Any] = {
            'aborted': True,
            'dry_run': not args.apply,
            'exceeding_projects': exceeding,
            'limit_per_project': args.limit_per_project,
            'generated_at': generated_at,
            'entities_complete': {
                pid: rec['entities_complete']
                for pid, rec in enumeration_by_project.items()
            },
            'entities_incomplete_kind': {
                pid: rec['entities_incomplete_kind']
                for pid, rec in enumeration_by_project.items()
            },
        }
        print(json.dumps(cap_payload, indent=2, default=str))
        print(
            f'ABORT: projects exceeding --limit-per-project={args.limit_per_project}: '
            f'{exceeding}. Pass --yes-i-am-sure to override.',
            file=sys.stderr,
        )
        return cap_payload

    # Second pass: scan only projects that passed the cap
    scan_results_by_project: dict[str, list[EntityScanResult]] = {}
    for pid in project_ids:
        edges_by_entity, paged = await memory.graphiti.enumerate_all_valid_edges(
            group_id=pid,
        )
        # Recorded before the policy call, for the same reason as the node
        # read above.
        enumeration_by_project[pid]['edges_complete'] = paged.complete
        enumeration_by_project[pid]['edges_incomplete_kind'] = paged.incomplete_kind
        apply_incompleteness_policy(
            paged,
            method='enumerate_all_valid_edges',
            group_id=pid,
            returned_count=len(edges_by_entity),
            noun='entities',
            consequence='must not be written back',
        )
        scan_results_by_project[pid] = scan_entities_for_snapshots(
            pid, entities_by_project[pid], edges_by_entity
        )

    # Apply or dry-run
    applied_edges: set[str] = set()
    failed_refreshes: list[dict[str, Any]] = []
    failed_invalidations: list[dict[str, Any]] = []

    if args.apply:
        all_results = [r for results in scan_results_by_project.values() for r in results]
        now = datetime.now(UTC)
        apply_result = await apply_cleanup(memory, all_results, now)
        applied_edges = apply_result['applied_edges']
        failed_refreshes = apply_result['failed_refreshes']
        failed_invalidations = apply_result.get('failed_invalidations', [])

    report = build_audit_report(
        scan_results_by_project=scan_results_by_project,
        applied_edges=applied_edges,
        failed_refreshes=failed_refreshes,
        failed_invalidations=failed_invalidations,
        dry_run=not args.apply,
        limit_per_project=args.limit_per_project,
        generated_at=generated_at,
        enumeration_by_project=enumeration_by_project,
    )

    # JSON report goes to stdout (machine-readable); summary table to stderr.
    print(json.dumps(report, indent=2, default=str))
    print(format_summary_table(report), file=sys.stderr)

    return report


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit invalidations (default: dry-run, report only)',
    )
    parser.add_argument(
        '--project-id', dest='project_id', default=None,
        help='Restrict sweep to a single project_id',
    )
    parser.add_argument(
        '--limit-per-project', dest='limit_per_project', type=int, default=1000,
        help='Abort if any project has more than this many entities (safety cap)',
    )
    parser.add_argument(
        '--yes-i-am-sure', dest='yes_i_am_sure', action='store_true',
        help='Override the per-project entity count safety cap',
    )
    args = parser.parse_args()

    async def _run_live() -> int:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        await memory.initialize()
        try:
            report = await run(args, memory=memory)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()
        return 1 if report.get('aborted') else 0

    return asyncio.run(_run_live())


if __name__ == '__main__':
    sys.exit(main())
