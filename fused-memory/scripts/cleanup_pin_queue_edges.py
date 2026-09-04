#!/usr/bin/env python3
"""One-shot cleanup: invalidate stale/unattributable pin-queue & priority-override
Graphiti edges, and refresh the affected entity summaries.

Filed against the esc-3834-1 operator gate (task 3834).  Sibling of
``cleanup_count_snapshots.py`` — same two-phase shape, same guard, same audit
contract — applied to the pin-queue/priority-override edge class rather than
the count-snapshot class.

Why this class needs a one-shot sweep at all
--------------------------------------------
``_emit_override_audit`` (server/tools.py) writes every
``set_task_priority_override`` / ``reorder_pin_queue`` call to Graphiti via
``add_memory(category='decisions_and_rationale')``.  That category is in
``GRAPHITI_PRIMARY``, so each audit line is LLM-extracted into per-task fact
edges.  The extraction is non-deterministic in three separate ways:

  * **phrasing** — roughly a fifth of pin-order facts omit the literal
    ``priority override`` phrase, which is the gate
    ``stale_priority_override_edge_sweep.extract_priority_override_task_id``
    requires, so the 2781 sweep can never see them;
  * **subject attribution** — some facts name no task at all ("Pin order is
    set to 10."), and are attributable only through ``r.episodes``;
  * **arity** — the single 2026-08-06 reorder was extracted into 69 pairwise
    "Task X is reordered with task Y" edges plus 23 "Task X is pinned before
    task Y" edges, and zero per-task pin_order assertions.

The result is an edge population the periodic sweep structurally cannot drain.
This script drains it once.  Preventing recurrence is the code-side half of
the gate ruling and is NOT this script's job.

Selection (recomputed live on every run — no hard-coded uuids)
--------------------------------------------------------------
A valid (``invalid_at IS NULL``) edge is a target when it is derived from an
override-audit episode (or is a reorder-derived ordering fact) AND one of:

  ``stale-absent``     subject task has no row at all in the live
                       ``scheduler_overrides.db`` — the override was consumed
                       or cleared, so the edge describes nothing live.
  ``value-drift``      subject task IS live but the edge asserts a different
                       ``pin_order`` than the live row.
  ``reorder-noise``    "X is reordered with Y" / "X is pinned before Y" — a
                       pairwise pseudo-fact with no single subject, which no
                       sweep can ever attribute or retire.

Subjects are resolved lexically first, then — when the fact names no task —
through the edge's source-episode content, which carries the deterministic
``Set priority override for task N: {...}`` audit line.

Edges whose subject task is live AND whose asserted value matches the live row
are never selected.  Invalidation sets ``invalid_at``; it never deletes, so the
audit trail survives and a false selection is reversible via
``update_edge(clear_invalid_at=True)``.

Usage
-----
  # Dry run (default): print the JSON audit report, touch nothing.
  python scripts/cleanup_pin_queue_edges.py

  # Commit the invalidations.
  python scripts/cleanup_pin_queue_edges.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

logger = logging.getLogger('cleanup_pin_queue_edges')

# Audit-episode prefixes emitted by server/tools.py::_emit_override_audit.
_AUDIT_PREFIXES = (
    'Set priority override',
    'Reordered pin queue',
    'Cleared ',
)

# Pairwise pseudo-facts produced by extracting a reorder event. Two subjects,
# so extract_priority_override_task_id can never yield a single subject id and
# the 2781 sweep can never retire them.
#
# BOTH the ordering phrase AND explicit pin-queue context are required, and the
# caller additionally requires two distinct task refs. An earlier version keyed
# on the ordering phrase alone and false-positived on ordinary English in the
# reify graph — 'Failed-event version expectations are pinned before the code
# under test...' (from a "Task 2554 shipped via ..." episode) and 'The fix task
# must be pinned before the blocked task can be resolved.' Neither has anything
# to do with a pin queue. Under-selection is the correct bias here: a missed
# pseudo-fact is inert, a wrongly-retired true fact is not.
_REORDER_NOISE_RE = re.compile(
    r'(?:reordered with|pinned (?:before|after)).*pin queue', re.I | re.S
)

# Terminal-event assertions: an edge saying an override was CLEARED, or that a
# task is ABSENT from the queue, states something that is true and STAYS true.
# The subject's absence from the live overrides table is the very fact such an
# edge asserts, so the "task absent => stale" rule inverts on it and would
# retire a correct record. This is the over-match defect already filed as task
# 4074; reify carries a live specimen (edge 0d3eb1e7, 'All priority overrides
# for task 4880 have been cleared.', from episode 'Cleared all priority
# override(s) for task 4880'). Checked BEFORE any staleness classification.
_TERMINAL_ASSERTION_RE = re.compile(
    r'\b(?:cleared|clearing|absence|absent|removed|unpinned|no longer|'
    r'not (?:in|present|pinned))\b',
    re.I,
)

# A pin_order value asserted in the edge's own fact text.
_ASSERTED_PIN_ORDER_RE = re.compile(
    r'pin[-\s]order (?:of |is set to |set to |= ?)(\d+)', re.I
)

# The deterministic audit line the episode carries, e.g.
# "Set priority override for task 3541: {'boost_tier': 'critical', ...}".
_EPISODE_SUBJECT_RE = re.compile(r'task (\d+): (\{.*?\})')
_EPISODE_PIN_ORDER_RE = re.compile(r"'pin_order': (\d+)")

_TASK_REF_RE = re.compile(r'[Tt]ask (\d{2,6})')

# Edge-scan page size. Must stay below FalkorDB's RESULTSET_SIZE (10000 by
# default) — see the pagination note in ``scan``.
_PAGE_SIZE = 5000


def read_live_overrides(project_root: str) -> dict[str, dict]:
    """Return ``{task_id: row}`` from the live scheduler overrides table.

    Reads the FULL table, not ``get_pin_queue``'s ``pinned=1`` projection —
    a non-pinned boost_tier override must not read as "absent" (the
    over-invalidation trap ``stale_priority_override_edge_sweep`` documents).
    """
    db_path = Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'
    if not db_path.exists():
        return {}
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = con.execute(
            'SELECT task_id, boost_tier, pinned, pin_order, ttl_until '
            'FROM overrides WHERE project_root=?',
            (project_root,),
        ).fetchall()
    finally:
        con.close()
    return {
        str(tid): {
            'boost_tier': bt,
            'pinned': bool(pin),
            'pin_order': po,
            'ttl_until': ttl,
        }
        for tid, bt, pin, po, ttl in rows
    }


def scan(graph: Any, live: dict[str, dict]) -> list[dict[str, Any]]:
    """Return the target edges. Pure read — issues no writes.

    *graph* is a falkordb graph handle for the project's group_id.
    """
    episodes: dict[str, str] = {}
    for prefix in _AUDIT_PREFIXES:
        for uuid, content in graph.query(
            'MATCH (e:Episodic) WHERE e.content STARTS WITH $p RETURN e.uuid, e.content',
            {'p': prefix},
        ).result_set:
            episodes[uuid] = content

    # Candidate edges: sourced from an audit episode, or a reorder-derived
    # ordering fact (whose source episode may itself have been reaped —
    # several of the 2026-08-06 ordering edges reference an Episodic node
    # that no longer exists, so they cannot be reached episode-first).
    #
    # This scan MUST be paginated. FalkorDB's RESULTSET_SIZE (10000 by default)
    # TRUNCATES a row-returning query SILENTLY — no error, no warning — and this
    # graph holds ~13.4k valid edges. An unpaginated scan drops ~25% of the
    # corpus and then reports a confidently wrong target list (measured: 12
    # targets instead of 104). Same trap task 4340 fixed in
    # graphiti_client.get_all_valid_edges. The fetched-vs-counted assertion
    # below is what makes any recurrence loud instead of silent.
    total_valid = graph.query(
        'MATCH ()-[r:RELATES_TO]->() WHERE r.invalid_at IS NULL RETURN count(r)'
    ).result_set[0][0]
    rows: list = []
    offset = 0
    while offset < total_valid:
        page = graph.query(
            'MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE r.invalid_at IS NULL '
            'RETURN r.uuid, r.fact, r.episodes, a.uuid, b.uuid '
            f'ORDER BY r.uuid SKIP {offset} LIMIT {_PAGE_SIZE}'
        ).result_set
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    if len(rows) != total_valid:
        raise RuntimeError(
            f'incomplete edge enumeration: fetched {len(rows)} of {total_valid} '
            'valid edges — refusing to compute a target list from a partial corpus'
        )

    candidates: dict[str, dict] = {}
    for uuid, fact, eps, src, tgt in rows:
        fact = (fact or '').strip()
        eps = list(eps or [])
        from_audit = any(e in episodes for e in eps)
        if not (from_audit or _REORDER_NOISE_RE.search(fact)):
            continue
        candidates[uuid] = {
            'edge_uuid': uuid,
            'fact': fact,
            'episodes': eps,
            'entity_uuids': [u for u in (src, tgt) if u],
        }

    targets: list[dict[str, Any]] = []
    for c in candidates.values():
        fact = c['fact']

        # Never retire a terminal-event assertion — see _TERMINAL_ASSERTION_RE.
        if _TERMINAL_ASSERTION_RE.search(fact):
            continue

        ids = set(_TASK_REF_RE.findall(fact))

        # A pairwise pseudo-fact names TWO subjects — that is precisely why no
        # single-subject extractor can ever attribute or retire it, and it is
        # the whole justification for retiring it here. Requiring the second
        # ref keeps ordinary prose that merely mentions a pin queue out.
        if _REORDER_NOISE_RE.search(fact) and len(ids) >= 2:
            targets.append({**c, 'reason': 'reorder-noise', 'subject': None})
            continue

        subject: str | None = None
        asserted: int | None = None
        attribution = 'lexical'
        if len(ids) == 1:
            subject = next(iter(ids))
            m = _ASSERTED_PIN_ORDER_RE.search(fact)
            asserted = int(m.group(1)) if m else None
        else:
            joined = ' | '.join(episodes[e] for e in c['episodes'] if e in episodes)
            found = _EPISODE_SUBJECT_RE.findall(joined)
            if not found:
                continue  # unattributable even via episode — leave it alone
            attribution = 'episode'
            subjects = {tid for tid, _ in found}
            if len(subjects) > 1:
                # The edge's source episodes name DIFFERENT tasks, so which one
                # this fact is about cannot be determined. Taking the first
                # match would be a coin flip that could retire a true fact.
                # Select only when the edge is stale under EVERY candidate
                # subject — then the verdict holds whichever is correct. If any
                # candidate is still live, skip (under-selection is the safe
                # bias). Real specimen: reify edge 3b111194 'The boost tier is
                # set to high.', sourced from episodes naming both 4880 and
                # 5166.
                if any(live.get(s) is not None for s in subjects):
                    continue
                targets.append({
                    **c, 'reason': 'stale-absent',
                    'subject': '+'.join(sorted(subjects)),
                    'attribution': 'episode-multi',
                })
                continue
            subject = next(iter(subjects))
            payload = next(p for tid, p in found if tid == subject)
            pm = _EPISODE_PIN_ORDER_RE.search(payload)
            asserted = int(pm.group(1)) if pm else None

        row = live.get(subject)
        if row is None:
            targets.append({
                **c, 'reason': 'stale-absent',
                'subject': subject, 'attribution': attribution,
            })
        elif asserted is not None and row['pin_order'] is not None and asserted != row['pin_order']:
            targets.append({
                **c, 'reason': 'value-drift',
                'subject': subject, 'attribution': attribution,
                'asserted': asserted, 'live': row['pin_order'],
            })

    targets.sort(key=lambda t: (t['reason'], t['edge_uuid']))
    return targets


async def apply_cleanup(
    memory: Any, targets: list[dict[str, Any]], project_id: str, now: datetime
) -> dict[str, Any]:
    """Invalidate each target, write its rollback-audit memory, refresh summaries."""
    applied: list[str] = []
    failures: list[dict[str, Any]] = []

    for t in targets:
        try:
            await memory.update_edge(
                edge_uuid=t['edge_uuid'],
                project_id=project_id,
                invalid_at=now,
                _source='cleanup_pin_queue_edges',
            )
        except Exception as exc:
            logger.warning('invalidate failed %s: %s', t['edge_uuid'], exc)
            failures.append({'edge_uuid': t['edge_uuid'], 'phase': 'update_edge', 'error': str(exc)})
            continue
        applied.append(t['edge_uuid'])

        # Rollback record: the fact text is only recoverable from here once the
        # edge is superseded, so this is written per edge, best-effort.
        try:
            await memory.add_memory(
                content=(
                    f'cleanup_pin_queue_edges invalidated Graphiti edge '
                    f'{t["edge_uuid"]} ({t["reason"]}) at {now.isoformat()}. '
                    f'Prior fact: {t["fact"]!r}. '
                    f'Subject task: {t.get("subject")}. '
                    'Rollback: update_edge(clear_invalid_at=True).'
                ),
                category='observations_and_summaries',
                project_id=project_id,
                agent_id='cleanup-pin-queue-edges',
                metadata={
                    'kind': 'pin_queue_edge_cleanup_audit',
                    'edge_uuid': t['edge_uuid'],
                    'reason': t['reason'],
                    'subject_task': t.get('subject'),
                    'prior_fact': t['fact'],
                },
                _source='cleanup_pin_queue_edges',
            )
        except Exception as exc:
            logger.warning('audit memory failed %s: %s', t['edge_uuid'], exc)
            failures.append({'edge_uuid': t['edge_uuid'], 'phase': 'add_memory', 'error': str(exc)})

    entity_uuids = {u for t in targets if t['edge_uuid'] in applied for u in t['entity_uuids']}
    failed_refreshes: list[dict[str, Any]] = []
    for euuid in sorted(entity_uuids):
        try:
            await memory.refresh_entity_summary(
                entity_uuid=euuid, project_id=project_id, _source='cleanup_pin_queue_edges'
            )
        except Exception as exc:
            logger.warning('refresh failed %s: %s', euuid, exc)
            failed_refreshes.append({'entity_uuid': euuid, 'error': str(exc)})

    return {
        'applied': applied,
        'failures': failures,
        'failed_refreshes': failed_refreshes,
        'entities_refreshed': len(entity_uuids) - len(failed_refreshes),
    }


async def run(args: argparse.Namespace, memory: Any = None) -> dict[str, Any]:
    from falkordb import FalkorDB  # noqa: PLC0415

    generated_at = datetime.now(UTC)

    if args.apply:
        # Fail-CLOSED, before the first read: update_edge supersedes an edge
        # BEFORE its rollback-audit memory is written, so a process that cannot
        # write mem0's history would strand invalidated edges with no record of
        # what they said.
        try:
            assert_store_mutation_allowed(operation='cleanup_pin_queue_edges --apply')
        except StoreMutationUnavailable:
            logger.error(
                'cleanup_pin_queue_edges: --apply NOT started (fail-closed) — this '
                'process cannot write mem0 history. Route through the fused-memory '
                'MCP server, or re-run without --apply for the report only.'
            )
            raise

    graph = FalkorDB(host=args.falkor_host, port=args.falkor_port).select_graph(args.project_id)
    live = read_live_overrides(args.project_root)
    targets = scan(graph, live)

    by_reason: dict[str, int] = {}
    for t in targets:
        by_reason[t['reason']] = by_reason.get(t['reason'], 0) + 1

    result: dict[str, Any] = {
        'dry_run': not args.apply,
        'generated_at': generated_at.isoformat(),
        'project_id': args.project_id,
        'live_override_task_ids': sorted(live),
        'target_count': len(targets),
        'by_reason': by_reason,
        'targets': targets,
    }

    if args.apply:
        outcome = await apply_cleanup(memory, targets, args.project_id, generated_at)
        result.update(outcome)
        result['applied_count'] = len(outcome['applied'])

    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true',
                        help='Commit invalidations (default: dry-run, report only)')
    parser.add_argument('--project-id', default='dark_factory')
    parser.add_argument('--project-root', default='/home/leo/src/dark-factory')
    parser.add_argument('--falkor-host', default='localhost')
    parser.add_argument('--falkor-port', type=int, default=6379)
    args = parser.parse_args()

    async def _run_live() -> int:
        memory = None
        if args.apply:
            from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
            from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

            memory = MemoryService(FusedMemoryConfig())
            await memory.initialize()
        try:
            report = await run(args, memory=memory)
        finally:
            if memory is not None and hasattr(memory, 'close'):
                await memory.close()
        print(json.dumps(report, indent=2, default=str))
        print(
            f'\n{"[DRY RUN] " if report["dry_run"] else ""}'
            f'targets={report["target_count"]} by_reason={report["by_reason"]}'
            + (f' applied={report.get("applied_count")}' if not report['dry_run'] else ''),
            file=sys.stderr,
        )
        return 0

    return asyncio.run(_run_live())


if __name__ == '__main__':
    sys.exit(main())
