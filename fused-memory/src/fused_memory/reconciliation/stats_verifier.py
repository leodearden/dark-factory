"""Cross-check self-reported stage stats against observed write-journal ops.

Stage agents self-report counts like ``memories_added: 3``, but Mem0 silently
deduplicates and returns an empty ``memory_ids`` list on filtered writes. The
agent then over-reports successful additions, the judge catches the mismatch,
and the run gets flagged moderate. This verifier reads the authoritative write
journal, buckets ops by stage via their timestamp windows, and overwrites the
self-reported stats with observed values. The originals are preserved under
``stats['_reported']`` so divergence remains visible to the judge.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fused_memory.models.reconciliation import StageReport

if TYPE_CHECKING:
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)


# Operation → stat-key mapping. Only keys listed here are overwritten on the
# stage's stats dict; any self-reported stat the verifier doesn't understand is
# left untouched.
_OP_TO_STAT: dict[str, str] = {
    'add_memory': 'memories_added',
    'delete_memory': 'memories_deleted',
    'add_episode': 'episodes_added',
    'delete_episode': 'episodes_deleted',
    'update_edge': 'edges_updated',
    'refresh_entity_summary': 'entity_summaries_refreshed',
    'merge_entities': 'entities_merged',
    'rebuild_entity_summaries': 'entity_summaries_rebuilt',
    'replay_dead_letters': 'dead_letters_replayed',
    'replay_to_graphiti': 'episodes_replayed',
}

# Alias map: alternative stat-key names that the LLM may emit, mapped to their
# canonical key. The verifier writes the observed value to BOTH the canonical
# key and each of its aliases so Stage 3's comparison sees a consistent picture.
_STAT_ALIASES: dict[str, str] = {
    'memories_written': 'memories_added',
}

# Stat keys the verifier writes back to stage stats. Includes virtual keys
# derived from ops rather than mapped 1:1 in _OP_TO_STAT (e.g.
# graphiti_writes_queued, which is the residual of add_memory ops where
# memory_ids is empty but stores includes graphiti), plus all alias keys.
_TRACKED_STAT_KEYS: frozenset[str] = (
    frozenset(_OP_TO_STAT.values())
    | {'graphiti_writes_queued'}
    | frozenset(_STAT_ALIASES)
)


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    return None


def _parse_result_summary(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _count_update_edge(op: dict) -> bool:
    """Return True only if the update_edge op was verified by a server-side readback.

    MemoryService.update_edge performs a ``get_edge_text`` round-trip after the
    save and sets ``verified=True`` in ``result_summary`` only when the returned
    fact matches. A missing ``verified`` key (legacy ops written before Guard 2
    was deployed) or ``verified=False`` (readback failed or returned a different
    fact) are both treated as unverified and excluded from ``edges_updated``.

    The strict ``is True`` identity check (not truthy) prevents strings like
    ``'true'`` or ``1`` from accidentally counting.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    return rs.get('verified') is True


def _count_add_memory(op: dict) -> bool:
    """Return True if the add_memory op actually produced a stored memory.

    Mem0 may dedup a write and return an empty ``memory_ids`` list. Only ops
    where ``memory_ids`` is a non-empty list count toward ``memories_added``.
    Graphiti-only async-enqueued writes (``memory_ids=[]``, ``stores=['graphiti']``)
    are tracked separately under ``graphiti_writes_queued`` via ``_count_graphiti_queued``.
    An op that reached neither store with a returned ID is a no-op here.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    memory_ids = rs.get('memory_ids')
    return bool(isinstance(memory_ids, list) and memory_ids)


def _count_graphiti_queued(op: dict) -> bool:
    """Return True if the add_memory op was a graphiti-only async enqueue.

    Graphiti writes are enqueued asynchronously and return no ``memory_ids``
    inline. An op counts as a graphiti-only enqueue iff it succeeded,
    ``memory_ids`` is empty (so it was not counted toward ``memories_added``),
    and ``stores``/``stores_written`` contains ``'graphiti'``. These are
    tallied separately under ``graphiti_writes_queued`` rather than
    ``memories_added`` to avoid inflating the memories count.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    memory_ids = rs.get('memory_ids')
    if isinstance(memory_ids, list) and memory_ids:
        # Non-empty IDs: this op is a memories_added, not a graphiti enqueue.
        return False
    stores = rs.get('stores') or rs.get('stores_written')
    return bool(isinstance(stores, list) and 'graphiti' in stores)


def _stage_window(report: StageReport | dict) -> tuple[datetime | None, datetime | None]:
    if isinstance(report, StageReport):
        return report.started_at, report.completed_at
    started = _parse_dt(report.get('started_at'))
    completed = _parse_dt(report.get('completed_at'))
    return started, completed


def _bucket_ops_by_stage(
    ops: Iterable[dict],
    stage_reports: dict[str, StageReport | dict],
) -> dict[str, list[dict]]:
    """Assign each op to the stage whose [started_at, completed_at] contains it.

    Ops outside every stage window fall into ``_unbucketed`` so nothing is
    silently lost.
    """
    windows: list[tuple[str, datetime, datetime]] = []
    for stage_id, report in stage_reports.items():
        started, completed = _stage_window(report)
        if started is None or completed is None:
            continue
        windows.append((stage_id, started, completed))

    buckets: dict[str, list[dict]] = {stage_id: [] for stage_id, _, _ in windows}
    buckets['_unbucketed'] = []

    for op in ops:
        created = _parse_dt(op.get('created_at'))
        if created is None:
            buckets['_unbucketed'].append(op)
            continue
        assigned = False
        for stage_id, started, completed in windows:
            if started <= created <= completed:
                buckets[stage_id].append(op)
                assigned = True
                break
        if not assigned:
            buckets['_unbucketed'].append(op)

    return buckets


def _observed_counts(ops: list[dict]) -> dict[str, int]:
    """Aggregate a list of ops into observed counts keyed by stat-key."""
    counts: dict[str, int] = {}
    for op in ops:
        # Only tally write_ops — backend_ops are a second audit layer of the
        # same write and would double-count.
        if op.get('layer') and op['layer'] != 'write_op':
            continue
        operation = op.get('operation')
        if not isinstance(operation, str):
            continue
        stat_key = _OP_TO_STAT.get(operation)
        if stat_key is None:
            continue
        if operation == 'add_memory':
            if _count_add_memory(op):
                counts[stat_key] = counts.get(stat_key, 0) + 1
            elif _count_graphiti_queued(op):
                # Graphiti-only async enqueue: no inline ID, but store accepted
                # the write. Track separately so it doesn't inflate memories_added.
                counts['graphiti_writes_queued'] = (
                    counts.get('graphiti_writes_queued', 0) + 1
                )
            # Either path accounts for the op — skip the generic counter.
            continue
        elif operation == 'update_edge':
            if not _count_update_edge(op):
                continue
        elif not op.get('success', 1):
            continue
        counts[stat_key] = counts.get(stat_key, 0) + 1
    return counts


def _apply_observed(
    report: StageReport | dict,
    observed: dict[str, int],
) -> None:
    """Overwrite stats in ``report`` with observed values, preserving originals.

    ``report`` is mutated in place. The stat-keys the verifier knows about are
    replaced by the observed counts; everything else is left as-is. The original
    values for replaced keys are stored under ``stats['_reported']``.
    """
    if isinstance(report, StageReport):
        stats = report.stats if isinstance(report.stats, dict) else {}
    else:
        stats = report.get('stats')
        if not isinstance(stats, dict):
            stats = {}
            report['stats'] = stats

    reported_snapshot: dict[str, Any] = {}

    # Write canonical keys from observed; aliases are written separately from
    # the same observed source so the two passes are order-independent.
    canonical_keys = _TRACKED_STAT_KEYS - frozenset(_STAT_ALIASES)
    for stat_key in canonical_keys:
        # Always record the observed value — zero is meaningful (means "no
        # writes happened for this op"). Only snapshot the original when it
        # actually existed, to avoid polluting with spurious None entries.
        if stat_key in stats:
            reported_snapshot[stat_key] = stats[stat_key]
        stats[stat_key] = observed.get(stat_key, 0)

    # Read from observed (not stats) so the pass is order-independent: even if a
    # future misconfiguration chains aliases, each alias resolves against the
    # immutable observed counts. _STAT_ALIASES values must be canonical keys
    # (enforced by a guard test).
    for alias_key, canonical_key in _STAT_ALIASES.items():
        if alias_key in stats:
            reported_snapshot[alias_key] = stats[alias_key]
        stats[alias_key] = observed.get(canonical_key, 0)

    if reported_snapshot:
        stats['_reported'] = reported_snapshot


async def verify_and_rewrite_stats(
    run_id: str,
    stage_reports: dict[str, StageReport | dict],
    write_journal: WriteJournal | None,
) -> dict[str, dict[str, int]]:
    """Cross-check self-reported stats against the write journal.

    Fetches write_ops tagged ``causation_id=run_id``, buckets them into stages
    by timestamp, and overwrites each stage's ``stats`` with observed counts.
    Originals are preserved under ``stats['_reported']``.

    Returns the per-stage observed-counts dict for logging/introspection. If the
    write_journal is unavailable, returns an empty dict and leaves stage_reports
    untouched.
    """
    if write_journal is None:
        return {}

    try:
        ops = await write_journal.get_ops_by_causation(run_id)
    except Exception as e:
        logger.warning(f'stats_verifier: failed to fetch ops for run {run_id}: {e}')
        return {}

    buckets = _bucket_ops_by_stage(ops, stage_reports)
    observed_by_stage: dict[str, dict[str, int]] = {}

    for stage_id, report in stage_reports.items():
        observed = _observed_counts(buckets.get(stage_id, []))
        _apply_observed(report, observed)
        observed_by_stage[stage_id] = observed

    unbucketed = buckets.get('_unbucketed') or []
    if unbucketed:
        logger.info(
            'stats_verifier: %d op(s) fell outside stage windows for run %s',
            len(unbucketed), run_id,
        )

    return observed_by_stage
