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


def _count_add_memory(op: dict) -> bool:
    """Return True if the add_memory op actually produced a stored memory.

    Mem0 may dedup a write and return an empty ``memory_ids`` list. Graphiti
    writes are enqueued asynchronously and don't produce IDs inline — we fall
    back to ``stores_written`` to know whether at least one backend accepted the
    write. An op that reached neither store is a no-op and must not count.
    """
    if not op.get('success', 1):
        return False
    rs = _parse_result_summary(op.get('result_summary'))
    memory_ids = rs.get('memory_ids')
    if isinstance(memory_ids, list) and memory_ids:
        return True
    stores = rs.get('stores') or rs.get('stores_written')
    return bool(isinstance(stores, list) and stores)


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
            if not _count_add_memory(op):
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
    for stat_key in set(_OP_TO_STAT.values()):
        # Always record the observed value — zero is meaningful (means "no
        # writes happened for this op"). Only snapshot the original when it
        # actually existed, to avoid polluting with spurious None entries.
        if stat_key in stats:
            reported_snapshot[stat_key] = stats[stat_key]
        stats[stat_key] = observed.get(stat_key, 0)

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
