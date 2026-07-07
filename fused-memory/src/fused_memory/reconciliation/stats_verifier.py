"""Override self-reported stage stats with write-journal-derived counters.

Stage agents self-report counts like ``memories_added: 3`` in their final
summary, but Mem0 silently deduplicates and returns an empty ``memory_ids``
list on filtered writes, so self-reported counts drift from what actually
happened. The write-shaped counters in a stage's stats are COMPUTED from the
write journal (see ``stage_stats.derive_stage_stats``), not self-reported by
the LLM. The LLM's self-reported numbers survive only under
``stats['_reported']`` as the judge's divergence signal.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.stage_stats import _COMPUTED_STAT_KEYS, derive_stage_stats

if TYPE_CHECKING:
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)


# The recognised LLM-reported counter namespace: every journal-computed key,
# plus 'memories_written' — a retired counter the LLM still sometimes emits.
# Any key in this set that appears in a stage's self-reported stats is popped
# into '_reported' as the judge's divergence signal. 'memories_written' is
# NOT in _COMPUTED_STAT_KEYS, so it is never written back — deleting the old
# alias mirror means it is dropped from the top-level stats entirely rather
# than kept in sync with memories_added.
_LLM_REPORTED_COUNTER_KEYS: frozenset[str] = _COMPUTED_STAT_KEYS | {'memories_written'}

# Every agent_id a stage's own writes can legitimately carry. Used to detect
# write_ops that claim to be a stage write (agent_id starts with
# 'recon-stage-') but don't match any real stage — e.g. a typo'd or stale
# stage id — so mis-tagging degrades to a visible log line instead of a
# silent 0 count (see the diagnostic in verify_and_rewrite_stats).
_KNOWN_STAGE_AGENT_IDS: frozenset[str] = frozenset(f'recon-stage-{sid.value}' for sid in StageId)


def _apply_observed(
    report: StageReport | dict,
    observed: dict[str, int],
) -> None:
    """Override a stage's write counters with journal-observed values.

    ``report`` is mutated in place. Only the recognised LLM-reported counter
    keys (``_LLM_REPORTED_COUNTER_KEYS``) present in ``stats`` are snapshotted
    into ``stats['_reported']`` before being removed — every other key (e.g.
    code-computed operational stats like ``graphiti_queue_health``) is left
    completely untouched. Every canonical computed key (``_COMPUTED_STAT_KEYS``)
    is then (re)written from ``observed``, 0-default, regardless of whether
    the stage originally reported it.
    """
    if isinstance(report, StageReport):
        stats = report.stats if isinstance(report.stats, dict) else {}
    else:
        stats = report.get('stats')
        if not isinstance(stats, dict):
            stats = {}
            report['stats'] = stats

    reported_snapshot: dict[str, Any] = {}
    for key in _LLM_REPORTED_COUNTER_KEYS:
        if key in stats:
            reported_snapshot[key] = stats.pop(key)

    for key in _COMPUTED_STAT_KEYS:
        stats[key] = observed.get(key, 0)

    if reported_snapshot:
        stats['_reported'] = reported_snapshot


async def verify_and_rewrite_stats(
    run_id: str,
    stage_reports: dict[str, StageReport | dict],
    write_journal: WriteJournal | None,
) -> dict[str, dict[str, int]]:
    """Cross-check self-reported stats against the write journal.

    Fetches all ops tagged ``causation_id=run_id`` once, groups the
    ``write_op``-layer entries by ``agent_id`` in a single pass, then for each
    stage derives that stage's own write counters from its
    ``agent_id=f'recon-stage-{stage_id}'`` slice — exact rather than a
    timestamp window — and overrides ``report.stats`` accordingly. Originals
    are preserved under ``stats['_reported']``.

    Returns the per-stage observed-counts dict for logging/introspection. If
    the write_journal is unavailable, returns an empty dict and leaves
    stage_reports untouched. Keys in ``stage_reports`` that aren't valid
    ``StageId`` values (e.g. the harness's ``'_error'`` entry) are skipped so
    error entries aren't polluted with spurious zero counters.

    Write ops whose ``agent_id`` looks like a stage tag (``recon-stage-*``)
    but matches no known ``StageId`` are logged at INFO — a mis-tagged or
    stale agent_id would otherwise silently zero out that stage's counters
    with no diagnostic trail.
    """
    if write_journal is None:
        return {}

    try:
        ops = await write_journal.get_ops_by_causation(run_id)
    except Exception as e:
        logger.warning(f'stats_verifier: failed to fetch ops for run {run_id}: {e}')
        return {}

    ops_by_agent: dict[str, list[dict]] = {}
    for op in ops:
        if op.get('layer') != 'write_op':
            continue
        agent_id = op.get('agent_id')
        if isinstance(agent_id, str):
            ops_by_agent.setdefault(agent_id, []).append(op)

    observed_by_stage: dict[str, dict[str, int]] = {}

    for stage_id, report in stage_reports.items():
        try:
            StageId(stage_id)
        except ValueError:
            continue
        stage_agent_id = f'recon-stage-{stage_id}'
        observed = derive_stage_stats(ops_by_agent.get(stage_agent_id, []), stage_agent_id)
        _apply_observed(report, observed)
        observed_by_stage[stage_id] = observed

    unmatched = sum(
        len(agent_ops)
        for agent_id, agent_ops in ops_by_agent.items()
        if agent_id.startswith('recon-stage-') and agent_id not in _KNOWN_STAGE_AGENT_IDS
    )
    if unmatched:
        logger.info(
            'stats_verifier: %d write_op(s) had a recon-stage-* agent_id that matched no known stage for run %s',
            unmatched, run_id,
        )

    return observed_by_stage
