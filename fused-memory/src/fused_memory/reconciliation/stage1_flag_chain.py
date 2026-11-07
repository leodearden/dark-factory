"""Stage 1's full-cycle flag chain: drop filters, suppression and marker acknowledgment.

``MemoryConsolidator.run`` hands a full cycle's flags to
:func:`run_stage1_flag_chain` after every flag-emitting step has appended its
flags. The chain runs, in order: the resolving filters in
:data:`RESOLVING_FILTERS_BEFORE_SUPPRESSION`; Hook A, the
entity-standing-decision suppression, then the per-cycle storm escape the
caller supplies; ``dedup_flags``, which annotates each flag against its
``stage1_flag_marker`` row; the resolving filters in
:data:`RESOLVING_FILTERS_AFTER_DEDUP`; marker acknowledgment; and gate-owned
``suggested_action`` normalization.

The acknowledgment rule. When a resolving filter drops a flag, the action the
flag requested is moot, so its persisted marker is acknowledged. Suppression is
not resolution. Hook A, and the suppression gate and cross-project fix-task gate
inside ``dedup_flags``, drop a flag because it is deliberately hidden or because
its fix is filed but not yet landed. Their signatures are excluded from
acknowledgment, so the marker keeps its recurrence history for the day the
suppression lifts. A step's kind decides which rule applies, not its position:
a new drop filter joins one of the two resolving tuples, and a new suppressing
step adds its drops to the suppressed set. A flag dropped before the chain
starts, by the preservation-specimen guard, is not in the acknowledgment
snapshot, so it is never acknowledged.

Gate-owned normalization runs last because it never drops a flag. It touches
only survivors, and the acknowledgment diff is complete before it can see the
rewrite. It also leaves marker identity alone: it rewrites only
``suggested_action``, and a gate-owned finding always carries the task id its
marker signature is keyed on.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fused_memory.models.reconciliation import ReconciliationEvent
from fused_memory.reconciliation.flag_dedup import (
    EntityStandingSuppressionResult,
    acknowledge_resolved_flags,
    compute_marker_signature,
    dedup_flags,
    filter_accounted_cluster_growth_flags,
    filter_already_recorded_caveat_flags,
    filter_already_tracked_systemic_patterns,
    filter_entity_standing_decisions,
    filter_false_absence_flags,
    filter_stale_bulk_get_statuses_flags,
    filter_stale_count_snapshot_corrections,
    filter_style_only_authorship_flags,
    filter_terminal_metadata_flags,
)
from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    extract_human_gated_task_ids,
    normalize_gate_owned_suggested_actions,
)
from fused_memory.reconciliation.sweep_deletion_guard import (
    filter_benign_sweep_deletion_flags,
)

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.services.memory_service import MemoryService

Flag = dict[str, Any]
SuppressionStormEscape = Callable[[EntityStandingSuppressionResult], Awaitable[None]]


@dataclass(frozen=True)
class FlagChainContext:
    """Everything one cycle's chain reads; the chain reads nothing else."""

    memory_service: MemoryService
    taskmaster: TaskBackendProtocol | None
    escalate_suppression_storm: SuppressionStormEscape
    project_id: str
    project_root: str
    known_projects: dict[str, str]
    run_id: str
    events: Sequence[ReconciliationEvent]
    active_tasks: Sequence[dict[str, Any]] | None


@dataclass(frozen=True)
class ResolvingFilter:
    """A drop filter whose drops are resolutions, published as *stat_key*."""

    stat_key: str
    apply: Callable[[list[Flag], FlagChainContext], Awaitable[list[Flag]]]


@dataclass(frozen=True)
class FlagChainOutcome:
    kept_flags: list[Flag]
    stats: Mapping[str, int]
    standing_decision_result: EntityStandingSuppressionResult


async def _drop_stale_count_snapshot_corrections(
    flags: list[Flag], ctx: FlagChainContext,
) -> list[Flag]:
    return filter_stale_count_snapshot_corrections(flags)


RESOLVING_FILTERS_BEFORE_SUPPRESSION: tuple[ResolvingFilter, ...] = (
    ResolvingFilter(  # task 1786
        'stale_count_snapshot_corrections_dropped', _drop_stale_count_snapshot_corrections,
    ),
    ResolvingFilter(  # task 1725
        'terminal_metadata_flags_dropped',
        lambda flags, ctx: filter_terminal_metadata_flags(
            taskmaster=ctx.taskmaster, project_root=ctx.project_root, flags=flags,
        ),
    ),
    ResolvingFilter(  # task 3007
        'stale_bulk_get_statuses_flags_dropped',
        lambda flags, ctx: filter_stale_bulk_get_statuses_flags(
            taskmaster=ctx.taskmaster, project_root=ctx.project_root, flags=flags,
        ),
    ),
    ResolvingFilter(  # tasks 2416, 4381: text coverage across every known project
        'systemic_pattern_already_tracked_dropped',
        lambda flags, ctx: filter_already_tracked_systemic_patterns(
            taskmaster=ctx.taskmaster, known_projects=ctx.known_projects, flags=flags,
        ),
    ),
    ResolvingFilter(  # task 3138
        'style_only_authorship_flags_dropped',
        lambda flags, ctx: filter_style_only_authorship_flags(
            memory_service=ctx.memory_service, project_id=ctx.project_id, flags=flags,
        ),
    ),
    ResolvingFilter(  # task 3476: ids resolve in the running project's root only
        'accounted_cluster_growth_flags_dropped',
        lambda flags, ctx: filter_accounted_cluster_growth_flags(
            taskmaster=ctx.taskmaster, project_root=ctx.project_root, flags=flags,
        ),
    ),
    ResolvingFilter(  # task 5271
        'already_recorded_caveat_flags_dropped',
        lambda flags, ctx: filter_already_recorded_caveat_flags(
            taskmaster=ctx.taskmaster, project_root=ctx.project_root, flags=flags,
        ),
    ),
    ResolvingFilter(  # task 5271
        'benign_sweep_deletion_flags_dropped',
        lambda flags, ctx: filter_benign_sweep_deletion_flags(
            memory_service=ctx.memory_service, project_id=ctx.project_id, flags=flags,
            events=ctx.events,
        ),
    ),
)

RESOLVING_FILTERS_AFTER_DEDUP: tuple[ResolvingFilter, ...] = (
    ResolvingFilter(  # fail-closed: a flag survives only when absence is confirmed
        'false_absence_flags_dropped',
        lambda flags, ctx: filter_false_absence_flags(
            taskmaster=ctx.taskmaster, project_root=ctx.project_root, flags=flags,
        ),
    ),
)


async def apply_resolving_filters(
    flags: list[Flag],
    ctx: FlagChainContext,
    filters: Iterable[ResolvingFilter],
) -> tuple[list[Flag], dict[str, int]]:
    """Run *filters* in order, each over the previous one's survivors.

    Returns the final survivors and each filter's drop count under its stat key.
    """
    stats: dict[str, int] = {}
    for resolving_filter in filters:
        before = len(flags)
        flags = await resolving_filter.apply(flags, ctx)
        stats[resolving_filter.stat_key] = before - len(flags)
    return flags, stats


async def run_stage1_flag_chain(flags: list[Flag], ctx: FlagChainContext) -> FlagChainOutcome:
    """Run the whole chain over a full cycle's *flags*; see the module docstring."""
    snapshot = list(flags)
    stats: dict[str, int] = {}

    flags, resolved_stats = await apply_resolving_filters(
        flags, ctx, RESOLVING_FILTERS_BEFORE_SUPPRESSION,
    )
    stats.update(resolved_stats)

    standing_decision_result = await _suppress_by_standing_decisions(flags, ctx)
    stats['entity_standing_decision_suppressed'] = sum(
        standing_decision_result.suppressed_by_decision.values()
    )
    suppressed = _dropped_signatures(flags, standing_decision_result.kept_flags, ctx.project_id)
    flags = standing_decision_result.kept_flags

    deduped, dedup_stats = await _dedup(flags, ctx)
    stats.update(dedup_stats)
    suppressed |= _dropped_signatures(flags, deduped, ctx.project_id)

    flags, resolved_stats = await apply_resolving_filters(
        deduped, ctx, RESOLVING_FILTERS_AFTER_DEDUP,
    )
    stats.update(resolved_stats)

    stats['stage1_flag_markers_acknowledged'] = await _acknowledge_resolved_drops(
        snapshot, flags, suppressed, ctx,
    )

    flags, stats['gate_owned_suggested_actions_normalized'] = _normalize_gate_owned_actions(
        flags, ctx,
    )
    return FlagChainOutcome(
        kept_flags=flags, stats=stats, standing_decision_result=standing_decision_result,
    )


async def _suppress_by_standing_decisions(
    flags: list[Flag], ctx: FlagChainContext,
) -> EntityStandingSuppressionResult:
    """Hook A, then the caller's per-cycle storm escape over its result."""
    result = await filter_entity_standing_decisions(
        memory_service=ctx.memory_service, project_id=ctx.project_id, flags=flags,
    )
    await ctx.escalate_suppression_storm(result)
    return result


async def _dedup(flags: list[Flag], ctx: FlagChainContext) -> tuple[list[Flag], dict[str, int]]:
    """dedup_flags and the stats it reports.

    The cross-project gate needs both taskmaster and known_projects, or it
    degrades to a silent no-op. Its count is published apart from the
    suppression gate's, because the signature diff merges the two causes.
    """
    dedup_counters: dict[str, Any] = {}
    deduped = await dedup_flags(
        memory_service=ctx.memory_service,
        project_id=ctx.project_id,
        run_id=ctx.run_id,
        flags=flags,
        taskmaster=ctx.taskmaster,
        known_projects=ctx.known_projects,
        stats=dedup_counters,
    )
    return deduped, {
        'stage1_flag_cross_project_fix_task_suppressed': int(
            dedup_counters.get('cross_project_fix_task_suppressed', 0)
        ),
        'stage1_flag_cross_project_fix_task_suppression_exhausted': int(
            dedup_counters.get('cross_project_fix_task_suppression_exhausted', 0)
        ),
        'stage1_completion_markers_self_deleted': sum(
            1 for f in deduped if f.get('completion_marker_self_deleted') is True
        ),
    }


async def _acknowledge_resolved_drops(
    snapshot: list[Flag],
    survivors: list[Flag],
    suppressed: set[tuple[str, str]],
    ctx: FlagChainContext,
) -> int:
    surviving = _marker_signatures(survivors, project_id=ctx.project_id)
    resolved = [
        f for f in snapshot
        if _marker_signatures([f], project_id=ctx.project_id) - surviving - suppressed
    ]
    return await acknowledge_resolved_flags(
        memory_service=ctx.memory_service,
        project_id=ctx.project_id,
        run_id=ctx.run_id,
        resolved_flags=resolved,
        mode='delete',
    )


def _normalize_gate_owned_actions(
    flags: list[Flag], ctx: FlagChainContext,
) -> tuple[list[Flag], int]:
    if ctx.active_tasks is None:
        return flags, 0
    gate_task_ids = extract_human_gated_task_ids(ctx.active_tasks)
    if not gate_task_ids:
        return flags, 0
    return normalize_gate_owned_suggested_actions(flags, gate_task_ids, project_id=ctx.project_id)


def _dropped_signatures(
    before: list[Flag], after: list[Flag], project_id: str,
) -> set[tuple[str, str]]:
    return _marker_signatures(before, project_id=project_id) - _marker_signatures(
        after, project_id=project_id,
    )


def _marker_signatures(flags: Iterable[Flag], *, project_id: str) -> set[tuple[str, str]]:
    """The ``stage1_flag_marker`` identities of *flags*, so every diff in the
    chain compares keys from one identity space."""
    return {
        sig for f in flags
        if (sig := compute_marker_signature(f, project_id=project_id)) is not None
    }
