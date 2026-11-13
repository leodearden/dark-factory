"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE1_DISALLOWED,
)
from fused_memory.reconciliation.consolidation_auto_executor import (
    auto_consolidation_initial_stats,
    run_auto_consolidation,
)
from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    extract_open_gate_task_ids,
    sweep_resolved_curator_gates,
)
from fused_memory.reconciliation.degenerate_task_node_sweep import (
    extract_terminal_task_ids,
    sweep_degenerate_task_nodes,
)
from fused_memory.reconciliation.episode_window import (
    UNFINISHED_WRITE_START_STAT,
    unfinished_write_start_stat,
)
from fused_memory.reconciliation.flag_dedup import EntityStandingSuppressionResult
from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    stamp_curator_gate_sweep_provenance,
)
from fused_memory.reconciliation.live_workflow_citation_guard import check_live_workflow_citations
from fused_memory.reconciliation.live_workflow_section import (
    CITATION_CONTRADICTIONS_STAT,
    LiveWorkflowSnapshot,
    build_live_workflow_snapshot,
)
from fused_memory.reconciliation.orphaned_recon_escalation_sweep import (
    sweep_orphaned_recon_escalations,
)
from fused_memory.reconciliation.overreach_flags import render_overreach_flags_section
from fused_memory.reconciliation.preservation_specimen_guard import (
    PreservationSuppressionResult,
    filter_preservation_specimen_flags,
    maybe_escalate_preservation_suppression_storm,
)
from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_pool_map import (
    STAGE1_CYCLE_SUMMARY_RECON_POOL as _STAGE1_CYCLE_SUMMARY_RECON_POOL,
)
from fused_memory.reconciliation.stage1_flag_chain import FlagChainContext, run_stage1_flag_chain
from fused_memory.reconciliation.stage1_payload_format import (
    append_note,
    format_context_fetch_failures,
    format_context_items,
    format_episodes,
    format_events,
    format_findings,
    format_memories,
    format_watermark,
    parse_instant,
    to_utc,
)
from fused_memory.reconciliation.stage1_stall_detector import (
    compute_stalled_task_ids,
    extract_human_operator_task_ids,
    extract_stalled_gate_backlog_task_ids,
    maybe_escalate_stalled_gate_backlog,
    maybe_escalate_stalled_tasks,
    track_human_operator_stalls,
)
from fused_memory.reconciliation.stages.base import BaseStage, RequiredSection
from fused_memory.reconciliation.stale_priority_override_edge_sweep import (
    PRIORITY_OVERRIDE_ENUMERATION_COMPLETE_STAT_KEY,
    PRIORITY_OVERRIDE_ENUMERATION_INCOMPLETE_KIND_STAT_KEY,
    sweep_stale_priority_override_edges,
)
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY,
    STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY,
    sweep_stale_status_snapshot_edges,
)
from fused_memory.reconciliation.standing_decision_constants import (
    SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
)
from fused_memory.reconciliation.standing_decision_storm_escape import (
    maybe_escalate_suppression_storm,
    maybe_escalate_suppression_streak,
    update_suppression_streaks,
)
from fused_memory.reconciliation.summary_pool import (
    write_cycle_summary,
)
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    detect_census_inconsistency,
    format_filtered_task_tree,
)
from fused_memory.reconciliation.tree_read_instant import resolve_tree_read_instant
from fused_memory.reconciliation.unlanded_path_check import (
    UnlandedPathCheckResult,
    check_unlanded_path_claims,
)
from fused_memory.utils.clip_disclosure import clip_note, fetch_cap_note

if TYPE_CHECKING:
    from fused_memory.services.memory_service import MemoryService
    from fused_memory.services.overreach_shadow import OverreachFlagPage

logger = logging.getLogger(__name__)

# Stage 1 per-cycle summary pool cap and related constants.
# The deterministic Mem0 mirror write (write_cycle_summary, task 2229) tags
# recon_pool='stage1_cycle_summary' — _STAGE1_CYCLE_SUMMARY_RECON_POOL is
# imported above from the shared leaf module recon_pool_map.py (task 2140),
# not redefined here. After each write, write_cycle_summary's internal
# enforce_summary_pool_cap call trims the pool to at most this many members by
# deleting the OLDEST entries — deterministically via Qdrant scroll, NOT semantic search.
STAGE1_CYCLE_SUMMARY_POOL_CAP: int = 2
_STAGE1_CYCLE_SUMMARY_TRIM_SOURCE = 'stage1_cycle_summary_trim'

_T = TypeVar('_T')

# (stat key, sweep result key) for each full-cycle sweep, in publication order.
_StatMap = tuple[tuple[str, str], ...]
_CURATOR_GATE_SWEEP_STATS: _StatMap = (
    ('curator_gate_resolution_scanned', 'scanned'),
    ('curator_gate_resolution_errors', 'errors'),
)
_CURATOR_GATE_FLAGS_EMITTED_STAT = 'curator_gate_resolution_flags_emitted'
_ORPHAN_SWEEP_STATS: _StatMap = (
    ('orphaned_recon_escalations_scanned', 'scanned'),
    ('orphaned_recon_escalations_terminal', 'terminal'),
    ('orphaned_recon_escalations_missing', 'missing'),
    ('orphaned_recon_escalations_live', 'live'),
    ('orphaned_recon_escalations_ambiguous', 'ambiguous'),
    ('orphaned_recon_escalations_unresolvable', 'unresolvable'),
    ('orphaned_recon_escalations_errors', 'errors'),
)
_ORPHAN_FLAGS_EMITTED_STAT = 'orphaned_recon_escalations_flags_emitted'
_DEGENERATE_SWEEP_STATS: _StatMap = (
    ('degenerate_task_nodes_swept', 'deleted'),
    ('degenerate_task_nodes_scanned', 'scanned'),
)
_STATUS_SNAPSHOT_SWEEP_STATS: _StatMap = (
    ('stale_status_snapshot_edges_invalidated', 'invalidated'),
    ('stale_status_snapshot_edges_scanned', 'scanned'),
    ('stale_status_snapshot_edges_candidates', 'candidate_edges'),
    ('stale_status_snapshot_edges_errors', 'errors'),
    ('stale_blocked_edges_superseded', 'superseded'),
    ('stale_blocked_edges_supersede_errors', 'supersede_errors'),
    ('stale_blocked_edges_supersede_skipped', 'supersede_skipped'),
    (STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY, 'enumeration_complete'),
    (STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY, 'enumeration_incomplete_kind'),
)
_PRIORITY_OVERRIDE_SWEEP_STATS: _StatMap = (
    ('stale_priority_override_edges_invalidated', 'invalidated'),
    ('stale_priority_override_edges_scanned', 'scanned'),
    (PRIORITY_OVERRIDE_ENUMERATION_COMPLETE_STAT_KEY, 'enumeration_complete'),
    (PRIORITY_OVERRIDE_ENUMERATION_INCOMPLETE_KIND_STAT_KEY, 'enumeration_incomplete_kind'),
)


async def write_stage1_cycle_summary(
    memory_service: MemoryService,
    project_id: str,
    report: StageReport,
    run_id: str,
) -> bool:
    """Write Stage 1's authoritative per-cycle ``cycle_summary`` ledger row.

    Thin Stage-1-parametrized wrapper around
    :func:`~fused_memory.reconciliation.summary_pool.write_cycle_summary`,
    binding the four Stage-1-specific values (``stage='memory_consolidator'``,
    ``recon_pool``, ``trim_source``, ``cap``) in exactly one place (task 2440).
    Extracted from the call formerly inlined in :meth:`MemoryConsolidator.run`
    (task 2229 W5-λ) so both the in-stage fast-path write (``run()`` below)
    and the harness-level raise-path backstop
    (``ReconciliationHarness._ensure_stage1_cycle_summary``, task 2440) share
    the identical binding — a future change to any of the four constants can
    never silently diverge between the two call sites.

    Returns:
        ``True`` when the authoritative ledger upsert succeeded, ``False``
        otherwise — see :func:`~fused_memory.reconciliation.summary_pool.write_cycle_summary`'s
        "Returns" section for the full contract (the Mem0 mirror and pool-cap
        trim run unconditionally regardless of this return value).
    """
    return await write_cycle_summary(
        memory_service,
        project_id,
        report,
        run_id,
        stage='memory_consolidator',
        recon_pool=_STAGE1_CYCLE_SUMMARY_RECON_POOL,
        trim_source=_STAGE1_CYCLE_SUMMARY_TRIM_SOURCE,
        cap=STAGE1_CYCLE_SUMMARY_POOL_CAP,
    )


class MemoryConsolidator(BaseStage):
    """Stage 1: Review and consolidate memories across Graphiti and Mem0."""

    # ── Inference-bearing payload sections (task 4708) ──────────────────────
    # INCLUSION CRITERION — a section belongs here iff the shipped Stage-1 prompt
    # tells the model to draw an inference from that section's ABSENCE. Today
    # prompts/stage1.py says: "If `### Live-Workflow Signals` is absent from the
    # payload, no task the `### Active Task Tree` lists is live this cycle —
    # neither through a per-task signal nor through the project-wide lock — and
    # none of them has landing evidence; no live-workflow suppression applies to
    # them …". That makes absence load-bearing: a
    # payload builder that omits the section does not merely produce a terser
    # payload, it makes the model conclude something FALSE. Every builder therefore renders these
    # via _render_required_sections() — adding a section is ONE edit here, not
    # one edit per builder. Enforced by
    # tests/reconciliation/test_stage1_payload_section_parity.py.
    #
    # Replaces the drift mechanism behind three hand-fixed instances of the same
    # defect — tasks 2150, 2552 and 3839 each wired ONE section into ONE missed
    # builder after the fact.
    #
    # Deliberately NOT registered:
    #   * _build_task_tree_section — 2 of 3 by design; registering it would dump
    #     the whole task tree into the findings-only remediation payload.
    #   * _build_task_count_census_section — 2 of 3 by design; its own contract
    #     returns '' when task_count_verification is None, which is exactly the
    #     remediation-pass state, so registering it would be a no-op.
    #   * _build_project_root_directive — required in all three payloads, but as
    #     an unconditional directive with NO absence-inference in any prompt, so
    #     it falls outside this tuple's inclusion criterion. It keeps its own
    #     dedicated tests instead (task 2552).
    #   * render_overreach_flags_section — 2 of 3 by design (task 6137): the
    #     shadow's flags are a full-cycle handoff, and no prompt draws an
    #     inference from the section's absence.
    REQUIRED_SECTIONS: tuple[RequiredSection, ...] = (
        RequiredSection('### Live-Workflow Signals', '_build_live_workflow_section'),
    )

    # Tier limits — set by harness before run(); None until explicitly assigned
    episode_limit: int | None = None
    memory_limit: int | None = None

    # Remediation support — set by harness for second pass
    remediation_findings: list[dict] | None = None
    prior_s3_findings: list[dict] | None = None

    # Cycle fence — set by harness to protect targeted-recon writes
    cycle_fence_time: datetime | None = None

    # Token-budget assembled payload — set by harness when using ContextAssembler.
    # When set, assemble_payload() uses this instead of the generic time-windowed fetch.
    assembled_payload: AssembledPayload | None = None

    # Active task tree — set by harness before run() (task 455)
    filtered_task_tree: FilteredTaskTree | None = None
    # Set with filtered_task_tree: the instant that tree, and every heartbeat_at in it, was read.
    filtered_task_tree_fetched_at: datetime | None = None

    # Authoritative task-count cross-verification record — set by harness (task 1785).
    # None when the status map was unavailable (e.g. taskmaster down) or in remediation passes.
    task_count_verification: dict | None = None

    # Graphiti async-queue health record — set by harness (task 1785).
    # None when durable_queue was unavailable or in remediation passes.
    graphiti_queue_health: dict | None = None

    # FalkorDB index-provisioning health record — set by harness (task 3709).
    # None when the graph was unreadable, absent, or in remediation passes.
    index_health: dict | None = None

    # The overreach shadow's next page of flags not yet handed over — set by harness
    # (design.md §6.1 O4, task 6137). None when unavailable or in remediation passes.
    overreach_flags: OverreachFlagPage | None = None

    # Cached project_status_correction memory vs. live get_statuses census
    # diff/supersede record — set by harness (task 1938).
    # None when the status map was unavailable or in remediation passes.
    status_correction_reconciliation: dict | None = None

    # Count of snapshot lines stripped from the payload in the current cycle (task 1547)
    _entity_summary_snapshot_lines_stripped: int = 0

    # Fetch degraded sources for the current run — reset at the top of assemble_payload
    # and _format_assembled_payload; copied to report.stats['stage1_fetch_degraded'] in run()
    # (task 1812). Empty list = genuine empty corpus; non-empty = fetch failure on that source.
    # 'event_context' means the ContextAssembler could not fetch some event's context (task 5370).
    # Initialized per-instance in __init__ to avoid the shared-mutable-default hazard.
    _fetch_degraded_sources: list

    # Count of episode/Mem0 records excluded from the "new" lists in the current
    # cycle because their timestamp was missing, empty, or unparseable (task 4574).
    # Reset at the top of assemble_payload and _format_assembled_payload, mirroring
    # _fetch_degraded_sources; copied to report.stats['stage1_undatable_freshness_records']
    # in run(). These records are excluded (never re-surfaced forever like 894fbe90)
    # but counted and logged at the failure point rather than silently dropped —
    # design invariant INV-2 structured-facts-at-failure.
    _undatable_freshness_records: int

    # Sampled just before the episode fetch; see reconciliation/episode_window.py.
    _unfinished_graphiti_write_start: datetime | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance initialisation: avoids shared-mutable-default hazard.
        # A class-level [] would be mutated in-place by any append() that ran before
        # the run-local reset in assemble_payload / _format_assembled_payload, leaking
        # state across all instances (and across reused-instance runs).
        self._fetch_degraded_sources: list = []
        self._undatable_freshness_records: int = 0
        self._unfinished_graphiti_write_start: datetime | None = None
        self._live_workflow_snapshot: LiveWorkflowSnapshot | None = None

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
    ) -> StageReport:
        """Run Stage 1's agent, then guard, extend and post-process its report.

        Phantom-citation verification already ran inside ``super().run()``, for
        every stage; repeating it here would double-count each citation stat.
        The two guards run above the remediation early-return because a
        remediation pass needs them too. A remediation pass re-emits a curated
        flag list, so it skips the flag chain and every full-cycle step.

        The flag-emitting steps (curator gate, orphan sweep, unlanded check)
        run above the flag chain, so their flags earn ``stage1_flag_marker``
        rows and can fire the chain on a cycle where the agent flagged nothing.
        """
        self._live_workflow_snapshot = None
        report = await super().run(
            events, watermark, prior_reports, run_id, model=model,
        )
        self._publish_always_present_stats(report)
        preservation_result = await self._apply_preservation_specimen_guard(report)
        self._apply_live_workflow_citation_guard(report)
        if self.remediation_findings is not None:
            return report

        await self._escalate_preservation_storm(preservation_result, run_id)
        await self._append_curator_gate_sweep_flags(report, run_id)
        await self._run_auto_consolidation(report, run_id)
        await self._append_orphaned_recon_escalation_flags(report, run_id)
        await self._append_unlanded_path_flags(report, events)
        standing_decision_result = await self._run_flag_chain(report, events, run_id)
        await self._account_suppression_streaks(run_id, standing_decision_result, report)
        self._detect_census_inconsistency(report, events, run_id)
        self._surface_harness_health_records(report, run_id)
        await self._escalate_human_operator_stalls(report, run_id)
        await self._escalate_stalled_gate_backlog(report, run_id)
        await self._run_graph_sweeps(report, run_id)

        ledger_written = await write_stage1_cycle_summary(
            self.memory, self.project_id, report, run_id,
        )
        report.stats['stage1_cycle_summary_ledger_written'] = 1 if ledger_written else 0
        return report

    def _publish_always_present_stats(self, report: StageReport) -> None:
        """Publish, at its default, every stat key a Stage 1 report always carries.

        Stage 1's stats blob is serialized verbatim into Stage 2's prompt
        (``stages/task_knowledge_sync.py::_format_report``), so each key
        published here is present on every report, remediation passes
        included; the full-cycle steps overwrite their own. Other keys stay
        absent until the step that observes them publishes them, because a
        default there would claim an observation that never happened;
        ``_run_graph_sweeps`` states the case. ``stage1_cycle_summary_ledger_written``
        reflects only the ledger upsert, never the Mem0 mirror that
        ``write_cycle_summary`` also attempts.
        """
        report.stats.update({
            'entity_summary_snapshot_lines_stripped': self._entity_summary_snapshot_lines_stripped,
            'stage1_fetch_degraded': self._fetch_degraded_sources or [],
            'stage1_assembled_budget': (
                self.assembled_payload.budget_report()
                if self.assembled_payload is not None
                else None
            ),
            'stage1_undatable_freshness_records': self._undatable_freshness_records,
            UNFINISHED_WRITE_START_STAT: unfinished_write_start_stat(
                self._unfinished_graphiti_write_start
            ),
            'stage1_completion_markers_self_deleted': 0,
            'stage1_cycle_summary_ledger_written': 0,
            **_zeroed(_CURATOR_GATE_SWEEP_STATS),
            _CURATOR_GATE_FLAGS_EMITTED_STAT: 0,
            **_zeroed(_ORPHAN_SWEEP_STATS),
            _ORPHAN_FLAGS_EMITTED_STAT: 0,
            **UnlandedPathCheckResult.empty().stats(),
            'entity_standing_decision_suppressed': 0,
            'entity_standing_decision_max_suppression_streak': 0,
            'preservation_specimen_suppressed': 0,
            'preservation_specimen_unresolved': 0,
            'preservation_specimen_citations': {},
            CITATION_CONTRADICTIONS_STAT: 0,
            'gate_owned_suggested_actions_normalized': 0,
            **auto_consolidation_initial_stats(),
        })

    async def _apply_preservation_specimen_guard(
        self, report: StageReport,
    ) -> PreservationSuppressionResult | None:
        """Drop stranded/reset findings about a documented preservation specimen (task 4223).

        Best-effort: on failure the flags pass through unfiltered, the stats
        keep their defaults, and None is returned. The unresolved count and
        the citations travel with the suppressed count, so a degraded or
        stale suppression is visible to Stage 2, not only in the log.
        """
        result = None
        try:
            result = await filter_preservation_specimen_flags(
                memory_service=self.memory,
                project_id=self.project_id,
                flags=report.items_flagged,
            )
            report.items_flagged = result.kept_flags
            report.stats['preservation_specimen_suppressed'] = sum(
                result.suppressed_by_task.values()
            )
            report.stats['preservation_specimen_unresolved'] = len(result.unresolved_task_ids)
            report.stats['preservation_specimen_citations'] = dict(result.citations_by_task)
        except Exception:
            logger.exception(
                'preservation-specimen guard failed for project %s (best-effort — '
                'passing %d flag(s) through unfiltered this cycle)',
                self.project_id, len(report.items_flagged or []),
            )
        return result

    def _apply_live_workflow_citation_guard(self, report: StageReport) -> None:
        """Annotate, never drop, a finding citing a signal this run's section did not render.

        Task 4874; the rationale is live_workflow_citation_guard's module docstring.
        """
        try:
            report.items_flagged, report.stats[CITATION_CONTRADICTIONS_STAT] = (
                check_live_workflow_citations(report.items_flagged or [], self._live_workflow_snapshot)
            )
        except Exception:
            logger.exception('live-workflow citation guard failed for project %s', self.project_id)

    async def _escalate_preservation_storm(
        self, preservation_result: PreservationSuppressionResult | None, run_id: str,
    ) -> None:
        """The preservation guard's per-cycle storm escape, with its unreadable-corroboration disclosure.

        It runs whether or not any flag survived: a storm is the cycle in which
        the guard may have suppressed every flag.
        """
        if self._escalation_queue is None or preservation_result is None:
            return
        await maybe_escalate_preservation_suppression_storm(
            escalation_queue=self._escalation_queue,
            project_id=self.project_id,
            run_id=run_id,
            result=preservation_result,
        )

    async def _append_curator_gate_sweep_flags(self, report: StageReport, run_id: str) -> None:
        """Flag open curator gates whose ruling is already in Mem0, for Stage 2 to close (task 3084).

        These flags skip the citation verification inside ``super().run()``, but
        their ids resolve by construction: they come from a same-cycle Qdrant
        scroll, and the sweep emits no flag unless that scroll returned a row.
        """
        if self.filtered_task_tree is None:
            return
        active_tasks = self.filtered_task_tree.active_tasks
        gate_ids = extract_open_gate_task_ids(active_tasks)
        swept = set(gate_ids)
        # Titles for the flag descriptions only; selection stays with extract_open_gate_task_ids.
        gate_tasks_by_id = {
            str(task.get('id')): task
            for task in active_tasks
            if isinstance(task, dict) and str(task.get('id')) in swept
        }
        gate_sweep = await _await_best_effort(
            lambda: sweep_resolved_curator_gates(
                self.memory, self.project_id, gate_ids, tasks_by_id=gate_tasks_by_id,
            ),
            event='reconciliation.curator_gate_resolution_sweep_failed',
            project_id=self.project_id,
            run_id=run_id,
            gate_id_count=len(gate_ids),
        )
        if gate_sweep is None:
            return
        report.items_flagged = (report.items_flagged or []) + (
            stamp_curator_gate_sweep_provenance(gate_sweep['flags'])
        )
        _publish_sweep_stats(report, _CURATOR_GATE_SWEEP_STATS, gate_sweep)
        report.stats[_CURATOR_GATE_FLAGS_EMITTED_STAT] = len(gate_sweep['flags'])

    async def _run_auto_consolidation(self, report: StageReport, run_id: str) -> None:
        """The auto-consolidation executor (task 5240); contract: plans/memory-auto-consolidation-prd.md C4."""
        await _await_best_effort(
            lambda: run_auto_consolidation(
                self.memory,
                self.taskmaster,
                self._escalation_queue,
                project_id=self.project_id,
                project_root=self.project_root,
                run_id=run_id,
                config=self.memory.config.consolidation_auto,
                active_tasks=self._active_tasks(),
                report=report,
            ),
            event='reconciliation.auto_consolidation_failed',
            project_id=self.project_id,
            run_id=run_id,
        )

    async def _append_orphaned_recon_escalation_flags(
        self, report: StageReport, run_id: str,
    ) -> None:
        """Flag pending L1 recon stale records whose subject task went terminal or vanished (task 3052).

        Detection only: the recon-escalation watcher is the sole closer. Flags
        name this project's orphans while the counts cover the whole shared
        queue; sweep_orphaned_recon_escalations states the trade-off.
        """
        if self._escalation_queue is None or self.taskmaster is None:
            return
        orphan_sweep = await _await_best_effort(
            lambda: sweep_orphaned_recon_escalations(
                self._escalation_queue,
                self.taskmaster,
                self.known_projects,
                running_project_id=self.project_id,
            ),
            event='reconciliation.orphaned_recon_escalation_sweep_failed',
            project_id=self.project_id,
            run_id=run_id,
        )
        if orphan_sweep is None:
            return
        report.items_flagged = (report.items_flagged or []) + orphan_sweep['flags']
        _publish_sweep_stats(report, _ORPHAN_SWEEP_STATS, orphan_sweep)
        report.stats[_ORPHAN_FLAGS_EMITTED_STAT] = len(orphan_sweep['flags'])

    async def _append_unlanded_path_flags(
        self, report: StageReport, events: list[ReconciliationEvent],
    ) -> None:
        """Flag done-task memories naming paths their branch added then removed before merge (task 5626).

        The check guards itself, so it needs no best-effort wrapper; see
        reconciliation/unlanded_path_check.py.
        """
        unlanded = await check_unlanded_path_claims(
            self.memory, self.project_id, self.project_root, events,
        )
        report.items_flagged = (report.items_flagged or []) + list(unlanded.flags)
        report.stats.update(unlanded.stats())

    async def _run_flag_chain(
        self, report: StageReport, events: list[ReconciliationEvent], run_id: str,
    ) -> EntityStandingSuppressionResult:
        """Run reconciliation/stage1_flag_chain.py over the cycle's flags; return Hook A's result.

        ``stage1_flag_markers_acknowledged`` is published on every full cycle,
        symmetric with Stage 2's ``stage2_flag_markers_acknowledged``. A cycle
        that flagged nothing returns the empty batch: no decision drained anything.
        """
        report.stats['stage1_flag_markers_acknowledged'] = 0
        if not report.items_flagged:
            return EntityStandingSuppressionResult.empty_batch()
        outcome = await run_stage1_flag_chain(
            report.items_flagged, self._flag_chain_context(events, run_id),
        )
        report.items_flagged = outcome.kept_flags
        report.stats.update(outcome.stats)
        return outcome.standing_decision_result

    def _flag_chain_context(
        self, events: list[ReconciliationEvent], run_id: str,
    ) -> FlagChainContext:
        return FlagChainContext(
            memory_service=self.memory,
            taskmaster=self.taskmaster,
            escalate_suppression_storm=functools.partial(
                self._escalate_suppression_storm, run_id=run_id,
            ),
            project_id=self.project_id,
            project_root=self.project_root,
            known_projects=self.known_projects,
            run_id=run_id,
            events=events,
            active_tasks=self._active_tasks(),
        )

    async def _escalate_suppression_storm(
        self, result: EntityStandingSuppressionResult, run_id: str,
    ) -> None:
        """Hook A's per-cycle storm escape, handed to the flag chain."""
        if self._escalation_queue is None:
            return
        await maybe_escalate_suppression_storm(
            escalation_queue=self._escalation_queue,
            project_id=self.project_id,
            run_id=run_id,
            result=result,
            threshold=SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
        )

    def _active_tasks(self) -> list[dict] | None:
        if self.filtered_task_tree is None:
            return None
        return self.filtered_task_tree.active_tasks

    def _detect_census_inconsistency(
        self, report: StageReport, events: list[ReconciliationEvent], run_id: str,
    ) -> None:
        """Record event task ids above the census max, the signature of a partial or wrong-source bulk read.

        Skipped when the max is 0 (an empty tree or no parseable id), where every
        id would read as above it: a false-positive flood, not a truncation signal.
        """
        tree = self.filtered_task_tree
        if tree is None or tree.max_task_id <= 0:
            return
        event_task_ids = [
            e.payload.get('task_id')
            for e in events
            if e.payload.get('task_id') is not None
        ]
        inconsistent = detect_census_inconsistency(tree.max_task_id, event_task_ids)
        if not inconsistent:
            return
        report.stats['task_tree_census_inconsistent'] = inconsistent
        logger.warning(
            'reconciliation.task_tree_census_inconsistent',
            extra={
                'project_id': self.project_id,
                'run_id': run_id,
                'census_max': tree.max_task_id,
                'offending_ids': inconsistent,
            },
        )

    def _surface_harness_health_records(self, report: StageReport, run_id: str) -> None:
        """Copy each harness-set health record into the report, warning when it is unhealthy.

        Surfacing only. Index drift is escalated by the harness detector that
        the startup sweep shares, so the two paths never fork (task 3709).
        """
        if self.task_count_verification is not None:
            report.stats['task_count_verification'] = self.task_count_verification
            if not self.task_count_verification.get('consistent', True):
                logger.warning(
                    'reconciliation.task_count_divergence',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'authoritative': self.task_count_verification.get('authoritative'),
                        'tree': self.task_count_verification.get('tree'),
                    },
                )
        if self.graphiti_queue_health is not None:
            report.stats['graphiti_queue_health'] = self.graphiti_queue_health
            if not self.graphiti_queue_health.get('healthy', True):
                logger.warning(
                    'reconciliation.graphiti_queue_unhealthy_stage1',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'dead_count': self.graphiti_queue_health.get('dead_count'),
                    },
                )
        if self.index_health is not None:
            report.stats['index_health'] = self.index_health
            if not self.index_health.get('healthy', True):
                logger.warning(
                    'reconciliation.index_drift_stage1',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'group_id': self.project_id,
                        'missing_count': len(self.index_health.get('missing') or []),
                        'expected_total': self.index_health.get('expected_total'),
                    },
                )
        if self.status_correction_reconciliation is not None:
            report.stats['status_correction_reconciliation'] = (
                self.status_correction_reconciliation
            )

    async def _escalate_human_operator_stalls(self, report: StageReport, run_id: str) -> None:
        """Escalate human-operator-required findings that keep recurring unactioned (task 1201).

        Skipped without an escalation queue: stall markers nothing would
        consume only waste Qdrant capacity.
        """
        if self._escalation_queue is None:
            return
        hor_task_ids = extract_human_operator_task_ids(report.items_flagged or [])
        if not hor_task_ids:
            return
        stall_counts = await track_human_operator_stalls(
            memory_service=self.memory,
            project_id=self.project_id,
            run_id=run_id,
            task_ids=hor_task_ids,
        )
        stalled = compute_stalled_task_ids(stall_counts)
        report.stats['stage1_human_operator_stalled'] = len(stalled)
        if not stalled:
            report.stats['stage1_human_operator_escalated'] = 0
            return
        escalated = await maybe_escalate_stalled_tasks(
            escalation_queue=self._escalation_queue,
            project_id=self.project_id,
            run_id=run_id,
            stalled_task_ids=stalled,
            stall_counts=stall_counts,
            flags=report.items_flagged or [],
        )
        report.stats['stage1_human_operator_escalated'] = len(escalated)

    async def _escalate_stalled_gate_backlog(self, report: StageReport, run_id: str) -> None:
        """Escalate blocked human-decision gates whose gate_escalated_at stamp is over 48h old (task 3017).

        Stateless: it reads the durable stamp off the active task tree, so it
        needs no Mem0 marker and no LLM finding.
        """
        if self._escalation_queue is None or self.filtered_task_tree is None:
            return
        active_tasks = self.filtered_task_tree.active_tasks
        now = datetime.now(UTC)
        gate_stalled = extract_stalled_gate_backlog_task_ids(active_tasks, now=now)
        report.stats['stage1_gate_backlog_stalled'] = len(gate_stalled)
        if not gate_stalled:
            report.stats['stage1_gate_backlog_escalated'] = 0
            return
        task_by_id = {str(t.get('id')): t for t in active_tasks if t.get('id') is not None}
        gate_escalated = await maybe_escalate_stalled_gate_backlog(
            escalation_queue=self._escalation_queue,
            project_id=self.project_id,
            run_id=run_id,
            stalled_task_ids=gate_stalled,
            task_by_id=task_by_id,
            now=now,
        )
        report.stats['stage1_gate_backlog_escalated'] = len(gate_escalated)

    async def _run_graph_sweeps(self, report: StageReport, run_id: str) -> None:
        """Run the three Graphiti sweeps: degenerate task nodes, stale status snapshots, stale priority overrides.

        Each is best-effort, and a sweep that raised publishes none of its
        stats. Absent is honest there, where a False ``enumeration_complete``
        would claim a corpus was observed and found incomplete.
        """
        terminal_ids = extract_terminal_task_ids(self.filtered_task_tree)
        if terminal_ids:
            _publish_sweep_stats(report, _DEGENERATE_SWEEP_STATS, await _await_best_effort(
                lambda: sweep_degenerate_task_nodes(self.memory, self.project_id, terminal_ids),
                event='reconciliation.degenerate_task_node_sweep_failed',
                project_id=self.project_id,
                run_id=run_id,
                terminal_id_count=len(terminal_ids),
            ))
        _publish_sweep_stats(report, _STATUS_SNAPSHOT_SWEEP_STATS, await _await_best_effort(
            lambda: sweep_stale_status_snapshot_edges(
                self.memory, self.taskmaster, self.project_id, self.project_root, run_id=run_id,
            ),
            event='reconciliation.stale_status_snapshot_edge_sweep_failed',
            project_id=self.project_id,
            run_id=run_id,
        ))
        _publish_sweep_stats(report, _PRIORITY_OVERRIDE_SWEEP_STATS, await _await_best_effort(
            lambda: sweep_stale_priority_override_edges(
                self.memory, self.project_id, self.project_root, run_id=run_id,
            ),
            event='reconciliation.stale_priority_override_edge_sweep_failed',
            project_id=self.project_id,
            run_id=run_id,
        ))

    async def _account_suppression_streaks(
        self,
        run_id: str,
        esd_result: EntityStandingSuppressionResult,
        report: StageReport,
    ) -> None:
        """Advance every standing decision's suppression streak and file the sustained drains.

        The storm escape's cross-cycle arm (task 2943).  The per-cycle escape
        catches one decision suppressing a flood in ONE cycle; this one catches
        a sustained drain — a streak of at least K consecutive full cycles
        whose last K suppressed more than N flags in total, not counting a
        cycle the per-cycle escape already reported — which can hide a
        genuinely new finding as surely as a single-cycle flood.  One flag per
        cycle, which is how a decision that works behaves, totals K <= N and
        never files.

        It runs on EVERY full cycle, zero-flag ones included, which is why
        run() calls it after the flag chain rather than from inside it.  A
        cycle that flagged nothing is one in which no decision drained
        anything, so it must reset every streak; skipping it would let a
        decision suppressing on alternate cycles reach K with gaps in between.
        On such a cycle *esd_result* is the empty batch _run_flag_chain returns.

        The state update sits outside the escalation-queue gate so the streak is
        already real when a queue is wired; only the filing needs the queue.
        Both callees are best-effort and never raise.  The updater's per-cycle
        cutoff is the N that _escalate_suppression_storm hands the per-cycle
        escape, so a cycle that escape reported is never counted a second time.
        """
        updates = await update_suppression_streaks(
            memory_service=self.memory,
            project_id=self.project_id,
            run_id=run_id,
            result=esd_result,
            per_cycle_threshold=SUPPRESSION_STORM_THRESHOLD_PER_CYCLE,
        )
        report.stats['entity_standing_decision_max_suppression_streak'] = max(
            (update.streak for update in updates), default=0
        )
        if self._escalation_queue is not None:
            await maybe_escalate_suppression_streak(
                escalation_queue=self._escalation_queue,
                project_id=self.project_id,
                run_id=run_id,
                updates=updates,
            )

    def get_system_prompt(self) -> str:
        return STAGE1_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE1_DISALLOWED

    def _filter_records_newer_than_watermark(
        self,
        records: list[dict],
        watermark: datetime,
        *,
        source: str,
        id_key: str,
        get_raw_ts: Callable[[dict], str | None],
    ) -> list[dict]:
        """Return the subset of *records* strictly newer than *watermark*.

        Parses each record's raw timestamp exactly once via ``parse_instant``
        and compares it against *watermark* normalized to UTC exactly once per
        call (hoisted out of the loop as ``wm_utc``) — the same two steps
        ``stage1_payload_format.py::is_newer_than_watermark`` (task 4574)
        performs per call, inlined here so a record's timestamp is never
        parsed twice and the watermark is never re-normalized per record.
        This method does not call that predicate: it needs the parsed instant
        itself to tell an undatable record from an older one.

        ALSO classifies and counts the "undatable" case here at the call
        site: a record whose timestamp is missing, empty, or unparseable. An
        undatable record is excluded from the result — same as a genuinely
        older record — but is counted in ``self._undatable_freshness_records``
        and logged with its id and raw value via a single structured
        ``logger.warning``, rather than silently dropped, per design
        invariant INV-2 ``structured-facts-at-failure``. Exclusion (not
        fail-open inclusion) matters specifically here because a record
        whose timestamp can never be parsed would otherwise re-surface as
        "new" on every single cycle forever — the 894fbe90 incident's
        symptom, reached by a different cause (a string-typed comparison
        degenerating to date-granularity, vs. an outright unparseable value
        here).

        *source* and *id_key* are caller-supplied purely to shape the log
        record (``'episodes'``/``'uuid'`` or ``'mem0'``/``'id'``); *get_raw_ts*
        lets the caller apply the mem0-only ``created_at`` -> ``updated_at``
        fallback without duplicating this method per record shape.
        """
        wm_utc = to_utc(watermark)
        new_records = []
        for record in records:
            raw_ts = get_raw_ts(record)
            ts = parse_instant(raw_ts)
            if ts is None:
                self._undatable_freshness_records += 1
                logger.warning(
                    'reconciliation.stage1_undatable_freshness_record',
                    extra={
                        'project_id': self.project_id,
                        'source': source,
                        'record_id': record.get(id_key),
                        'raw_timestamp': raw_ts,
                    },
                )
                continue
            if ts > wm_utc:
                new_records.append(record)
        return new_records

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        # Token-budget assembled payload — skip generic fetch
        if self.assembled_payload is not None:
            return await self._format_assembled_payload(watermark)

        # Validate that limits were explicitly set by the harness
        if self.episode_limit is None or self.memory_limit is None:
            raise ValueError(
                f'episode_limit and memory_limit must be explicitly set by the harness before run(); '
                f'got episode_limit={self.episode_limit}, memory_limit={self.memory_limit}'
            )

        # Reset run-local degraded list before fetches (must precede every early-return so
        # reused instances never leak a stale list into a subsequent remediation run's stats)
        self._fetch_degraded_sources = []
        # Reset run-local undatable-record counter (task 4574) — same reused-instance
        # leak hazard as _fetch_degraded_sources above, so it gets the identical
        # precede-every-early-return placement.
        self._undatable_freshness_records = 0
        self._unfinished_graphiti_write_start = None

        # Remediation mode: return focused payload with findings only
        if self.remediation_findings is not None:
            return await self._assemble_remediation_payload()

        self._unfinished_graphiti_write_start = (
            self.memory.earliest_unfinished_graphiti_write_start(self.project_id)
        )
        # 1. Episodes since last reconciliation
        try:
            episodes = await self.memory.get_episodes(
                project_id=self.project_id, last_n=self.episode_limit
            )
        except Exception as e:
            logger.warning(
                'reconciliation.stage1_episodes_fetch_failed',
                extra={'project_id': self.project_id, 'error': str(e)},
            )
            episodes = []
            self._fetch_degraded_sources.append('episodes')
        new_episodes = episodes
        if watermark.last_episode_timestamp:
            # WHY an instant comparison and not `str(watermark...) > str(...)`:
            # str(datetime) renders with a space separator
            # ('2026-08-20 12:00:00+00:00'), while e['created_at'] here comes
            # from services/memory_service.py::_created_at_to_utc_iso as
            # ISO-8601 with a 'T' separator ('2026-08-20T01:52:27+00:00'). A
            # lexical `>` between the two short-circuits at index 10 ('T'
            # 0x54 > ' ' 0x20) and degenerates to date-granularity, so any
            # episode from the watermark's own calendar day compares as
            # "newer" regardless of its actual time (task 4574). Task 2055
            # fixed retrieve_episodes' ordering and its follow-up 2079 added
            # _created_at_to_utc_iso to normalize the producer side, but left
            # this consumer comparing strings — do not reintroduce that.
            new_episodes = self._filter_records_newer_than_watermark(
                episodes,
                watermark.last_episode_timestamp,
                source='episodes',
                id_key='uuid',
                get_raw_ts=lambda e: e.get('created_at'),
            )

        # 2. Mem0 memories (recent)
        from fused_memory.models.scope import Scope
        scope = Scope(project_id=self.project_id)
        try:
            all_memories = await self.memory.mem0.get_all(scope, limit=self.memory_limit)
            results = all_memories.get('results') if isinstance(all_memories, dict) else None
            if not isinstance(results, list):
                logger.warning(
                    "mem0.get_all returned malformed/absent 'results' key; "
                    "treating as empty. got: %s",
                    repr(all_memories)[:200],
                )
                results = []
            mem0_memories = results
        except Exception:
            logger.warning(
                "mem0.get_all raised an exception; treating as empty memories.",
                exc_info=True,
            )
            mem0_memories = []
            self._fetch_degraded_sources.append('mem0')

        new_memories = mem0_memories
        if watermark.last_memory_timestamp:
            # Same str(watermark) lexical-compare defect as the episode filter
            # above (task 4574) — Mem0 created_at/updated_at values are also
            # ISO-with-'T' and, unlike episodes, are not normalized by
            # _created_at_to_utc_iso, so they can carry non-UTC offsets too.
            # Keep the `or` (not `if/else`) so an empty-string created_at
            # still falls through to updated_at, matching prior behavior.
            new_memories = self._filter_records_newer_than_watermark(
                mem0_memories,
                watermark.last_memory_timestamp,
                source='mem0',
                id_key='id',
                get_raw_ts=lambda m: m.get('created_at') or m.get('updated_at'),
            )

        # 3. Store stats
        status = await self._fetch_status()

        # 4. Events summary
        event_summary = format_events(events)

        # 5. Prior S3 findings (backstop from last completed run)
        prior_s3_section = ''
        if self.prior_s3_findings:
            prior_s3_section = (
                f'\n### Prior Stage 3 Findings ({len(self.prior_s3_findings)})\n'
                f'These issues were found in the last integrity check and should be addressed '
                f'during this consolidation pass if possible.\n'
                f'{format_findings(self.prior_s3_findings)}\n'
            )

        # 6. Cycle fence section
        cycle_fence_section = ''
        if self.cycle_fence_time:
            cycle_fence_section = (
                f'\n### Cycle Fence\n'
                f'This cycle started at {self.cycle_fence_time.isoformat()}.\n'
                f'Do NOT delete, merge, or modify any memory whose metadata includes '
                f'`source=targeted_reconciliation` and was created after this timestamp. '
                f'These are recent targeted reconciliation writes that must be preserved.\n'
            )

        # 7. Active task tree (task 455)
        task_tree_section = self._build_task_tree_section()

        # 7b. Task Count Census (task 1785)
        task_count_census_section = self._build_task_count_census_section()

        # 7c. Overreach shadow flags (design.md §6.1 O4, task 6137)
        overreach_flags_section = render_overreach_flags_section(self.overreach_flags)

        # 8. Format
        episodes_str, ep_n = format_episodes(new_episodes)
        memories_str, mem_n = format_memories(new_memories)
        self._entity_summary_snapshot_lines_stripped = ep_n + mem_n
        # Episodes arrive newest-first (undatable last), so one the watermark filter
        # dropped proves nothing newer lay past the cap. mem0's get_all scrolls in
        # point-id order, which proves nothing, so its note rests on the count alone.
        if len(new_episodes) == len(episodes):
            episodes_str = append_note(episodes_str, fetch_cap_note(
                fetched=len(episodes), noun='episode(s)', cap_name='episode_limit', cap=self.episode_limit,
            ))
        memories_str = append_note(memories_str, fetch_cap_note(
            fetched=len(mem0_memories), noun='memory record(s)', cap_name='memory_limit',
            cap=self.memory_limit,
        ))

        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(events)})
{event_summary}

### New Episodes Since Last Reconciliation ({len(new_episodes)})
{episodes_str}

### New Mem0 Memories Since Last Reconciliation ({len(new_memories)})
{memories_str}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{format_watermark(watermark, include_freshness_cutoffs=True)}
{prior_s3_section}{cycle_fence_section}{task_tree_section}{task_count_census_section}{overreach_flags_section}{await self._render_required_sections()}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}"""

    async def _fetch_status(self) -> dict:
        """Fetch store status, logging a WARNING and tracking degradation on failure.

        Single source of truth for the status-fetch try/except that both
        assemble_payload (legacy time-windowed path) and _format_assembled_payload
        (assembled/token-budget path) share.  The caller is responsible for resetting
        self._fetch_degraded_sources before calling this method.
        """
        try:
            return await self.memory.get_status(project_id=self.project_id)
        except Exception as e:
            logger.warning(
                'reconciliation.stage1_status_fetch_failed',
                extra={'project_id': self.project_id, 'error': str(e)},
            )
            self._fetch_degraded_sources.append('status')
            return {}

    async def _format_assembled_payload(self, watermark: Watermark) -> str:
        """Format a payload from ContextAssembler output — event-driven context."""
        ap = self.assembled_payload
        assert ap is not None

        # assemble_payload's line-262 reset is bypassed by the early return at lines 250-251.
        # Reset here so each assembled-path call starts clean (no leak from a prior run).
        self._fetch_degraded_sources = []
        if ap.context_fetch_failures:
            self._fetch_degraded_sources.append('event_context')
        # This path does no freshness filtering (it formats ContextAssembler output
        # directly, not the episodes/mem0-memories filters below), so the count stays
        # 0 here — but reset it anyway so a reused instance never inherits a stale
        # value from a prior time-windowed assemble_payload call (task 4574, same
        # leak hazard as _fetch_degraded_sources above).
        self._undatable_freshness_records = 0
        self._unfinished_graphiti_write_start = None

        event_summary = format_events(ap.events)
        deferred_events_note = clip_note(
            omitted=ap.events_remaining,
            omitted_noun='buffered event(s)',
            cap_name='reconciliation.token_budget',
            cap=ap.token_budget,
            dropped_first='the newest events (still buffered for a later chunk)',
        )
        event_summary = append_note(event_summary, deferred_events_note)

        # Store status (cheap fetch, always useful)
        status = await self._fetch_status()

        # Prior S3 findings
        prior_s3_section = ''
        if self.prior_s3_findings:
            prior_s3_section = (
                f'\n### Prior Stage 3 Findings ({len(self.prior_s3_findings)})\n'
                f'These issues were found in the last integrity check and should be addressed '
                f'during this consolidation pass if possible.\n'
                f'{format_findings(self.prior_s3_findings)}\n'
            )

        # Cycle fence
        cycle_fence_section = ''
        if self.cycle_fence_time:
            cycle_fence_section = (
                f'\n### Cycle Fence\n'
                f'This cycle started at {self.cycle_fence_time.isoformat()}.\n'
                f'Do NOT delete, merge, or modify any memory whose metadata includes '
                f'`source=targeted_reconciliation` and was created after this timestamp. '
                f'These are recent targeted reconciliation writes that must be preserved.\n'
            )

        # Active task tree (task 455)
        task_tree_section = self._build_task_tree_section()

        # Task Count Census (task 1785)
        task_count_census_section = self._build_task_count_census_section()

        # Overreach shadow flags (design.md §6.1 O4, task 6137)
        overreach_flags_section = render_overreach_flags_section(self.overreach_flags)

        ctx_str, ctx_n = format_context_items(ap.context_items)
        self._entity_summary_snapshot_lines_stripped = ctx_n
        ctx_str = append_note(ctx_str, format_context_fetch_failures(ap.context_fetch_failures))

        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(ap.events)})
{event_summary}

### Related Context ({len(ap.context_items)} items)
{ctx_str}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{format_watermark(watermark)}
{prior_s3_section}{cycle_fence_section}{task_tree_section}{task_count_census_section}{overreach_flags_section}{await self._render_required_sections()}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}"""

    async def _render_required_sections(self) -> str:
        """Render every inference-bearing payload section, in registry order.

        Every Stage-1 payload builder MUST interpolate this — enforced
        structurally by
        ``tests/reconciliation/test_stage1_payload_section_parity.py``, which
        discovers the builders by AST introspection so a fourth builder is in
        scope the day it is added. See :attr:`REQUIRED_SECTIONS` for which
        sections qualify and why three others deliberately do not.

        Each registered renderer keeps its own conditional-empty contract, so
        ``''`` is a normal result and keeps the payload tight — that is why this
        adds no separator of its own. Note the sections DO render on remediation
        passes: the harness sets ``filtered_task_tree`` there too
        (``ReconciliationHarness._configure_consolidator``), a measured fact from
        task 3839 — the remediation call site is not a no-op.

        Adding a section is a single :attr:`REQUIRED_SECTIONS` edit rather than
        one edit per builder; that is the whole point of routing through here.

        Every registered renderer is a coroutine method (task 3778: the
        live-workflow renderer's git probes are async), awaited in registry
        order.
        """
        return ''.join(
            [await getattr(self, section.renderer)() for section in self.REQUIRED_SECTIONS]
        )

    def _build_project_root_directive(self) -> str:
        """Return the project_root directive line for payload footers.

        A stage is only constructible with a validated ``ProjectScope``, so
        ``self.project_root`` is always a non-empty absolute path — this always
        returns ``'\\nUse project_root=...'`` (leading newline keeps it on its own
        line after the preceding guideline).  Both payload methods call this
        helper so the f-string fragment stays in a single place.
        """
        return f'\nUse project_root="{self.project_root}" for tasks scoped to this project.'

    def _build_task_tree_section(self) -> str:
        """Return the Active Task Tree prompt section, or empty string if no tree set.

        Eliminates the duplicate block across assemble_payload and
        _format_assembled_payload. (task 455)
        """
        if self.filtered_task_tree is None:
            return ''
        return '\n' + format_filtered_task_tree(self.filtered_task_tree) + '\n'

    async def _build_live_workflow_section(self) -> str:
        """Return the Live-Workflow Signals prompt section, or empty string if inapplicable.

        Renders over Stage 2's source, ``self.filtered_task_tree.active_tasks``
        (task 1655). A stage is only constructible with a validated
        ``ProjectScope``, so ``self.project_root`` is always a non-empty
        absolute path and is not part of this guard (task 2150).
        Reuses Stage 2's snapshot builder,
        ``reconciliation/live_workflow_section.py::build_live_workflow_snapshot``,
        so both stages emit byte-identical section formatting for the same
        underlying live-workflow signals (task 1977).
        This reuse means detector-layer behavior changes apply here
        automatically with no code change — including the blocked-normal
        bare-orchestrator-signal suppression added in task 2409.

        Returns '' when no task tree is set or no active task is listed — keeps
        the payload tight, matching _build_task_tree_section's pattern.  The
        snapshot is retained for :meth:`run`'s citation guard.  A tree with no
        active task yields a real, empty snapshot, because an absent section
        tells this stage that no task is live; only an unset tree leaves it
        None, and the guard inert.  Claimant heartbeats are aged at
        ``filtered_task_tree_fetched_at``, the instant the tree was read, not
        at render time (task 6027).
        """
        if self.filtered_task_tree is None:
            return ''
        claims_read_at = resolve_tree_read_instant(
            self.filtered_task_tree_fetched_at,
            event='reconciliation.stage1_task_tree_read_instant_missing',
            extra={'project_id': self.project_id},
        )
        self._live_workflow_snapshot = await build_live_workflow_snapshot(
            self.filtered_task_tree.active_tasks,
            self.scope.project_root,
            claims_read_at=claims_read_at,
        )
        section = self._live_workflow_snapshot.render()
        return '\n' + section if section else ''

    def _build_task_count_census_section(self) -> str:
        """Return the Task Count Census payload section, or empty string if unavailable.

        Mirrors _build_task_tree_section — inserted into both assemble_payload and
        _format_assembled_payload alongside task_tree_section.  The authoritative
        counts come from get_statuses (a deterministic compact read) rather than the
        async-queued Graphiti snapshot edge that triggered the run-929b4135 incident.

        Returns '' when task_count_verification is None so the payload is unchanged
        for remediation passes and for cycles where get_statuses was unavailable.
        """
        if self.task_count_verification is None:
            return ''
        v = self.task_count_verification
        if not v.get('available'):
            return ''
        auth = v.get('authoritative') or {}
        auth_total = auth.get('total', 'unknown')
        auth_done = auth.get('done', 'unknown')
        divergence_note = ''
        if not v.get('consistent', True):
            tree = v.get('tree') or {}
            divergence_note = (
                f'\n⚠ Divergence detected: tree reports done={tree.get("done", "?")} '
                f'vs authoritative done={auth_done}.'
            )
        return (
            f'\n### Task Count Census (authoritative — from get_statuses)\n'
            f'Total: {auth_total}, Done: {auth_done}{divergence_note}\n'
        )

    async def _assemble_remediation_payload(self) -> str:
        """Focused payload for remediation runs — findings only, no full data."""
        self._entity_summary_snapshot_lines_stripped = 0
        findings = self.remediation_findings or []
        return f"""## Remediation Run — Stage 1: Targeted Memory Fixes
## Project: {self.project_id}

### Actionable Findings to Remediate ({len(findings)})
{format_findings(findings)}{await self._render_required_sections()}

## Your Task
This is a focused remediation run. Address ONLY the specific findings listed above:
1. For each finding: investigate the affected IDs, apply the suggested action, verify the fix.
2. If a finding cannot be resolved, flag it for Stage 2 with an explanation.
3. Do NOT perform general consolidation — only fix the listed findings.
4. Report each finding's resolution status in your structured report.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}
"""


async def _await_best_effort(
    make: Callable[[], Awaitable[_T]],
    *,
    event: str,
    project_id: str,
    run_id: str,
    **extra: Any,
) -> _T | None:
    """Await ``make()``; on an Exception, log *event* with its extras and return None.

    *make* is called inside the try, so building the call's arguments is
    guarded too. Cancellation is not an Exception, so it propagates.
    """
    try:
        return await make()
    except Exception:
        logger.exception(event, extra={'project_id': project_id, 'run_id': run_id, **extra})
        return None


def _zeroed(stat_map: _StatMap) -> dict[str, int]:
    return dict.fromkeys((stat_key for stat_key, _ in stat_map), 0)


def _publish_sweep_stats(
    report: StageReport, stat_map: _StatMap, sweep_result: Mapping[str, Any] | None,
) -> None:
    """Copy *sweep_result*'s counts into the report; a sweep that failed (None) publishes none."""
    if sweep_result is None:
        return
    report.stats.update({stat_key: sweep_result[result_key] for stat_key, result_key in stat_map})
