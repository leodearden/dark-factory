"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance initialisation: avoids shared-mutable-default hazard.
        # A class-level [] would be mutated in-place by any append() that ran before
        # the run-local reset in assemble_payload / _format_assembled_payload, leaking
        # state across all instances (and across reused-instance runs).
        self._fetch_degraded_sources: list = []
        self._undatable_freshness_records: int = 0
        self._live_workflow_snapshot: LiveWorkflowSnapshot | None = None

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
        resume_session_id: str | None = None,
    ) -> StageReport:
        """Execute Stage 1 and post-process items_flagged through the flag deduplicator.

        Remediation runs (``remediation_findings`` set) skip dedup — the whole
        point of a remediation pass is to re-emit a curated list, and running
        dedup on those flags would defeat the remediation contract.
        """
        self._live_workflow_snapshot = None
        report = await super().run(
            events, watermark, prior_reports, run_id, model=model,
            resume_session_id=resume_session_id,
        )

        # Phantom-citation verification (task 2978) used to run here. Task 2979
        # HOISTED it into BaseStage.run(), which performs it on the shared
        # items_flagged assembly for all three stages — see the rationale
        # comment there. The stage1_* citation stats therefore arrive on
        # report.stats via super().run() above, already present on full AND
        # remediation passes. Do not re-add a call here: it would double-count
        # every stage1_* citation stat and double the get_memory_by_id load.

        report.stats['entity_summary_snapshot_lines_stripped'] = (
            self._entity_summary_snapshot_lines_stripped
        )
        report.stats['stage1_fetch_degraded'] = self._fetch_degraded_sources or []
        report.stats['stage1_assembled_budget'] = (
            self.assembled_payload.budget_report() if self.assembled_payload is not None else None
        )

        # Always present (task 4574, mirrors stage1_fetch_degraded above): set BEFORE
        # the remediation early-return so the key is unconditionally present, including
        # on remediation passes (which never call assemble_payload's freshness filters
        # and so leave the per-instance counter at its __init__/reset value of 0).
        report.stats['stage1_undatable_freshness_records'] = self._undatable_freshness_records

        # Always present (task-2312): set BEFORE the remediation early-return below
        # so downstream consumers see this key on every run, including remediation
        # passes (which skip dedup_flags entirely and so never reach the overwrite
        # below). Overwritten after dedup_flags on a full (non-remediation) run with
        # items_flagged; stays 0 otherwise.
        report.stats['stage1_completion_markers_self_deleted'] = 0

        # Always present (task 2229 amendment, mirrors the task-2312 pattern
        # above): set BEFORE the remediation early-return so this key is
        # never conditionally absent. Stage 2's equivalent
        # stage2_cycle_summary_ledger_written (task_knowledge_sync.py) is set
        # unconditionally, so a consumer aggregating both stages' stats
        # should not need a .get(..., 0) fallback for Stage 1's key.
        # Overwritten below on a full (non-remediation) cycle once
        # write_cycle_summary actually runs; stays 0 on remediation passes,
        # which intentionally skip the write (see that call site below).
        #
        # Named "..._ledger_written", not "..._written" (reviewer finding
        # observability, task 2229 amendment pass round 2): this reflects
        # ONLY the authoritative ReconLedgerStore upsert. write_cycle_summary
        # also attempts a best-effort Mem0 mirror write regardless of ledger
        # outcome (see its docstring), so a deployment running with
        # recon_ledger_enabled=False can have this stat at 0 while the
        # mirror was in fact written — the "_ledger_" qualifier makes that
        # distinction explicit instead of implying "no summary at all".
        report.stats['stage1_cycle_summary_ledger_written'] = 0

        # Always present (task 3084, mirroring the two pre-inits above): set
        # BEFORE the remediation early-return so no key is ever conditionally
        # absent — Stage 1's whole report.stats blob is serialized verbatim
        # into Stage 2's prompt by _format_report (task_knowledge_sync.py), so
        # a consumer should not need a .get(..., 0) fallback.  Overwritten
        # below on a full (non-remediation) cycle once the sweep actually runs;
        # stays 0 on remediation passes (which deliberately skip the sweep, see
        # that block below) and when filtered_task_tree is unset.
        #
        # _errors is reported alongside the other two (reviewer finding
        # "observability", amendment pass) because without it a cycle in which
        # EVERY gate failed its Qdrant read is byte-identical, in the report and
        # therefore in Stage 2's prompt, to a cycle in which every gate was
        # cleanly checked and none was resolved — both read scanned=N,
        # flags_emitted=0.  The failure would exist only in the process log,
        # which is exactly the silent-degradation shape the no-silent-fail-soft
        # invariant targets.  With all three present, the reader can also spot
        # the sweep's zero-recall signature (scanned > 0, flags_emitted == 0,
        # errors == 0 — see the sweep module docstring on source-key drift).
        report.stats['curator_gate_resolution_scanned'] = 0
        report.stats['curator_gate_resolution_flags_emitted'] = 0
        report.stats['curator_gate_resolution_errors'] = 0
        # Same always-present contract for the orphaned-recon-escalation sweep
        # (task 3052).  Zeroed HERE, above the remediation early-return, so a
        # remediation pass — which never runs the sweep — still publishes the
        # keys.  With all eight present a reader can distinguish a clean cycle
        # that found nothing (scanned > 0, flags_emitted == 0, errors == 0)
        # from a degraded one (errors > 0), from a registry gap
        # (unresolvable > 0) and from a cross-tag id collision
        # (ambiguous > 0), without a .get(..., 0) fallback.  The classification
        # counts are fleet-wide while flags_emitted is this project's only, so
        # terminal + missing > flags_emitted is normal.
        report.stats['orphaned_recon_escalations_scanned'] = 0
        report.stats['orphaned_recon_escalations_terminal'] = 0
        report.stats['orphaned_recon_escalations_missing'] = 0
        report.stats['orphaned_recon_escalations_live'] = 0
        report.stats['orphaned_recon_escalations_ambiguous'] = 0
        report.stats['orphaned_recon_escalations_unresolvable'] = 0
        report.stats['orphaned_recon_escalations_errors'] = 0
        report.stats['orphaned_recon_escalations_flags_emitted'] = 0
        # Same always-present contract for the unlanded-path check (task 5626).
        report.stats.update(UnlandedPathCheckResult.empty().stats())

        # Always present (task 2896 γ): count of recon flags suppressed this cycle
        # by an ACTIVE entity_standing_decision (Hook A).  Overwritten below to the
        # actual sum on a full cycle with items_flagged; stays 0 when nothing was
        # flagged and on a remediation pass, which returns just below.  Pre-inited
        # HERE, above that early return, rather than beside the filter it belongs to
        # (reviewer finding correctness, amendment pass): a key set below the return
        # is simply ABSENT from a remediation report, which is the .get(..., 0)
        # fallback the always-present convention exists to spare consumers — and
        # Stage 1's whole stats blob is serialized verbatim into Stage 2's prompt.
        # The longest cross-cycle suppression streak (task 2943) follows the same
        # convention: overwritten by _account_suppression_streaks on every full
        # cycle, 0 on a remediation pass.
        report.stats['entity_standing_decision_suppressed'] = 0
        report.stats['entity_standing_decision_max_suppression_streak'] = 0

        # ── Preservation-specimen corroboration guard (task 4223) ──────────────
        # Decline stranded/reset recommendations for tasks whose odd state is
        # DOCUMENTED as deliberate — a preserved validation specimen.  Task 3105
        # is in-progress with a null claimant and a null heartbeat on purpose,
        # and this finding twice became an operator-gate task asking for it to be
        # reset (tasks 5080 and 5104, the second born-at-L2 critical); both were
        # declined by hand.
        #
        # Placement is the whole point, and it is NOT beside Hook A below.  The
        # remediation early-return three lines down sits ABOVE the entire filter
        # chain, so anything wired next to filter_entity_standing_decisions is
        # unreachable on a remediation pass — and a BLANKET stage1_flag_suppression
        # for task 3105 failed to stop the recurrence for exactly that reason: the
        # recurrences that continued after it was widened were remediation runs.
        # This is the placement verify_cited_memories already uses to cover both
        # passes, for the same reason.
        #
        # Both stats are pre-inited here, above the return, so neither key is ever
        # conditionally absent from a remediation report (the always-present
        # convention the three pre-inits above follow).
        #
        # Best-effort: a guard failure must never abort the stage or leave
        # items_flagged partially mutated, so items_flagged is reassigned only on
        # success and the stats stay at their pre-inited 0.
        report.stats['preservation_specimen_suppressed'] = 0
        report.stats['preservation_specimen_unresolved'] = 0
        report.stats['preservation_specimen_citations'] = {}
        preservation_result = None
        try:
            preservation_result = await filter_preservation_specimen_flags(
                memory_service=self.memory,
                project_id=self.project_id,
                flags=report.items_flagged,
            )
            report.items_flagged = preservation_result.kept_flags
            report.stats['preservation_specimen_suppressed'] = sum(
                preservation_result.suppressed_by_task.values()
            )
            # INV-11: a cycle whose corroboration reads FAILED is not a clean
            # cycle.  Surfacing the count here keeps the degradation visible in
            # the blob Stage 2's prompt serializes, not only in the process log.
            report.stats['preservation_specimen_unresolved'] = len(
                preservation_result.unresolved_task_ids
            )
            # The evidence travels with the number.  A count alone is the
            # anonymous drop citations_by_task exists to prevent: the storm
            # escape only names a citation above its threshold, so in normal
            # operation this map is the ONLY place an operator (or Stage 2,
            # which is handed this blob verbatim) can see WHICH record
            # authorised the suppression and judge whether it is still current.
            report.stats['preservation_specimen_citations'] = dict(
                preservation_result.citations_by_task
            )
            # Suppression is NOT resolution — but unlike Hook A below, that needs
            # no signature bookkeeping here: this guard runs ABOVE the
            # acknowledgment snapshot (_pre_filter_flags), so a flag it drops is
            # already gone before acknowledgment candidates are computed and can
            # never be reclaimed as resolved.  Enforced by position, not by
            # exclusion; the snapshot comment below names the same dependency.
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                'preservation-specimen guard failed for project %s (best-effort — '
                'passing %d flag(s) through unfiltered this cycle)',
                self.project_id, len(report.items_flagged or []),
            )

        # Live-workflow citation guard (task 4874): annotate, never drop, a finding
        # citing a signal THIS run's section did not render (rationale: the module
        # docstring of live_workflow_citation_guard).  Above the remediation
        # early-return, like the guard above, so both passes are covered.
        report.stats[CITATION_CONTRADICTIONS_STAT] = 0
        try:
            report.items_flagged, report.stats[CITATION_CONTRADICTIONS_STAT] = (
                check_live_workflow_citations(report.items_flagged or [], self._live_workflow_snapshot)
            )
        except Exception:
            logger.exception('live-workflow citation guard failed for project %s', self.project_id)

        # Always present (task 4814, same convention as the three pre-inits
        # above — tasks 2312 / 2229 / 3084): set BEFORE the remediation
        # early-return so this key is never conditionally absent.  Stage 1's
        # whole report.stats blob is serialized verbatim into Stage 2's prompt
        # by _format_report (task_knowledge_sync.py), so a consumer should not
        # need a .get(..., 0) fallback.  Overwritten at the END of the
        # ``if report.items_flagged:`` block below on a full cycle; stays 0 on
        # remediation passes, when filtered_task_tree is unset, and when no
        # active task is human-gate-owned.
        report.stats['gate_owned_suggested_actions_normalized'] = 0

        # Always present (task 5240), the same convention as the pre-inits above;
        # the key set's one home is auto_consolidation_initial_stats.
        report.stats.update(auto_consolidation_initial_stats())

        # Skip dedup for remediation passes
        if self.remediation_findings is not None:
            return report

        # Per-cycle "storm escape" for the guard above (INV-4), and its
        # unreadable-corroboration disclosure.  Full-cycle only, mirroring the
        # Hook A storm call below; skipped when no escalation queue is wired
        # (nothing would consume the escalation).
        #
        # Deliberately NOT inside the ``if report.items_flagged:`` block below:
        # a storm is precisely the cycle in which the guard may have suppressed
        # EVERY flag, which would leave that block unentered and the storm
        # unreported — the one cycle it most needs filing.
        if self._escalation_queue is not None and preservation_result is not None:
            await maybe_escalate_preservation_suppression_storm(
                escalation_queue=self._escalation_queue,
                project_id=self.project_id,
                run_id=run_id,
                result=preservation_result,
            )

        # ── Resolved human-curator-gate sweep (task 3084) ──────────────────────
        # Flag open ``operational_mode == 'gate'`` tasks for which the reify
        # curator has ALREADY written its ruling to Mem0 (an entry stamped
        # ``metadata.source == 'curator_gate_{task_id}'``).  Detection was an
        # ad-hoc Stage-3 spot-check that missed ~25% of cases (run ec45eed0:
        # gates 5561 and 5563 were resolved-but-stale and went undetected).
        # Stage 1 runs under DISALLOW_TASK_WRITES, so this only FLAGS; Stage 2
        # (which holds set_task_status) acts.
        #
        # Placement, deliberately unlike the three other Stage-1 sweeps below
        # (degenerate_task_node / stale_status_snapshot / stale_priority_override,
        # which all run well after the filter chain): those only mutate Graphiti
        # and return int stats — none of them touch the flag channel.  This one
        # EMITS flags, so it must sit ABOVE dedup_flags, for two reasons.
        # (1) Each appended flag then gets a stage1_flag_marker ledger row keyed
        #     on (task_id, flag_type), giving cross-cycle recurrence tracking and
        #     honoring explicit suppression records.  Appending below dedup_flags
        #     would bypass dedup entirely, so an un-actioned gate flag would
        #     re-emit unmarked every cycle with no recurrence history and no way
        #     for an operator to suppress it.  (dedup_flags never DROPS on a hit —
        #     only filter_suppressed drops — so re-emission until Stage 2 closes
        #     the gate is preserved, which is the desired behaviour.)
        # (2) It lets the ``if report.items_flagged:`` guard below fire on a cycle
        #     where the LLM emitted zero flags of its own but the sweep found a
        #     resolved gate.
        #
        # That placement DOES step over verify_cited_memories (above, right after
        # super().run()), so the cited_memories these flags carry are the one
        # citation set that citation_verifier.py's end-to-end "a cited memory id
        # must resolve" invariant never sees, and stage1_citations_verified /
        # stage1_phantom_citations_dropped deliberately exclude them (reviewer
        # finding "architecture", amendment pass).  That exemption is sound
        # BECAUSE of where these ids come from: the verifier exists to catch
        # LLM-authored ids that were never real and ids whose queued add_memory
        # write later failed, whereas these ids are read straight off a Qdrant
        # scroll microseconds earlier in the same cycle — they resolve by
        # construction, and sweep_resolved_curator_gates additionally refuses to
        # emit a flag at all unless that scroll returned at least one row (its
        # count/scroll-divergence guard), so an uncitable gate flag is impossible
        # by a different route.  Moving the sweep above the verifier is NOT the
        # cheap fix it looks like: the verifier sits above the remediation
        # early-return so that both full and remediation passes are covered, and
        # this sweep must run on full cycles ONLY, so the move would require
        # duplicating the remediation guard here.
        #
        # Best-effort: a whole-sweep failure must never abort the stage or leave
        # items_flagged partially mutated — it is logged and swallowed, and all
        # three stats stay at their pre-early-return 0 for this cycle.
        if self.filtered_task_tree is not None:
            gate_ids = extract_open_gate_task_ids(self.filtered_task_tree.active_tasks)
            # Title-enrichment map (reviewer finding "dead-code", amendment
            # pass).  extract_open_gate_task_ids deliberately returns bare ids,
            # so without this the sweep can only name a gate by number and
            # build_gate_resolution_flag's title branch would be unreachable in
            # production.  Keyed on the same str(id) coercion the selector uses
            # and restricted to the swept ids; it is load-bearing for the
            # description ONLY — selection remains the selector's job alone, so
            # a partial map can never change which gates are swept or flagged.
            _gate_id_set = set(gate_ids)
            gate_tasks_by_id = {
                str(task.get('id')): task
                for task in self.filtered_task_tree.active_tasks
                if isinstance(task, dict) and str(task.get('id')) in _gate_id_set
            }
            try:
                gate_sweep = await sweep_resolved_curator_gates(
                    self.memory, self.project_id, gate_ids,
                    tasks_by_id=gate_tasks_by_id,
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logger.exception(
                    'reconciliation.curator_gate_resolution_sweep_failed',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'gate_id_count': len(gate_ids),
                    },
                )
            else:
                # Stamped here because this call site is the only place that
                # KNOWS these flags came from the sweep; the task-4814 phrasing
                # carve-out below needs that provenance rather than the
                # free-form flag_type string an LLM finding may also pick.
                report.items_flagged = (report.items_flagged or []) + (
                    stamp_curator_gate_sweep_provenance(gate_sweep['flags'])
                )
                report.stats['curator_gate_resolution_scanned'] = gate_sweep['scanned']
                report.stats['curator_gate_resolution_flags_emitted'] = len(
                    gate_sweep['flags'],
                )
                report.stats['curator_gate_resolution_errors'] = gate_sweep['errors']

        # ── Auto-consolidation executor (task 5240) ────────────────────────────
        # Contract: plans/memory-auto-consolidation-prd.md C4.
        try:
            await run_auto_consolidation(
                self.memory,
                self.taskmaster,
                self._escalation_queue,
                project_id=self.project_id,
                project_root=self.project_root,
                run_id=run_id,
                config=self.memory.config.consolidation_auto,
                active_tasks=(
                    self.filtered_task_tree.active_tasks
                    if self.filtered_task_tree is not None
                    else None
                ),
                report=report,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                'reconciliation.auto_consolidation_failed',
                extra={'project_id': self.project_id, 'run_id': run_id},
            )

        # ── Orphaned recon-escalation sweep (task 3052) ────────────────────────
        # Flag pending L1 ``reconciliation_stale_*`` records whose subject task
        # has gone terminal (done/cancelled) or vanished from its own project's
        # task store.  Those records are filed only while the subject is
        # ``status == 'blocked'``
        # (stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids), so
        # they can never re-file, yet nothing closes them: the harness never
        # resolves its own escalation queue (A7b), and the orchestrator's
        # revalidation sweep reads a different queue and returns early on
        # ``level != 2`` while every recon record is L1.
        #
        # DETECTION ONLY — the sweep never calls queue.resolve(); the port-8103
        # watcher session is the sole closer, reached via
        # skills/recon-escalation-watcher/SKILL.md.
        #
        # The sweep emits flags only for THIS project's orphans, while its
        # counts cover the whole shared queue; the trade-off is stated at
        # orphaned_recon_escalation_sweep.py::sweep_orphaned_recon_escalations.
        #
        # Placement mirrors the curator-gate block above, for the same two
        # reasons: this sweep EMITS flags, so it must sit ABOVE dedup_flags so
        # (1) each appended flag earns a stage1_flag_marker ledger row keyed on
        # (task_id, flag_type) — cross-cycle recurrence tracking plus honoring
        # explicit suppression, where appending below would re-emit unmarked
        # forever — and (2) the ``if report.items_flagged:`` guard below can fire
        # on a cycle where the LLM emitted no flags of its own.
        #
        # Unlike the curator-gate flags, these carry NO cited_memories at all, so
        # the verify_cited_memories exemption reasoning above does not apply to
        # them in either direction: there is no citation set for the verifier to
        # miss.  Their evidence is the escalation record's own fields plus a
        # status census read microseconds earlier in this same call.
        #
        # Best-effort: a whole-sweep failure must never abort the stage or leave
        # items_flagged partially mutated — it is logged and swallowed, and all
        # seven stats stay at their pre-early-return 0 for this cycle.
        if self._escalation_queue is not None and self.taskmaster is not None:
            try:
                orphan_sweep = await sweep_orphaned_recon_escalations(
                    self._escalation_queue,
                    self.taskmaster,
                    self.known_projects,
                    running_project_id=self.project_id,
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logger.exception(
                    'reconciliation.orphaned_recon_escalation_sweep_failed',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                    },
                )
            else:
                report.items_flagged = (
                    (report.items_flagged or []) + orphan_sweep['flags']
                )
                report.stats['orphaned_recon_escalations_scanned'] = orphan_sweep[
                    'scanned'
                ]
                report.stats['orphaned_recon_escalations_terminal'] = orphan_sweep[
                    'terminal'
                ]
                report.stats['orphaned_recon_escalations_missing'] = orphan_sweep[
                    'missing'
                ]
                report.stats['orphaned_recon_escalations_live'] = orphan_sweep['live']
                report.stats['orphaned_recon_escalations_ambiguous'] = orphan_sweep[
                    'ambiguous'
                ]
                report.stats['orphaned_recon_escalations_unresolvable'] = orphan_sweep[
                    'unresolvable'
                ]
                report.stats['orphaned_recon_escalations_errors'] = orphan_sweep[
                    'errors'
                ]
                report.stats['orphaned_recon_escalations_flags_emitted'] = len(
                    orphan_sweep['flags'],
                )

        # Done-task memories naming paths the branch added then removed before merge (task 5626);
        # emits flags, so it sits above dedup_flags — see reconciliation/unlanded_path_check.py.
        unlanded = await check_unlanded_path_claims(self.memory, self.project_id, self.project_root, events)
        report.items_flagged = (report.items_flagged or []) + list(unlanded.flags)
        report.stats.update(unlanded.stats())

        # Always present (task-2029 amendment), symmetric with Stage 2's
        # stage2_flag_markers_acknowledged; the chain overwrites it on a cycle with flags.
        report.stats['stage1_flag_markers_acknowledged'] = 0
        standing_decision_result = EntityStandingSuppressionResult.empty_batch()
        if report.items_flagged:
            outcome = await run_stage1_flag_chain(
                report.items_flagged, self._flag_chain_context(events, run_id),
            )
            report.items_flagged = outcome.kept_flags
            report.stats.update(outcome.stats)
            standing_decision_result = outcome.standing_decision_result

        await self._account_suppression_streaks(run_id, standing_decision_result, report)

        # ── Census inconsistency detection ────────────────────────────────────
        # Compare task IDs referenced in this cycle's events against the census
        # max from the filtered task tree.  IDs exceeding the max indicate a
        # partial/wrong-source bulk task read (the "highest-ID below known task IDs"
        # signature) and should be surfaced as a structured observation so operators
        # can investigate the root cause.
        #
        # Guard: skip when max_task_id == 0 (empty tree or all IDs unparseable).
        # If every referenced task ID is "above" a zero ceiling the result is a
        # false-positive flood rather than a genuine truncation signal.
        if self.filtered_task_tree is not None and self.filtered_task_tree.max_task_id > 0:
            event_task_ids = [
                e.payload.get('task_id')
                for e in events
                if e.payload.get('task_id') is not None
            ]
            inconsistent = detect_census_inconsistency(
                self.filtered_task_tree.max_task_id, event_task_ids
            )
            if inconsistent:
                report.stats['task_tree_census_inconsistent'] = inconsistent
                logger.warning(
                    'reconciliation.task_tree_census_inconsistent',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'census_max': self.filtered_task_tree.max_task_id,
                        'offending_ids': inconsistent,
                    },
                )

        # ── Task-count cross-verification (task 1785) ─────────────────────────
        # Surface the authoritative get_statuses census vs. get_tasks-derived
        # FilteredTaskTree comparison so the async judge / recon report can act
        # on count divergence without relying on a possibly-dropped Graphiti edge.
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

        # ── Graphiti async-queue health (task 1785) ────────────────────────────
        # Surface the dead-letter count from DurableWriteQueue so the silent
        # add_memory drop (success=True at enqueue, dead-lettered asynchronously)
        # is visible in the Stage 1 report.
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

        # ── FalkorDB index-provisioning health (task 3709 / PRD δ) ────────────
        # Surface the index drift record so a graph serving queries without the
        # indices it should have is visible in the Stage 1 report — ζ's
        # activation verification reads this, so the HEALTHY case is surfaced
        # too, not just the drifted one.
        #
        # Deliberately does NOT file the escalation here, unlike the
        # HOR/gate-backlog path which is stage-only: the Q3 startup sweep has no
        # stage, so the filing lives in the harness detector both paths share.
        # A stage-resident filer would fork δ into two divergent detectors and
        # leave the startup half unable to escalate at all.  This stage is a
        # surfacing point only.
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

        # ── Status-correction reconciliation (task 1938) ───────────────────────
        # Surface the cached project_status_correction memory vs. live get_statuses
        # diff/supersede record so a stale Mem0 correction never silently outlives
        # the authoritative census.
        if self.status_correction_reconciliation is not None:
            report.stats['status_correction_reconciliation'] = (
                self.status_correction_reconciliation
            )

        # ── Stale human-operator-required detector (task 1201) ────────────────
        # Short-circuit when the escalation queue is unavailable — writing Mem0
        # markers that nothing will ever consume only wastes Qdrant capacity.
        # If a queue becomes available in a future cycle the counter starts
        # from zero for that cycle; that is an acceptable trade-off.
        if self._escalation_queue is not None:
            hor_task_ids = extract_human_operator_task_ids(report.items_flagged or [])
            if hor_task_ids:
                stall_counts = await track_human_operator_stalls(
                    memory_service=self.memory,
                    project_id=self.project_id,
                    run_id=run_id,
                    task_ids=hor_task_ids,
                )
                stalled = compute_stalled_task_ids(stall_counts)
                report.stats['stage1_human_operator_stalled'] = len(stalled)
                if stalled:
                    escalated = await maybe_escalate_stalled_tasks(
                        escalation_queue=self._escalation_queue,
                        project_id=self.project_id,
                        run_id=run_id,
                        stalled_task_ids=stalled,
                        stall_counts=stall_counts,
                        flags=report.items_flagged or [],
                    )
                    report.stats['stage1_human_operator_escalated'] = len(escalated)
                else:
                    report.stats['stage1_human_operator_escalated'] = 0

        # ── Stale gate-backlog age check (task 3017) ──────────────────────────
        # Deterministic, stateless age check: a blocked human-decision gate task
        # whose metadata.gate_escalated_at (the born-at-L2 idempotency stamp) has
        # aged past 48h files an independent level-1
        # reconciliation_stale_gate_backlog escalation.  Unlike the HOR path above
        # this needs NO Mem0 marker accumulation and NO LLM finding — it reads the
        # durable stamp straight off the Stage-1 task tree.  Guarded on both the
        # escalation queue AND filtered_task_tree (active_tasks is the input, and
        # restricting to it excludes resolved-and-done gates for free); full-cycle
        # only (below the remediation early-return).
        if self._escalation_queue is not None and self.filtered_task_tree is not None:
            now = datetime.now(UTC)
            gate_stalled = extract_stalled_gate_backlog_task_ids(
                self.filtered_task_tree.active_tasks, now=now,
            )
            report.stats['stage1_gate_backlog_stalled'] = len(gate_stalled)
            if gate_stalled:
                task_by_id = {
                    str(t.get('id')): t
                    for t in self.filtered_task_tree.active_tasks
                    if t.get('id') is not None
                }
                gate_escalated = await maybe_escalate_stalled_gate_backlog(
                    escalation_queue=self._escalation_queue,
                    project_id=self.project_id,
                    run_id=run_id,
                    stalled_task_ids=gate_stalled,
                    task_by_id=task_by_id,
                    now=now,
                )
                report.stats['stage1_gate_backlog_escalated'] = len(gate_escalated)
            else:
                report.stats['stage1_gate_backlog_escalated'] = 0

        # ── Degenerate task-node sweep (task 2107) ─────────────────────────────
        # Delete degenerate ("tasks {id}", edge_count == 0) placeholder Graphiti
        # nodes for terminal (done + cancelled) tasks. extract_terminal_task_ids
        # returns [] for a None filtered_task_tree, so this naturally no-ops when
        # the harness hasn't set one. Best-effort: a sweep failure must never
        # abort the stage or leave a partial/incorrect stat — it is logged and
        # swallowed, and no degenerate_task_nodes_* stat is set for this cycle.
        terminal_ids = extract_terminal_task_ids(self.filtered_task_tree)
        if terminal_ids:
            try:
                sweep_stats = await sweep_degenerate_task_nodes(
                    self.memory, self.project_id, terminal_ids,
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logger.exception(
                    'reconciliation.degenerate_task_node_sweep_failed',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'terminal_id_count': len(terminal_ids),
                    },
                )
            else:
                report.stats['degenerate_task_nodes_swept'] = sweep_stats['deleted']
                report.stats['degenerate_task_nodes_scanned'] = sweep_stats['scanned']

        # ── Stale task-status snapshot edge sweep (tasks 2613, 3037) ──────────
        # Invalidate VALID (invalid_at IS NULL) task-status-snapshot Graphiti
        # edges whose asserted status is now contradicted, via a deterministic
        # direct-lookup sweep (never semantic search). Two selection rules:
        #   - the TERMINAL rule (task 2613): an asserted active/pending/
        #     in-progress status contradicted by a terminal (done/cancelled)
        #     task;
        #   - the BLOCKED-ASSERTION rule (task 3037): an asserted BLOCKED
        #     status contradicted by ANY other positively-known status —
        #     'pending', 'in-progress', 'review', … — not merely a terminal
        #     one. Without it a blocked->pending unblock left 'Task N remains
        #     blocked' asserted as current until the task eventually reached
        #     done, which is most of a task's life.
        # Second half of the blocked rule's deterministic step: after each
        # successful blocked-rule invalidation the sweep writes ONE superseding
        # resulting-state-only temporal_fact per contradicted task per cycle,
        # so the graph records what replaced the retired assertion instead of
        # merely losing it — surfaced here as stale_blocked_edges_superseded.
        # Its two failure modes are surfaced beside it rather than left to the
        # log: a bare superseded=0 cannot distinguish "no blocked edge needed
        # superseding this cycle" from "every superseding write failed" or
        # "the per-cycle write ceiling truncated them" — the same
        # 0-vs-N ambiguity this task was filed against.  (amendment,
        # reviewer_comprehensive observability finding)
        # The enumeration-completeness pair is surfaced for the SAME reason
        # and by the same argument (task 4386): a bare _scanned/_invalidated
        # pair cannot distinguish a clean cycle over the WHOLE corpus from a
        # clean cycle over whatever part of it could be fetched.  When
        # `..._enumeration_complete` is not True, this cycle's counts describe
        # a partial corpus and its absences are not evidence;
        # `..._enumeration_incomplete_kind` names which way it was partial.
        # Two additions, one pattern — 0-vs-N ambiguities resolved in the
        # stats rather than left for a reader to reconstruct from logs.
        # The `_candidates`/`_errors` pair closes the last one (task 4851):
        # `_invalidated` alone cannot tell "nothing was stale" from "every
        # invalidation failed".
        # Best-effort: a sweep failure must never abort the stage or leave a
        # partial/incorrect stat — it is logged and swallowed, and NONE of
        # these stats is set for this cycle.  That includes the enumeration
        # pair: ABSENT is honest for a sweep that raised, where False would
        # claim a corpus was observed and found incomplete.
        try:
            snapshot_sweep_stats = await sweep_stale_status_snapshot_edges(
                self.memory, self.taskmaster, self.project_id, self.project_root,
                run_id=run_id,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                'reconciliation.stale_status_snapshot_edge_sweep_failed',
                extra={
                    'project_id': self.project_id,
                    'run_id': run_id,
                },
            )
        else:
            report.stats['stale_status_snapshot_edges_invalidated'] = (
                snapshot_sweep_stats['invalidated']
            )
            report.stats['stale_status_snapshot_edges_scanned'] = (
                snapshot_sweep_stats['scanned']
            )
            report.stats['stale_status_snapshot_edges_candidates'] = (
                snapshot_sweep_stats['candidate_edges']
            )
            report.stats['stale_status_snapshot_edges_errors'] = (
                snapshot_sweep_stats['errors']
            )
            report.stats['stale_blocked_edges_superseded'] = (
                snapshot_sweep_stats['superseded']
            )
            report.stats['stale_blocked_edges_supersede_errors'] = (
                snapshot_sweep_stats['supersede_errors']
            )
            report.stats['stale_blocked_edges_supersede_skipped'] = (
                snapshot_sweep_stats['supersede_skipped']
            )
            report.stats[STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY] = (
                snapshot_sweep_stats['enumeration_complete']
            )
            report.stats[STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY] = (
                snapshot_sweep_stats['enumeration_incomplete_kind']
            )

        # ── Stale priority-override / pin-queue edge sweep (task 2781) ─────────
        # Invalidate VALID (invalid_at IS NULL) priority-override / pin-queue
        # Graphiti edges whose task is absent from live scheduler-override state
        # (override consumed by dispatch/clear_terminal or expired-and-cleared by
        # clear_expired) or whose TTL edge's live ttl_until has elapsed, via a
        # deterministic direct-lookup sweep beside the task 2613 sweep above.
        # Carries the same enumeration-completeness pair as the task 2613
        # sweep above, under its own prefix and from its own read: the two key
        # sets are INDEPENDENT, so a truncated corpus for one sweep never
        # marks the other.  Same meaning in both places — not True means this
        # cycle's counts describe whatever corpus could be fetched.  (4386)
        # Best-effort: a sweep failure must never abort the stage or leave a
        # partial/incorrect stat — it is logged and swallowed, and no
        # stale_priority_override_edges_* stat is set for this cycle.
        try:
            priority_override_sweep_stats = await sweep_stale_priority_override_edges(
                self.memory, self.project_id, self.project_root,
                run_id=run_id,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                'reconciliation.stale_priority_override_edge_sweep_failed',
                extra={
                    'project_id': self.project_id,
                    'run_id': run_id,
                },
            )
        else:
            report.stats['stale_priority_override_edges_invalidated'] = (
                priority_override_sweep_stats['invalidated']
            )
            report.stats['stale_priority_override_edges_scanned'] = (
                priority_override_sweep_stats['scanned']
            )
            report.stats[PRIORITY_OVERRIDE_ENUMERATION_COMPLETE_STAT_KEY] = (
                priority_override_sweep_stats['enumeration_complete']
            )
            report.stats[PRIORITY_OVERRIDE_ENUMERATION_INCOMPLETE_KIND_STAT_KEY] = (
                priority_override_sweep_stats['enumeration_incomplete_kind']
            )

        # ── Deterministic per-cycle summary write (task 2229 W5-λ) ────────────
        # Python writes the authoritative cycle_summary ledger row directly
        # from `report` — no LLM turn, no nonce, no verify/reconstruct
        # self-heal (all retired by this task; see
        # summary_pool.write_cycle_summary). Mirrors Stage 2's identical
        # wiring in task_knowledge_sync.py. Only reached on full
        # (non-remediation) cycles — the remediation payload
        # (_assemble_remediation_payload) never asked for a cycle_summary
        # under the old LLM-driven mechanism either, so writing one here would
        # fabricate a spurious summary every remediation pass. The stat set
        # below overwrites the `= 0` default assigned before the
        # remediation early-return above.
        ledger_written = await write_stage1_cycle_summary(
            self.memory, self.project_id, report, run_id,
        )
        # Reflects the ledger upsert only — see the "_ledger_written" naming
        # note above the `= 0` default earlier in this method.
        report.stats['stage1_cycle_summary_ledger_written'] = 1 if ledger_written else 0

        return report

    def _flag_chain_context(
        self, events: list[ReconciliationEvent], run_id: str,
    ) -> FlagChainContext:
        return FlagChainContext(
            memory_service=self.memory,
            taskmaster=self.taskmaster,
            escalation_queue=self._escalation_queue,
            project_id=self.project_id,
            project_root=self.project_root,
            known_projects=self.known_projects,
            run_id=run_id,
            events=events,
            active_tasks=(
                self.filtered_task_tree.active_tasks
                if self.filtered_task_tree is not None
                else None
            ),
        )

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
        run() calls it after the ``if report.items_flagged:`` block rather than
        beside Hook A inside it.  A cycle that flagged nothing is one in which
        no decision drained anything, so it must reset every streak; skipping it
        would let a decision suppressing on alternate cycles reach K with gaps
        in between.  On such a cycle *esd_result* is the pre-bound empty batch.

        The state update sits outside the escalation-queue gate so the streak is
        already real when a queue is wired; only the filing needs the queue.
        Both callees are best-effort and never raise.  The updater's per-cycle
        cutoff is the N run() hands the per-cycle escape, so a cycle that
        escape reported is never counted a second time.
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
        ``is_newer_than_watermark`` (task 4574) performs per call, inlined
        here so a record's timestamp is never parsed twice and the watermark
        is never re-normalized per record. ``is_newer_than_watermark`` stays
        the public, independently unit-tested predicate for external callers;
        this method does not call it.

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

        # Remediation mode: return focused payload with findings only
        if self.remediation_findings is not None:
            return await self._assemble_remediation_payload()

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
            # WHY is_newer_than_watermark and not `str(watermark...) > str(...)`:
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
        # This path does no freshness filtering (it formats ContextAssembler output
        # directly, not the episodes/mem0-memories filters below), so the count stays
        # 0 here — but reset it anyway so a reused instance never inherits a stale
        # value from a prior time-windowed assemble_payload call (task 4574, same
        # leak hazard as _fetch_degraded_sources above).
        self._undatable_freshness_records = 0

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
