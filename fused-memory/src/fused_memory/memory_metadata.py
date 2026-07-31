"""Normative Mem0 metadata vocabulary registry (task 3195, leaf β).

This module is the **single normative home** for the Mem0 metadata
vocabulary defined by ``docs/prds/memory-metadata-vocabulary.md`` (V1).
Per INV-5 and PRD §6, consumers **import** from here — they never restate
the vocabulary.  A second copy of any constant in this module is a bug.

Contents
--------
* ``TOPIC_SLUG_RE`` / ``TOPIC_SLUG_MAX_LEN`` — the shared ``topic`` slug
  shape (PRD D4: ``ProceduralTopicCluster.topic_id`` and
  ``metadata.topic`` are one namespace with one regex).
* ``normalize_supersedes`` — PRD D2's scalar/list/None normalizer.

Measured basis for the slug shape
---------------------------------
Derived from leaf α's census
(``plans/memory-metadata-census-report.json`` @ ``b5af3e4b03``,
``coverage.complete = true``) rather than guessed:

* accepts all **5** seeded ``ProceduralTopicCluster.topic_id`` values
  (PRD §10's one hard requirement); longest is 52 chars;
* accepts **254 of 352** distinct live ``topic`` values (355 of 491
  records); the longest conforming live value is 69 chars, so the
  100-char cap bounds the key while rejecting nothing observed;
* the 98 non-conforming live values are all snake_case.  Under the
  warn-mode default (``memory_metadata.enforce = False``) these emit a
  census line and the write proceeds — leaf θ's bounded retro-stamping
  sweep is the intended normalizer.  This is why the warn default is
  load-bearing rather than merely cautious.

``kind`` is deliberately **NOT** slug-validated: 321 of the 329 live
``kind`` values are snake_case, so applying this regex to ``kind`` would
reject essentially the entire live population.  ``kind`` is
registry-membership-validated instead, exactly as V1 specifies.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    'KIND_REGISTRY',
    'TOPIC_SLUG_MAX_LEN',
    'TOPIC_SLUG_RE',
    'normalize_supersedes',
]


#: Shared ``topic`` slug shape (PRD D4 — one namespace for
#: ``ProceduralTopicCluster.topic_id`` and ``metadata.topic``).
#:
#: Lowercase alphanumeric segments joined by single hyphens.  Anchored at
#: both ends with ``\Z`` rather than ``$`` so a trailing newline cannot
#: sneak past (``$`` matches before a final ``\n``).
TOPIC_SLUG_RE = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*\Z')

#: Maximum ``topic`` slug length.  See the module docstring for the
#: measured basis (longest conforming live topic 69, longest seeded
#: cluster id 52 — 100 has headroom and rejects nothing observed).
TOPIC_SLUG_MAX_LEN = 100


def normalize_supersedes(value: Any) -> list[Any]:
    """Normalize a ``supersedes`` metadata value to a list (PRD D2).

    ``supersedes`` is a list in V1, but the corpus carries 81 records with
    a **scalar** value and 65 with a list.  The live scalar writer is
    ``reconciliation/harness.py:1167``; the readers are
    ``reconciliation/targeted.py:1464`` (truthiness discriminator) and
    leaf 3112's closure predicate.  Both go through this helper so the
    legacy scalar shape stays tolerated on read.

    Accepts ``None`` (→ ``[]``), a scalar (→ single-element list), or any
    non-``str`` sequence (→ list copy).  The returned list is always a
    fresh object, never an alias of the caller's list.

    This function **never drops or coerces members**.  A malformed member
    (short hex, non-string — the census counts 3 and 8 live respectively)
    survives normalization intact so that
    :func:`validate_memory_metadata` can reject it *by name*.  Silently
    dropping it here would be a silent-fail-soft: the write would succeed
    having quietly discarded a supersession edge.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    # Any other scalar (int, dict, ...) is wrapped rather than rejected —
    # the shape validator owns rejection, this function owns shape only.
    return [value]


#: The normative closed registry of Mem0 ``metadata.kind`` values
#: (PRD D3).  Membership is checked by :func:`validate_memory_metadata`
#: and enforced only when ``memory_metadata.enforce_kind_registry`` is
#: on — see the module docstring for why that flag defaults to False.
#:
#: Generated mechanically from the census artifact, NOT read at import
#: time: a registry that mutates when an artifact is regenerated is not
#: a registry.  ``tests/test_memory_metadata.py::TestKindRegistry`` loads
#: the artifact and asserts this literal against it, so a regeneration
#: that adds a kind fails the suite loudly instead of drifting silently.
KIND_REGISTRY: frozenset[str] = frozenset({
    # ---------------------------------------------------------------
    # BLOCK 1 — CENSUS-MEASURED (329 values)
    #
    # Source: plans/memory-metadata-census-report.json @ b5af3e4b03,
    # coverage.complete = true.  Measured over 49,628 records, of which
    # 47,150 (95.0%) carry NO `kind` at all (kind_missing), leaving
    # 2,478 kinded records across 329 distinct values.
    # Top five: cycle_summary (1,323), cgl_eta_cross_target_rehome (253),
    # task_completion_note (101), task_completion (68), completion_note (62).
    #
    # WARNING (PRD §10 open question 1): 242 of these 329 values are
    # SINGLETONS.  The population is agent-invented free text and open
    # in practice, which measured D3's "kind writers are in-repo code +
    # prompts" premise FALSE.  That is precisely why
    # `enforce_kind_registry` ships defaulted off: a day-one
    # strict-reject would turn every newly invented kind into a hard
    # memory-write failure on the live fleet.
    # ---------------------------------------------------------------
    'admin_cleanup_task_filed',
    'architectural_failure_mode',
    'batch_completion',
    'batch_completion_note',
    'block_analysis',
    'block_reason_documentation',
    'block_state_resolution_closure',
    'branch_arg_closure',
    'cancellation_note',
    'cancellation_rationale',
    'canonical_counting_methodology',
    'cfg_prd_chain_status',
    'cgl_eta_cross_target_rehome',
    'citation_provenance_discipline',
    'companion_marker_cleanup_canonical',
    'completion_capture',
    'completion_guard',
    'completion_knowledge',
    'completion_memory_existence_check_norm',
    'completion_note',
    'completion_note_backfill_tag',
    'completion_note_guard',
    'completion_note_unknown_provenance',
    'completion_observation',
    'completion_record',
    'condition_flag_type_dedup_norm',
    'consolidated_canonical',
    'consolidated_incident_open',
    'consolidated_merge',
    'consolidated_norm',
    'consolidated_observation',
    'consolidated_procedural_knowledge',
    'consolidation_closure_norm',
    'consolidation_closure_record',
    'consolidation_outcome',
    'consolidation_pointer',
    'consolidation_residual',
    'contradiction_resolved',
    'coordination_resolved',
    'corrected_block_diagnosis',
    'correction',
    'correction_note',
    'corrective_note',
    'corrective_stage2_summary',
    'cross_project_routing_deferral_note',
    'cross_project_routing_note',
    'curation_method',
    'curator_retained_short_sibling',
    'cycle_summary',
    'cycle_summary_correction',
    'cycle_summary_ledger_false_positive_note',
    'dead_letter_replay_confirmation',
    'decision_outcome_note',
    'dedup_guidance_correction',
    'deferral_note',
    'deletion_recovery',
    'delivered_check_reblock_guidance',
    'delivered_check_reblock_recurrence',
    'dep_chain_status',
    'dep_resolution_note',
    'design_tension',
    'deterministic_gate_escalation_missing_correction',
    'done_provenance_correction',
    'done_task_completion_note',
    'done_task_knowledge_capture',
    'dry_run_contradiction_note',
    'episode_disposition_note',
    'esc_5330_1_residual_correction',
    'escalation_5556_hint_sweep_complete',
    'escalation_lookup_anomaly',
    'evidence_refinement',
    'failure_mode',
    'false_done_correction',
    'fix_effectiveness_resolution',
    'flag_closure_note',
    'flag_correction',
    'flag_for_stage2',
    'flag_marker_id_mismatch_norm',
    'flag_resolution_note',
    'followup_prd_pointer',
    'followup_scope_note',
    'found_on_main',
    'found_on_main_spurious_stamp',
    'gate_evaluation_note',
    'gate_race_incident',
    'gate_reopen_record',
    'gate_resolution_rationale',
    'gate_task_reopen_rationale',
    'gotcha',
    'gr016_alpha_beta_resolution',
    'guard_backfill',
    'handoff_brief_norm',
    'historical_annotation',
    'human_curator_gate_status_correction',
    'human_gate_closure_confirmation',
    'human_gate_closure_discipline_recipe',
    'identification_aid',
    'incident_archive',
    'infra_observation',
    'investigation_finding',
    'investigation_outcome',
    'latent-bug',
    'live_workflow_resolution',
    'live_workflow_signal_citation_convention',
    'live_workflow_signal_granularity_caution',
    'live_workflow_suppression_disposition',
    'live_workflow_write_rejected',
    'main_gate_audit_hazard',
    'memory_contradiction_dedup_norm',
    'memory_correction',
    'merge_blocker_note',
    'merge_confirmation',
    'merge_queue_observation',
    'merged_duplicate',
    'norm_reinforcement',
    'norms_consolidation',
    'per_cycle_summary',
    'phantom_done_annotation',
    'phantom_done_finding',
    'phantom_done_recurrence_note',
    'post-reboot-resume',
    'prepared_consolidation_plan',
    'procedural_consolidation',
    'procedural_consolidation_gui_feature_gate',
    'procedural_consolidation_npx_vitest_tsc',
    'procedural_consolidation_ptodo_phantom_tracking',
    'procedural_consolidation_round1',
    'procedural_consolidation_round2',
    'procedural_consolidation_round2_corrected',
    'procedural_consolidation_round3',
    'procedural_consolidation_round4',
    'procedural_consolidation_round5',
    'procedural_consolidation_todo_own_number_cite',
    'procedural_convention',
    'procedural_correction',
    'procedural_gotcha',
    'procedural_heuristic',
    'procedural_knowledge_companion_guard',
    'procedural_norm',
    'procedural_rule',
    'procedure',
    'process_lesson_generalization',
    'promoted_procedural_pattern',
    'provenance_correction',
    'provenance_resolution',
    'provenance_smear_residual',
    'quota_outage_resolution_and_gap_note',
    'reasoning_correction',
    'rebase_integrity_note',
    'reblock_cohort_remediation',
    'recon_cycle_summary',
    'recon_guard',
    'recon_observation',
    'recon_probe_correction',
    'recon_procedure',
    'recon_stage2_norm',
    'reconciliation_action',
    'reconciliation_gotcha',
    'recurring_flag_standing_decision',
    'red_main_record',
    'refinement',
    'refresh_entity_summary_general_noop_repro',
    'remediation_action_record',
    'remediation_confirmation',
    'remediation_correction',
    'remediation_decision',
    'remediation_investigation',
    'remediation_norm',
    'remediation_note',
    'remediation_replacement',
    'remediation_resolution',
    'remediation_verification',
    'resolution_summary',
    'review_suggestions_triage',
    'root_cause',
    'root_cause_confirmation',
    'scheduler_churn_evidence',
    'scope_qualifier',
    'session_summary',
    'session_summary_continuation',
    'sigabrt_task_id_remap_correction',
    'site_count_correction',
    'snapshot_gap_root_cause',
    'snapshot_norm_cross_reference',
    'snapshot_write_norm_clarification',
    'snapshot_write_norm_corrected',
    'snapshot_write_prohibition_corrected',
    'soak_gate_clarification',
    'soak_gate_tracking',
    'split_out_fact',
    'stage1_consolidated_canonical',
    'stage1_consolidation_convention',
    'stage1_consolidation_merge',
    'stage1_cycle_summary',
    'stage1_discipline',
    'stage1_disposition',
    'stage1_finding_correction',
    'stage1_flag',
    'stage1_flag_followup',
    'stage1_flag_for_stage2',
    'stage1_flag_marker_content',
    'stage1_flag_relay',
    'stage1_flag_resolution',
    'stage1_flag_suppression',
    'stage1_flag_suppression_retirement',
    'stage1_flag_sweep_dual_key_fix',
    'stage1_investigation_note',
    'stage1_project_db_correction',
    'stage1_relay',
    'stage1_remediation_clarification',
    'stage1_remediation_consolidation',
    'stage1_remediation_correction',
    'stage1_remediation_finding',
    'stage1_remediation_note',
    'stage1_remediation_relay',
    'stage1_residual_scope_reanchor',
    'stage1_unresolved_relay',
    'stage2_bookkeeping_norm',
    'stage2_completion_guard',
    'stage2_completion_note',
    'stage2_correction',
    'stage2_cycle_summary',
    'stage2_deferral_note',
    'stage2_disposition',
    'stage2_disposition_note',
    'stage2_finding_correction',
    'stage2_flag_resolution',
    'stage2_guard',
    'stage2_procedural_norm',
    'stage2_procedure_note',
    'stage2_protective_annotation',
    'stage2_remediation_action',
    'stage2_remediation_note',
    'stage2_remediation_recommendation',
    'stage2_scope_decision',
    'stage2_suppress_backfill_guard',
    'stage2_suppress_guard',
    'stage2_suppress_guard_backfill',
    'stage2_sync',
    'stage2_task_reconciliation_note',
    'stage2_task_resolution',
    'stage2_verification',
    'stage2_write_convention',
    'stage3_finding_relay',
    'stage3_procedure',
    'stale_assertion_correction',
    'stale_blocker_verification',
    'stale_claim_corrected',
    'stale_completion_correction',
    'stale_memory_supersession',
    'status_correction',
    'stranded_gate_recurrence',
    'strategy_rescope_note',
    'stray_commit_collision_resolution',
    'supplement',
    'system_improvements_summary',
    'systemic_gap_resolved',
    'systemic_infra_pattern',
    'systemic_infra_pattern_2026_06_18',
    'systemic_pattern_observation',
    'systemic_pattern_status_confirmed',
    'task_absence_confirmation',
    'task_amendment_correction',
    'task_block_diagnosis_correction',
    'task_block_reason_correction',
    'task_blocking_context',
    'task_cancellation_rationale',
    'task_completion',
    'task_completion_capture',
    'task_completion_correction',
    'task_completion_enrichment',
    'task_completion_guard',
    'task_completion_knowledge',
    'task_completion_note',
    'task_completion_note_correction',
    'task_completion_note_guard',
    'task_completion_observation',
    'task_completion_reconciliation_note',
    'task_completion_state',
    'task_completion_summary',
    'task_completion_summary_guard_retag',
    'task_context',
    'task_correction_note',
    'task_count_snapshot',
    'task_count_snapshot_norm_supersession',
    'task_count_snapshot_recurrence_note',
    'task_decision_reissue',
    'task_dep_successor_resolution',
    'task_diagnosis_note',
    'task_dispatch_diagnosis',
    'task_disposition',
    'task_done_note',
    'task_fix_note',
    'task_incident_log',
    'task_knowledge_sync_note',
    'task_lifecycle_correction',
    'task_lifecycle_note',
    'task_lifecycle_reset_detected',
    'task_memory_correction',
    'task_memory_mismatch_clarification',
    'task_metadata_correction',
    'task_premise_correction',
    'task_reconciliation_finding',
    'task_recurrence_finding',
    'task_reopen_rationale',
    'task_resolution_correction',
    'task_resolution_note',
    'task_risk_observation',
    'task_routing_convention',
    'task_routing_note',
    'task_series_summary',
    'task_status_correction',
    'task_status_investigation',
    'task_telemetry_correction',
    'task_unblock_note',
    'taskid_renumber_285x_mem0_smear_correction',
    'technique',
    'terminal_state_correction',
    'test_design_norm',
    'trend_observation',
    'triage_pattern_occt_scope_timeout',
    'unblock_procedure',
    'unblock_resolution_guidance',
    'unknown_provenance_note',
    'vanishing_escalation_recheck_cohort',
    'verification_note',
    'verified_fix_confirmation',
    'verify_mechanism_fact',
    'zombie_reset_bypass_rule',
    # ---------------------------------------------------------------
    # BLOCK 2 — IN-REPO DECLARED, ZERO LIVE RECORDS (5 values)
    #
    # These five were named by the PRD's original §6 grandfather row
    # but measure ZERO live records (esc-3194-1, re-verified against
    # main @ 2814b01202).  They are retained anyway: grandfathering
    # means "what is WRITTEN", not only what survives an aging sweep.
    # Dropping them would reject in-repo code the moment
    # `enforce_kind_registry` flips.
    # ---------------------------------------------------------------
    'consolidated_scope_correction',  # reconciliation/scope_freshness.py:97 (declared), written :251, :495
    'project_status_correction',      # reconciliation/harness.py:1166 (same dict as the scalar `supersedes` at :1167)
    'count_snapshot_cleanup_audit',   # scripts/cleanup_count_snapshots.py:210
    # No live Mem0 writer found — retained per esc-3194-1 pending the
    # PRD §10 open questions.  `entity_standing_decision` is a SQLite
    # recon_ledger record kind (reconciliation/standing_decision_writer.py,
    # list_entity_standing_decisions), i.e. a DIFFERENT store's vocabulary;
    # `stage1_flag_marker`'s Mem0 mirror write was RETIRED by task 2406
    # (see reconciliation/flag_dedup.py:10), leaving only prompt/doc
    # references.  Both are harmless allowlist entries.
    'entity_standing_decision',
    'stage1_flag_marker',
    # ---------------------------------------------------------------
    # BLOCK 3 — NEW IN THIS PRD (2 values)
    #
    # V1 child kinds, triage attach outcomes only.  Both confirmed
    # ABSENT from the live corpus, so they are added explicitly rather
    # than inherited from the census.
    # ---------------------------------------------------------------
    'amendment',
    'sighting',
})
