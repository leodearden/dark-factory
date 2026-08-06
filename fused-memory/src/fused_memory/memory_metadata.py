"""Normative Mem0 metadata vocabulary registry (task 3195, leaf β).

This module is the **single normative home** for the Mem0 metadata
vocabulary defined by ``docs/prds/memory-metadata-vocabulary.md`` (V1).
Per INV-5 and PRD §6, consumers **import** from here — they never restate
the vocabulary.  A second copy of any constant in this module is a bug.

Contents
--------
* ``TOPIC_SLUG_RE`` / ``TOPIC_SLUG_MAX_LEN`` / ``is_valid_topic_slug`` —
  the shared ``topic`` slug shape (PRD D4:
  ``ProceduralTopicCluster.topic_id`` and ``metadata.topic`` are one
  namespace with one regex).  **Defined in**
  :mod:`fused_memory.topic_slug` (task 3198, leaf ε) and re-exported here
  bound to the *same objects*, so this registry remains the advertised
  import site while ``config/schema.py`` — which cannot import this
  module without a cycle — validates through the identical rule.  See
  that module's docstring for the measured ``ImportError`` behind the
  split.
* ``normalize_supersedes`` — PRD D2's scalar/list/None normalizer.

``kind`` is deliberately **NOT** slug-validated: 321 of the 329 live
``kind`` values are snake_case, so applying this regex to ``kind`` would
reject essentially the entire live population.  ``kind`` is
registry-membership-validated instead, exactly as V1 specifies.

Layering note
-------------
This module imports **downward** into :mod:`fused_memory.backends.mem0_client`
for :data:`MEM0_MANAGED_METADATA_KEYS`.  That direction is deliberate, not
accidental: PRD D12 / task 3055 §6 name ``backends/mem0_client.py`` as the
decided home for that set, and INV-5 forbids a second copy.  The name is
re-exported here (bound to the *same object*) so callers classifying keys
have one import site.

That single home has a MEASURED COST, stated here rather than left for a
consumer to discover: ``backends/mem0_client.py`` does ``from mem0 import
AsyncMemory`` and ``from fused_memory.config.schema import
FusedMemoryConfig`` at module scope, so importing this registry transitively
pulls in the mem0 SDK and the pydantic-settings config model.  This module
is therefore **not import-cheap**, and an out-of-process consumer that
cannot install ``mem0`` cannot import it at all.  Making the import
genuinely light would mean re-homing the set into a leaf module — a change
to a home decided by task 3055 §6 / PRD D12 and still awaited by task 3088,
so it is deliberately NOT made unilaterally here.

What the split from :mod:`fused_memory.services.memory_metadata_census`
*does* buy — and the reason the observability half (census log line, storm
detector, escalation filer) lives there rather than here — is narrower than
"cheap import" but real: this module carries **no process-lifetime mutable
state, no queue writer, and no dependency on the optional ``escalation``
package**, so leaf ι's prompt-pinning tests and the drift test import a
pure vocabulary rather than a running subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS

# Re-exported, NOT redefined (task 3198, leaf ε).  These are bound to the
# *same objects* the leaf module defines, so this registry remains the
# advertised import site for the vocabulary while ``config/schema.py`` —
# which cannot import this module without the measured
# ``ImportError: cannot import name 'FusedMemoryConfig'`` cycle — validates
# through the identical rule.  Per INV-5 there is one copy of the pattern
# and the cap in the tree; ``tests/test_topic_slug_namespace.py`` asserts
# that by ``is``, so retyping either here fails by design.
from fused_memory.topic_slug import (
    TOPIC_SLUG_MAX_LEN,
    TOPIC_SLUG_RE,
    is_valid_topic_slug,
)

# The canonical-full-UUID predicate that used to live here (task 3132, leaf η).
# It moved to ``utils/validation.py`` so the delete_memory guards, recon_report's
# citation gate and citation_verifier's forwarding-pointer guard all answer
# "is this a canonical full UUID" from ONE definition (INV-5).
from fused_memory.utils.validation import is_full_uuid

#: Back-compat alias for the pre-move private spelling, bound to the *same
#: object* — the same re-export-not-redefine treatment applied to
#: ``TOPIC_SLUG_RE`` above, for the same INV-5 reason.  ``scripts/
#: retro_stamp_topics.py`` imports this name and
#: ``tests/test_retro_stamp_topics.py`` pins the binding by ``is``, so the
#: alias is load-bearing rather than cosmetic.  New callers should import
#: ``is_full_uuid`` from :mod:`fused_memory.utils.validation` directly.
_is_full_uuid = is_full_uuid

__all__ = [
    'BLESSED_METADATA_KEYS',
    'KIND_REGISTRY',
    'MEM0_MANAGED_METADATA_KEYS',
    'CanonicalUniquenessViolation',
    'MemoryMetadataValidationError',
    'MetadataViolation',
    'RESERVED_VOCABULARY_KEYS',
    'SERVER_STAMPED_KEYS',
    'TOPIC_SLUG_MAX_LEN',
    'TOPIC_SLUG_RE',
    'classify_unknown_keys',
    'is_valid_topic_slug',
    'normalize_supersedes',
    'validate_memory_metadata',
]

#: Prefix marking a deliberately-unvalidated experimental key (Tier-C).
#: Ported from the task-metadata boundary (``shared/src/shared/task_metadata.py``),
#: which uses the same escape hatch for the same reason: a writer that knows
#: it is experimenting should be able to say so without manufacturing census
#: noise.
EXPERIMENTAL_KEY_PREFIX = 'x_'


def normalize_supersedes(value: Any) -> list[Any]:
    """Normalize a ``supersedes`` metadata value to a list (PRD D2).

    This docstring is the SINGLE HOME of the ``supersedes`` writer/reader
    map; the other sites that touch the key point here instead of
    re-deriving it, and name SYMBOLS rather than line numbers (a pinned
    line number is falsified by the next edit made above it — the churn
    that motivated this consolidation).

    ``supersedes`` is a list in V1, but the corpus carries 81 records with
    a **scalar** value and 65 with a list.  The writer —
    ``ReconciliationHarness._reconcile_status_correction`` in
    ``reconciliation/harness.py`` — emitted a scalar until task 3196
    migrated it to the canonical list shape; the 81 measured records
    predate that migration and are not rewritten by it (PRD D2 defers
    retro normalization to leaf θ's stamping sweep), which is why read
    tolerance for the legacy scalar stays.  The readers are
    ``reconciliation.targeted._is_authoritative_resolution`` (truthiness
    discriminator, which tests ``any()`` of the normalized members — see
    ITS docstring for why member-level and not container truthiness) and
    leaf 3112's closure predicate.  Both go through this helper — INV-5:
    never a second ``supersedes`` parser.

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
    # ReconciliationHarness._reconcile_status_correction (same dict as `supersedes`)
    'project_status_correction',
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


# ---------------------------------------------------------------------------
# Top-level metadata key layers
#
# Four deliberately DISJOINT layers.  A key classified twice would make the
# census line ambiguous about who owns it, so
# ``tests/test_memory_metadata.py::TestKeyLayers`` asserts pairwise
# disjointness across all six pairs.  ``run_id`` is the live trap: it is
# server-stamped AND mem0-managed AND measures 4,518 live occurrences, so it
# is the one key that would otherwise land in three tiers at once.  It is
# classified under the mem0-managed layer only (see below) -- that layer is
# the one with an external owner, so it wins the tie.
# ---------------------------------------------------------------------------

#: Keys this server stamps onto metadata itself.  They are NOT all stamped at
#: the same layer, and a reviewer must not read these as three write-seam
#: stamps:
#:
#: * ``category``   -- stamped at the write seam:
#:                     ``memory_service.py:2193`` (``add_memory``) and
#:                     ``:2542`` (``add_system_record``).
#: * ``recon_pool`` -- stamped inside ``_apply_cycle_summary_metadata_tagging``
#:                     (``memory_service.py:389``), cycle-summary writes only.
#: * ``planned``    -- NOT a write-seam stamp at all.  It is a server-owned
#:                     SEARCH-RESULT annotation (``memory_service.py:1932``,
#:                     ``:2921``, read back at ``:2962``).  It is listed here
#:                     so that a round-tripped search result re-written as
#:                     metadata does not census-warn on the server's own field.
#:
#: DELIBERATELY ABSENT: ``run_id``.  It *is* server-stamped, by the same
#: ``_apply_cycle_summary_metadata_tagging`` helper (``memory_service.py:389``)
#: as ``recon_pool`` -- but mem0 also owns and re-derives it, so it is already
#: carried by :data:`MEM0_MANAGED_METADATA_KEYS`
#: (``backends/mem0_client.py:35``).  Listing it in both sets would break the
#: disjointness invariant above and leave the census line ambiguous about who
#: owns the key.  Classification is unaffected either way --
#: :data:`_KNOWN_METADATA_KEYS` is a union -- so the single-home rule decides.
SERVER_STAMPED_KEYS: frozenset[str] = frozenset({
    'category',
    'recon_pool',
    'planned',
})

#: The five keys this PRD reserves and gives shape rules to (V1).  Every one
#: is OPTIONAL: absence is never a violation.
RESERVED_VOCABULARY_KEYS: frozenset[str] = frozenset({
    'topic',
    'canonical',
    'kind',
    'parent_id',
    'supersedes',
})

#: Tier-A: conventional keys that are deliberate, documented and high-volume,
#: and so must NOT manufacture census noise.  Ported from the task-metadata
#: boundary's ``_BLESSED_METADATA_KEYS``.
#:
#: Seeded from leaf α's census at a stated **count >= 1000** cut, excluding
#: anything already mem0-managed, server-stamped or reserved (counts inline,
#: from plans/memory-metadata-census-report.json @ b5af3e4b03).
#:
#: WHY THIS TIER IS MANDATORY: the corpus carries 1,627 distinct top-level
#: keys, 715 of them singletons.  Without a blessed tier, essentially every
#: write emits an unknown-key census line -- which drowns the signal the
#: census line exists to produce and makes the storm detector fire
#: permanently against normal traffic, i.e. the warn path logged into
#: oblivion (precisely what INV-4 exists to prevent).
#:
#: Everything BELOW the cut keeps warning.  That is the intended Tier-B/C
#: drift signal, not noise: the 715 singleton keys are what a census is for.
BLESSED_METADATA_KEYS: frozenset[str] = frozenset({
    'task_id',               # 18,850
    'source',                # 16,364
    'transition',            # 15,529
    '_deferred',             #  8,263
    '_causation_id',         #  8,262
    'stage',                 #  2,865
    'stage2_suppress',       #  1,588
    'echo_used_provenance',  #  1,284
})

#: Union of every known layer, precomputed for the unknown-key scan.
_KNOWN_METADATA_KEYS: frozenset[str] = (
    MEM0_MANAGED_METADATA_KEYS
    | SERVER_STAMPED_KEYS
    | RESERVED_VOCABULARY_KEYS
    | BLESSED_METADATA_KEYS
)


def classify_unknown_keys(meta: dict[str, Any]) -> list[str]:
    """Return the top-level keys of *meta* that belong to no known layer.

    A key is known if it is mem0-managed, server-stamped, reserved
    vocabulary, blessed conventional, or carries the ``x_`` experimental
    prefix.  Structurally identical to the task-metadata boundary's
    unknown-key scan (``shared/src/shared/task_metadata.py:912-920``).

    Returns a list (not a set) in the caller's key order so the census line
    and any escalation summary are stable and readable.
    """
    return [
        key
        for key in meta
        if key not in _KNOWN_METADATA_KEYS
        and not key.startswith(EXPERIMENTAL_KEY_PREFIX)
    ]


# ---------------------------------------------------------------------------
# Violations and the strict-mode error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetadataViolation:
    """A single metadata shape/vocabulary violation.

    Modelled on the task-metadata boundary's ``SchemaWarning``
    (``shared/src/shared/task_metadata.py:680``) plus the ``fatal`` flag
    this domain needs: an unknown KEY only ever censuses, while a
    malformed reserved key is rejectable once ``memory_metadata.enforce``
    is on.  Frozen so violations are hashable and directly assertable.

    :param key:     the offending top-level metadata key.
    :param code:    stable machine code, e.g. ``invalid_topic_slug``.  This
                    is what the census line reports and what operators grep.
    :param message: human-readable text naming the violated RULE and the
                    registry location (see
                    :class:`MemoryMetadataValidationError`).
    :param fatal:   whether this violation would reject the write under
                    ``enforce``.  Non-fatal violations always still census.
    """

    key: str
    code: str
    message: str
    fatal: bool


class MemoryMetadataValidationError(Exception):
    """Raised at the write seam when ``memory_metadata.enforce`` is on and at
    least one violation is fatal.

    V1 requires the rejection hint to name BOTH the violated rule and where
    the registry lives, so an agent that trips it can find the vocabulary
    rather than guessing.  That is satisfied *by construction* here: every
    violation ``message`` produced by :func:`validate_memory_metadata`
    already names its rule, and this class appends the registry location, so
    the guarantee does not depend on each call site remembering to add it.
    """

    #: Where the vocabulary lives.  Named in every rejection message.
    REGISTRY_LOCATION = 'fused_memory.memory_metadata'

    def __init__(self, violations: list[MetadataViolation]) -> None:
        self.violations = list(violations)
        detail = '; '.join(
            f'{v.code}: {v.message}' for v in self.violations
        )
        super().__init__(
            f'memory metadata rejected ({len(self.violations)} violation(s)): '
            f'{detail} -- the metadata vocabulary is defined in '
            f'{self.REGISTRY_LOCATION}'
        )


class CanonicalUniquenessViolation(Exception):
    """A second ``canonical`` write for a ``(project, topic)`` already taken.

    V1 contract-fixes this type alongside
    :class:`MemoryMetadataValidationError`, and requires the message to
    NAME the incumbent's id -- counting is not enough, because an operator
    who trips this needs to go look at the record that already holds the
    topic. That obligation is why the seam counts first and only then
    scrolls: ``count_memories_by_metadata`` is the contracted INV-3 gate
    but returns an ``int``, which structurally cannot carry an id.

    Deliberately **NOT** a subclass of
    :class:`MemoryMetadataValidationError`. The two are separate rejection
    contracts -- one is "your metadata is malformed", the other is "your
    metadata is well-formed but collides with live state" -- and a caller
    must be able to tell them apart at the ``except``. Subclassing would
    also let the pre-existing
    ``pytest.raises(MemoryMetadataValidationError)`` sites silently
    swallow a uniqueness violation, i.e. pass for the wrong reason. The
    structured fields below satisfy INV-2's structured-rejection
    obligation without needing inheritance.

    Raised ONLY from
    :func:`~fused_memory.services.memory_service._apply_memory_metadata_validation`,
    never from :func:`validate_memory_metadata`: the check needs live
    store state, and the pure validator structurally cannot perform a
    store lookup. Look for it at the seam, not in the shape validator.
    """

    def __init__(self, *, project_id: str, topic: str, incumbent_id: str) -> None:
        self.project_id = project_id
        self.topic = topic
        self.incumbent_id = incumbent_id
        super().__init__(
            f'canonical uniqueness violated: memory {incumbent_id} is already '
            f'canonical for topic {topic!r} in project {project_id!r}; at most '
            f'one canonical memory is allowed per (project, topic). Update or '
            f'supersede that memory instead of adding a second canonical -- the '
            f'metadata vocabulary is defined in '
            f'{MemoryMetadataValidationError.REGISTRY_LOCATION}'
        )


# ---------------------------------------------------------------------------
# The shape validator
# ---------------------------------------------------------------------------

#: Appended to every violation message so
#: :class:`MemoryMetadataValidationError`'s V1 hint obligation ("name the
#: rule AND where the registry lives") is satisfied BY CONSTRUCTION rather
#: than by each call site remembering to add it.
_REGISTRY_HINT = f'(vocabulary: {MemoryMetadataValidationError.REGISTRY_LOCATION})'


def _violation(key: str, code: str, rule: str, *, fatal: bool) -> MetadataViolation:
    """Build a violation whose message names the rule and the registry."""
    return MetadataViolation(
        key=key, code=code, message=f'{rule} {_REGISTRY_HINT}', fatal=fatal
    )


def validate_memory_metadata(
    meta: dict[str, Any], *, enforce_kind_registry: bool
) -> list[MetadataViolation]:
    """Shape-check the reserved Mem0 metadata vocabulary (PRD V1, leaf β).

    Returns **every** violation found — it never short-circuits at the first
    failure, because a validator that stopped early would make an operator
    fix one violation per write attempt.  Returning violations rather than
    raising mirrors ``parse_metadata``'s ``(value, warnings)`` contract
    (``shared/src/shared/task_metadata.py``): the CALLER owns the
    warn-vs-reject decision, driven by ``memory_metadata.enforce``.

    *meta* is mutated in place in exactly one way: a ``supersedes`` value is
    normalized to a list (PRD D2).  See
    :func:`normalize_supersedes` and the β-before-γ note below.

    SCOPE — SHAPE ONLY.  This is a pure synchronous function taking only a
    dict, so it structurally **cannot** perform a store lookup.  That is
    deliberate, not an oversight:

    * ``parent_id`` **liveness** (does the parent still exist?) is leaf δ's,
    * ``canonical`` per-(project, topic) **uniqueness** is leaf ε's, and
      lives at the async seam
      (:func:`~fused_memory.services.memory_service._apply_memory_metadata_validation`)
      because it needs live store state.

    Leaf ε (task 3198) split ``canonical`` along exactly that line.  Its
    SHAPE half — a canonical assertion requires a ``topic`` (§2b's
    ``canonical_without_topic``) — is checked HERE, because it needs
    nothing but the dict.  Its UNIQUENESS half stays at the seam.  The
    boundary moved no work into this function that a store lookup could
    have been needed for.

    Making the boundary structural rather than a comment means a later leaf
    must add liveness at the seam and cannot accidentally grow a second
    implementation of it in here (INV-5).

    Ordering inside the function is load-bearing:

    1. normalize ``supersedes`` first, so the member checks below see a
       uniform list regardless of which live shape arrived;
    2. shape-check each reserved key that is PRESENT — absence is never a
       violation, since none of V1's five keys is required;
    3. classify unknown top-level keys (census-only, never fatal).

    :param enforce_kind_registry: whether an unknown ``kind`` is fatal.  The
        violation is emitted either way — this flag only decides whether it
        can reject.  It exists because leaf α measured PRD D3's stated
        premise ("kind writers are in-repo code + prompts, enumerated and
        grandfathered") FALSE: 242 of the 329 live ``kind`` values are
        singletons, i.e. the population is agent-invented free text and open
        in practice.  Day-one strict rejection would convert every newly
        invented kind into a hard memory-write failure on the live fleet.
        See PRD §10 open question 1.
    """
    violations: list[MetadataViolation] = []

    # 1. supersedes — NORMALIZE, never reject the container shape.
    #
    # The β-before-γ hazard: PRD §9 sequenced γ after β, and
    # `reconciliation/harness.py` wrote a SCALAR `supersedes` (81 live
    # records) for that whole window.  Rejecting the scalar form here would
    # have broken the recon harness's own writes until γ migrated it.  That
    # window is now CLOSED — task 3196 (γ) migrated that writer
    # (`ReconciliationHarness._reconcile_status_correction`) to the canonical
    # list shape.
    #
    # The in-place coercion below is nonetheless RETAINED as
    # defense-in-depth, not left over: no corpus rewrite shipped with γ, so
    # the 81 pre-migration records still carry scalars (PRD D2 defers retro
    # normalization to leaf θ's stamping sweep), and out-of-repo writers are
    # not bound by the in-repo migration.  Coercing at the write boundary
    # mirrors `_normalize_task_id_metadata`, which already does exactly this
    # for the same class of problem in the same two functions, and is
    # compatible with V1's "legacy scalar tolerated on read".  Only malformed
    # MEMBERS are rejected.
    if 'supersedes' in meta:
        members = normalize_supersedes(meta['supersedes'])
        meta['supersedes'] = members
        for member in members:
            if not is_full_uuid(member):
                violations.append(_violation(
                    'supersedes',
                    'invalid_supersedes_member',
                    f'supersedes member {member!r} must be a canonical full-UUID '
                    f'string; the census counts 3 live short-hex and 8 live '
                    f'non-string members',
                    fatal=True,
                ))

    # 2a. topic — the shared slug namespace (PRD D4).
    if 'topic' in meta:
        topic = meta['topic']
        if not isinstance(topic, str):
            violations.append(_violation(
                'topic',
                'invalid_topic_type',
                f'topic must be a str, got {type(topic).__name__}',
                fatal=True,
            ))
        # `is_valid_topic_slug`, NOT an inlined `TOPIC_SLUG_RE.match(...) or
        # len(...) >` — the predicate is THE shared verdict (D4: one
        # namespace, one answer), and the config-side validator and the
        # seam's uniqueness guard both call it.  Re-expressing the rule here
        # would split the closure the moment the predicate gains a step
        # (a normalization pass, a reserved-prefix rule): the other two call
        # sites would get it and the shape validator would not.  The
        # `isinstance(topic, str)` branch above already separated the type
        # defect from the shape defect, so both violation codes survive.
        elif not is_valid_topic_slug(topic):
            violations.append(_violation(
                'topic',
                'invalid_topic_slug',
                f'topic {topic!r} must match TOPIC_SLUG_RE '
                f'{TOPIC_SLUG_RE.pattern!r} and be at most '
                f'{TOPIC_SLUG_MAX_LEN} chars',
                fatal=True,
            ))

    # 2b. canonical — bools only.
    if 'canonical' in meta:
        canonical = meta['canonical']
        # isinstance(..., bool), NOT truthiness: `1 == True` in Python, so a
        # truthiness check would silently accept the int form.  The census
        # counts exactly 1 live non-bool value — the record this rule exists
        # to surface.
        if not isinstance(canonical, bool):
            violations.append(_violation(
                'canonical',
                'invalid_canonical_type',
                f'canonical must be a bool, got {type(canonical).__name__} '
                f'({canonical!r}); the check is isinstance(x, bool), not '
                f'truthiness, so 1/0 do not pass as True/False',
                fatal=True,
            ))
        # `is True`, not truthiness — for the reason 2b already documents
        # above (`1 == True`), and so a non-bool value reports only the TYPE
        # defect rather than being double-counted here.
        #
        # Only the `topic` KEY's presence is tested: when it is present but
        # malformed, 2a's invalid_topic_slug already describes that defect,
        # and emitting both would report one defect twice while implying the
        # writer forgot a key they did supply.
        elif canonical is True and 'topic' not in meta:
            violations.append(_violation(
                'canonical',
                'canonical_without_topic',
                'canonical=True requires a topic: the marker asserts "this is '
                'THE entry for its topic", so outside a topic there is no set '
                'for it to be canonical of and the <=1-per-(project, topic) '
                'uniqueness rule has no key to check',
                fatal=True,
            ))

    # 2c. kind — registry membership, NOT slug shape.
    #
    # 321 of the 329 live `kind` values are snake_case, so applying
    # TOPIC_SLUG_RE here would reject essentially the whole live population.
    if 'kind' in meta:
        kind = meta['kind']
        if not isinstance(kind, str):
            violations.append(_violation(
                'kind',
                'invalid_kind_type',
                f'kind must be a str, got {type(kind).__name__}',
                fatal=True,
            ))
        elif kind not in KIND_REGISTRY:
            violations.append(_violation(
                'kind',
                'unknown_kind',
                f'kind {kind!r} is not in KIND_REGISTRY '
                f'({len(KIND_REGISTRY)} grandfathered values)',
                fatal=enforce_kind_registry,
            ))

    # 2d. parent_id — SHAPE only; liveness is leaf δ's (see the docstring).
    if 'parent_id' in meta:
        parent_id = meta['parent_id']
        if not is_full_uuid(parent_id):
            violations.append(_violation(
                'parent_id',
                'invalid_parent_id_shape',
                f'parent_id {parent_id!r} must be a canonical full-UUID string '
                f'(shape only — liveness resolution is leaf δ\'s)',
                fatal=True,
            ))

    # 3. Unknown top-level keys — census-only, never fatal.
    #
    # 1,627 distinct live keys, 715 of them singletons.  This axis exists to
    # PRODUCE a drift signal, not to reject; making it fatal would turn the
    # long tail into an outage the moment `enforce` flipped.
    for key in classify_unknown_keys(meta):
        violations.append(_violation(
            key,
            'unknown_key',
            f'metadata key {key!r} belongs to no known layer '
            f'(mem0-managed / server-stamped / reserved / blessed) and lacks '
            f'the {EXPERIMENTAL_KEY_PREFIX!r} experimental prefix',
            fatal=False,
        ))

    return violations
