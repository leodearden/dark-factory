"""The registered escalation-category vocabulary and its one check.

Every filed ``Escalation.category`` must be a member of
:class:`EscalationCategory`. A category is registered iff it has a live
producer, a live reader, an agent/skill instruction, or a record still pending
in some queue.

Enforced by :func:`require_registered_category` at ``EscalationQueue.submit``,
``submit_resolved``, ``has_open_l1`` and ``find_terminal_by_citation``, and by
the ``escalate_blocker``, ``escalate_info`` and ``promote_to_l2`` tools.
Rehydrating a stored record is never checked. ``escalation/tests/
test_category_census.py`` fails verify on a filer passing an unregistered one.
"""

from __future__ import annotations

import difflib
from enum import StrEnum, unique


@unique
class EscalationCategory(StrEnum):
    ADJUDICATOR_CONFIG_DEFECT = 'adjudicator_config_defect'
    ALREADY_LANDED = 'already_landed'
    BRANCH_RESET = 'branch_reset'
    BYPASS_DONE = 'bypass_done'
    CANDIDATE_KEY_RESIDUAL_DUPLICATES = 'candidate_key_residual_duplicates'
    CASCADE_STATUS_REJECTION = 'cascade_status_rejection'
    CLEANUP_NEEDED = 'cleanup_needed'
    CROSS_REPO_DELIVERABLE = 'cross_repo_deliverable'
    CURATOR_ADJUDICATION_MISSING = 'curator_adjudication_missing'
    CURATOR_BACKEND_BINARY_UNRESOLVABLE = 'curator_backend_binary_unresolvable'
    CURATOR_CONSECUTIVE_DEGRADED = 'curator_consecutive_degraded'
    CURATOR_FAILURE = 'curator_failure'
    CURATOR_SCHEMA_TOOL_DENIED = 'curator_schema_tool_denied'
    CURATOR_ZERO_OUTPUT_HANG = 'curator_zero_output_hang'
    CURATOR_ZOT_DUPLICATE = 'curator_zot_duplicate'
    DEPENDENCY_CAPABILITY = 'dependency_capability'
    DEPENDENCY_DISCOVERED = 'dependency_discovered'
    DESIGN_CONCERN = 'design_concern'
    DURABLE_WRITE_DEAD_LETTER = 'durable_write_dead_letter'
    ENTITY_MINT_STORM = 'entity_mint_storm'
    EVAL_REGRESSION = 'eval_regression'
    ILLEGAL_DEPLOY_TRANSITION = 'illegal_deploy_transition'
    INFRA_ISSUE = 'infra_issue'
    INTEGRATION_SKEW = 'integration_skew'
    LANDING_DETECTOR_GIT_ERROR = 'landing_detector_git_error'
    LIMIT_EXHAUSTED = 'limit_exhausted'
    MCP_MARKUP_BOUNDARY_STORM = 'mcp_markup_boundary_storm'
    MCP_MARKUP_RESIDUE = 'mcp_markup_residue'
    MCP_MARKUP_STORM = 'mcp_markup_storm'
    MCP_MARKUP_WRITE_STORM = 'mcp_markup_write_storm'
    MEM0_IN_PLACE_UPDATE_STORM = 'mem0_in_place_update_storm'
    MEMORY_METADATA_UNKNOWN_KEY_STORM = 'memory_metadata_unknown_key_storm'
    MERGE_COALESCE_LIVE_ORIGINAL = 'merge_coalesce_live_original'
    MERGE_DEFERRED_ABANDONED = 'merge_deferred_abandoned'
    MERGE_ERROR = 'merge_error'
    MERGE_FF_FAILED = 'merge_ff_failed'
    MERGE_FLAKE_SUPPRESSION_STORM = 'merge_flake_suppression_storm'
    MERGE_LIFECYCLE_TRANSITION_REJECTED = 'merge_lifecycle_transition_rejected'
    MERGE_REQUEST_STUCK = 'merge_request_stuck'
    MERGE_RESOURCE_LEAK = 'merge_resource_leak'
    MERGE_SCOPE_MISMATCH = 'merge_scope_mismatch'
    MILESTONE_CHECK_FAILED = 'milestone_check_failed'
    MILESTONE_CHECK_STALLED = 'milestone_check_stalled'
    MILESTONE_GATE = 'milestone_gate'
    MISSING_PREMISE = 'missing_premise'
    OUT_OF_SCOPE = 'out_of_scope'
    PLANNING_STALLED = 'planning_stalled'
    POST_MERGE_VERIFY = 'post_merge_verify'
    PREEXISTING_MAIN_BREAK = 'preexisting_main_break'
    PROVENANCE_CONFLICT = 'provenance_conflict'
    PROVENANCE_UNATTRIBUTED = 'provenance_unattributed'
    RECON_AUTO_CONSOLIDATION_BACKLOG = 'recon_auto_consolidation_backlog'
    RECON_AUTO_CONSOLIDATION_REFUSED = 'recon_auto_consolidation_refused'
    RECON_BACKLOG_OVERFLOW = 'recon_backlog_overflow'
    RECON_FAILURE = 'recon_failure'
    RECON_INTEGRITY_ISSUE = 'recon_integrity_issue'
    RECON_MISSING_INDEX = 'recon_missing_index'
    RECON_REMEDIATION_PHANTOM_CITATION_STORM = 'recon_remediation_phantom_citation_storm'
    RECON_REMEDIATION_PLACEHOLDER_STORM = 'recon_remediation_placeholder_storm'
    RECON_RESUME_FAILURE_STORM = 'recon_resume_failure_storm'
    RECON_STALE_RUN = 'recon_stale_run'
    RECON_STALE_TASK_COUNT_SNAPSHOT = 'recon_stale_task_count_snapshot'
    RECON_TASK_FINDING = 'recon_task_finding'
    RECON_WATCHDOG_KILL_STORM = 'recon_watchdog_kill_storm'
    RECONCILE_PERSISTENT_REJECTION = 'reconcile_persistent_rejection'
    RECONCILIATION_GROWTH_SWEEP_FAILURE = 'reconciliation_growth_sweep_failure'
    RECONCILIATION_PRESERVATION_SPECIMEN_STORM = 'reconciliation_preservation_specimen_storm'
    RECONCILIATION_STALE_GATE_BACKLOG = 'reconciliation_stale_gate_backlog'
    RECONCILIATION_STALE_HUMAN_OPERATOR = 'reconciliation_stale_human_operator'
    RECONCILIATION_STANDING_DECISION_STORM = 'reconciliation_standing_decision_storm'
    REFERENT_REPAIR_STORM = 'referent_repair_storm'
    REGRESSED_AFTER_RESOLUTION = 'regressed_after_resolution'
    RETRY_CAP_EXHAUSTED = 'retry_cap_exhausted'
    REVIEW_DUE = 'review_due'
    REVIEW_ISSUES = 'review_issues'
    REVIEW_SUGGESTIONS = 'review_suggestions'
    RISK_IDENTIFIED = 'risk_identified'
    SCHEDULER_PAUSED = 'scheduler_paused'
    SCOPE_VIOLATION = 'scope_violation'
    STASH_FAILED = 'stash_failed'
    STOP_INSTRUCTION = 'stop_instruction'
    STRANDED_BLOCKED = 'stranded_blocked'
    STRANDED_MERGE_FAILED = 'stranded_merge_failed'
    TASK_FAILURE = 'task_failure'
    TICKET_FAILURE = 'ticket_failure'
    UNMERGED_STATE = 'unmerged_state'
    UNVERIFIED_COMPLETION_CLAIM = 'unverified_completion_claim'
    VALIDATION_LANE_RED = 'validation_lane_red'
    VERIFY_CROSS_CHECK_MISMATCH ='verify_cross_check_mismatch'
    VERIFY_DRIFT_DIVERGENCE = 'verify_drift_divergence'
    VERIFY_ENV_DIVERGENCE = 'verify_env_divergence'
    VERIFY_HOST_UNREACHABLE = 'verify_host_unreachable'
    VERIFY_WORKTREE_CONTENTION = 'verify_worktree_contention'
    WIP_CONFLICT = 'wip_conflict'
    WORKFLOW_EXIT_CONTRACT = 'workflow_exit_contract'
    WORKFLOW_SLOT_WEDGED = 'workflow_slot_wedged'
    WRITE_CLASSIFIER_FALLBACK_STORM = 'write_classifier_fallback_storm'
    WRITE_TRIAGE_FAIL_OPEN_STORM = 'write_triage_fail_open_storm'


_REGISTERED_VALUES: frozenset[str] = frozenset(m.value for m in EscalationCategory)


class UnregisteredEscalationCategory(ValueError):
    """A category outside :class:`EscalationCategory` reached an enforcement point."""


def require_registered_category(category: object) -> None:
    """Raise :class:`UnregisteredEscalationCategory` unless *category* is registered.

    Membership is exact: no stripping, no case folding.
    """
    if isinstance(category, str) and category in _REGISTERED_VALUES:
        return
    message = (
        'escalation category must be a member of '
        f'escalation.categories.EscalationCategory; got {category!r}'
    )
    close = difflib.get_close_matches(str(category), sorted(_REGISTERED_VALUES), n=3)
    if close:
        message += f' (did you mean: {", ".join(close)}?)'
    raise UnregisteredEscalationCategory(message)
