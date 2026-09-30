"""Task status TRANSITION AUTHORITY — Table A of the task-status-authority contract.

See PRD ``plans/task-status-authority-prd.md`` (C1/D1/D5, findings 4.1 and the
TABLE half of finding 1.2). This module is the pure, shared companion to
``shared.task_statuses`` (the vocabulary, shipped by task 2163): it adds the
actor taxonomy (``ActorClass``, ``derive_actor_class``), the enumerated
(from, to, actor) -> allowed legality table (``TRANSITIONS``,
``is_legal_transition``), and the WorkflowOutcome -> TaskStatus consistency
map (``outcome_allows_status``).

This module is intentionally PURE: it imports only ``shared.task_statuses``
and defines no I/O, no enforcement wiring, and no orchestrator/fused-memory
imports. Enforcement (threading ``agent_id`` through every write call site,
consulting this table in log-mode) is owned by a separate task (rho1b);
``WorkflowStateMachine``'s use of ``outcome_allows_status`` is owned by W9.

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (``from shared.task_transitions
import ...``), consistent with the ``task_statuses``/``mcp_envelope``/
``neutral_cwd``/``config_dir`` sub-module convention.
"""

from __future__ import annotations

import enum
from typing import cast

from shared.task_statuses import TERMINAL, TaskStatus

__all__ = [
    'ActorClass',
    'derive_actor_class',
    'TRANSITIONS',
    'is_legal_transition',
    'outcome_allows_status',
]


class ActorClass(enum.StrEnum):
    """The taxonomy of write actors recognized by the transition authority.

    Members are genuine ``str`` instances (``enum.StrEnum``), matching the
    ``TaskStatus`` idiom in ``shared.task_statuses``.
    """

    ORCHESTRATOR = 'orchestrator'
    RECONCILIATION = 'reconciliation'
    ESCALATION = 'escalation'
    DETERMINISTIC = 'deterministic'
    HUMAN = 'human'


def derive_actor_class(agent_id: str | None) -> ActorClass:
    """Classify a write's ``agent_id`` into its :class:`ActorClass` (D5).

    Ordering is CRITICAL — later rules would otherwise be shadowed by an
    earlier, broader prefix:

    1. ``None`` (header-less write; orchestrator pipeline/scheduler/harness
       callbacks/deterministic_runner/crash-recovery all write without a
       per-write ``agent_id`` today) -> HUMAN, the safe-open default (D5).
    2. ``recon-stage-*`` (live prefix, verified at
       fused-memory/reconciliation/stages/task_knowledge_sync.py:2787) or the
       defensive doc-convention variant ``reconciliation-stage*`` ->
       RECONCILIATION.
    3. ``orchestrator-deterministic*`` (``DETERMINISTIC_AGENT_ROLE``, verified
       at orchestrator/deterministic_runner.py:171) -> DETERMINISTIC. This
       MUST be checked before rule 4 — it is a more specific prefix of the
       plain ``orchestrator*`` rule.
    4. ``orchestrator*`` / ``harness*`` / ``steward*`` -> ORCHESTRATOR.
    5. ``escalation*`` -> ESCALATION.
    6. Anything else (e.g. a ``claude-task-*`` interactive/agent session) ->
       HUMAN, the safe-open default (D5).
    """
    if agent_id is None:
        return ActorClass.HUMAN
    if agent_id.startswith(('recon-stage-', 'reconciliation-stage')):
        return ActorClass.RECONCILIATION
    if agent_id.startswith('orchestrator-deterministic'):
        return ActorClass.DETERMINISTIC
    if agent_id.startswith(('orchestrator', 'harness', 'steward')):
        return ActorClass.ORCHESTRATOR
    if agent_id.startswith('escalation'):
        return ActorClass.ESCALATION
    return ActorClass.HUMAN


# ---------------------------------------------------------------------------
# TRANSITIONS — the enumerated (from, to) legality union, derived by
# enumerating every live status-write call site across orchestrator/ and
# fused-memory/ (first derived 2026-07-06 for task 2168; every anchor below
# re-verified against the tree and re-cited as path::symbol by task 3542).
# Each pair carries its call-site anchors as an inline comment, grouped by
# kind. A bare file name is under orchestrator/src/orchestrator/; recon's
# targeted.py is fused-memory/src/fused_memory/reconciliation/targeted.py.
# This is the LOAD-BEARING artifact — is_legal_transition
# and every downstream consumer (rho1b's interceptor enforcement, W9's
# WorkflowStateMachine) trust this set completely, so it is derived, not
# guessed.
#
# Only ONE actor restriction exists (task-1655, D5): RECONCILIATION may not
# transition FROM in-progress. That subset is added by impl-actor-restriction
# (the next TDD pair) — TRANSITIONS intentionally has no RECONCILIATION entry
# yet, so it currently falls through to the safe-open `_UNION` default in
# is_legal_transition below.
# ---------------------------------------------------------------------------

_UNION: frozenset[tuple[TaskStatus, TaskStatus]] = frozenset(
    {
        # dispatch
        (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),  # workflow.py::TaskWorkflow._setup_worktree_and_artifacts
        # completion
        (
            TaskStatus.IN_PROGRESS,
            TaskStatus.DONE,
        ),  # merged workflow.py::TaskWorkflow._finalise_merged_done; already-merged recovery workflow.py::TaskWorkflow._finalise_recovery_done; found_on_main workflow.py::TaskWorkflow._handle_already_done_report / _on_architect_merge_done, harness.py::Harness._mark_in_progress_done
        (
            TaskStatus.MERGE_DEFERRED,
            TaskStatus.DONE,
        ),  # train workflow.py::TaskWorkflow._maybe_enqueue_group_merge (its _mark_member_done closure) / _attribute_train_failure, harness.py::build_train_callback_factory (mark_member_done / redrive_member)
        (TaskStatus.BLOCKED, TaskStatus.DONE),  # train attribution workflow.py::TaskWorkflow._attribute_train_failure
        (
            TaskStatus.PENDING,
            TaskStatus.DONE,
        ),  # recon found_on_main; operator direct-complete of a never-dispatched task is an out-of-band manual set_task_status/update_task write with no enumerated call site (unlike the anchored half of this pair)
        (
            TaskStatus.DEFERRED,
            TaskStatus.DONE,
        ),  # operator/vehicle direct-complete of a deferred task whose deliverable landed out-of-band (planning-mode/merge-vehicle task landed via merge queue); manual set_task_status write, no enumerated orchestrator call site — mirrors the (PENDING, DONE) found-on-main rationale above
        # park
        (
            TaskStatus.IN_PROGRESS,
            TaskStatus.MERGE_DEFERRED,
        ),  # workflow.py::TaskWorkflow._enter_merge_deferred / _handle_superseded
        # requeue
        (
            TaskStatus.IN_PROGRESS,
            TaskStatus.PENDING,
        ),  # workflow.py::TaskWorkflow._repend_for_requeue (callers include _requeue_on_server_error, task 3316) / _plan / _handle_blocking_dep_report; blast-radius scheduler.py::Scheduler.handle_blast_radius_expansion; stranded-revert harness.py::Harness._revert_in_progress_if_no_live_claimant; Table B restart harness.py::Harness._action_teardown_and_set_status
        (
            TaskStatus.BLOCKED,
            TaskStatus.PENDING,
        ),  # steward re-pend workflow.py::TaskWorkflow._mark_blocked (its _requeue closure); escalation resume harness.py::Harness._cascade_unblock_member; stranded-blocked redispatch scheduler.py::Scheduler._phase_redispatch_stranded_blocked; recon dependency unblock targeted.py::TargetedReconciler._unblock_dependent
        (
            TaskStatus.MERGE_DEFERRED,
            TaskStatus.PENDING,
        ),  # train re-drive harness.py::build_train_callback_factory (redrive_member / _revert_withheld_member), workflow.py::TaskWorkflow._revert_withheld_member
        (
            TaskStatus.DEFERRED,
            TaskStatus.PENDING,
        ),  # planning-mode commit fused-memory/src/fused_memory/server/tools.py::create_mcp_server.commit_planning
        # block
        (
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
        ),  # workflow.py::TaskWorkflow._mark_blocked via _persist_blocked_row; CONVERT_TO_BLOCKED harness.py::Harness._reconcile_one_stranded; Table B park harness.py::Harness._action_teardown_and_set_status and escalation/src/escalation/server.py::release_workflow
        (TaskStatus.MERGE_DEFERRED, TaskStatus.BLOCKED),  # train failer workflow.py::TaskWorkflow._attribute_train_failure
        (TaskStatus.DEFERRED, TaskStatus.BLOCKED),  # recon targeted.py::TargetedReconciler._sweep_block_orphan
        (
            TaskStatus.PENDING,
            TaskStatus.BLOCKED,
        ),  # born-at-L2 deterministic_runner.py::DeterministicRunner's *_and_block helpers (the pure gate's _file_milestone_gate_and_block among them); dispatch gates harness.py::Harness._block_and_escalate_external_dep / _delivered_check / _cross_repo / _substrate_flip (all before TaskWorkflow's in-progress claim); retry-cap scheduler.py::Scheduler.trigger_retry_cap_exhausted (after a REQUEUED exit re-pended the row); human block of a pending task is an out-of-band manual write with no enumerated call site
        # cancel — recon's any-non-terminal->cancelled
        # (targeted.py::TargetedReconciler._sweep_cancel_orphan) covers
        # pending/blocked/deferred/merge-deferred; Table B abandon
        # (harness.py::Harness._action_teardown_and_set_status) covers the
        # orchestrator path from in-progress/blocked.
        (TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED),  # abandon harness.py::Harness._action_teardown_and_set_status
        (TaskStatus.PENDING, TaskStatus.CANCELLED),  # recon targeted.py::TargetedReconciler._sweep_cancel_orphan
        (
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        ),  # abandon harness.py::Harness._action_teardown_and_set_status; recon targeted.py::TargetedReconciler._sweep_cancel_orphan
        (TaskStatus.DEFERRED, TaskStatus.CANCELLED),  # recon targeted.py::TargetedReconciler._sweep_cancel_orphan
        # W9-θ: a cancel (hard task.cancel() or soft _cancel_event) can land
        # on a train member parked in merge-deferred — it awaits
        # _await_cancellable(future) inside
        # workflow.py::TaskWorkflow._maybe_enqueue_group_merge, after
        # workflow.py::TaskWorkflow._enter_merge_deferred has persisted the
        # merge-deferred row. workflow.py::TaskWorkflow._finalise_cancellation
        # then drives the machine to CANCELLED (WorkflowStateMachine.transition
        # consults this table), so the merge-deferred origin needs its own
        # cancel edge — completing the "any-non-terminal->cancelled" family the
        # comment above describes. Also the persisted-write sibling of recon's
        # any-non-terminal->cancelled sweep, which already covers
        # merge-deferred rows.
        (
            TaskStatus.MERGE_DEFERRED,
            TaskStatus.CANCELLED,
        ),  # W9-θ cancel workflow.py::TaskWorkflow._finalise_cancellation; recon targeted.py::TargetedReconciler._sweep_cancel_orphan
        # (blocked, in-progress) has NO orchestrator status writer since task
        # 3538 (γ3): the infra resume re-pends instead
        # (harness.py::Harness._cascade_unblock_member). The pair stays because
        # WorkflowStateMachine's BLOCKED/ESCALATED -> working-phase moves
        # project onto it through
        # orchestrator/src/orchestrator/workflow_types.py::STATE_TO_STATUS.
        (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
        # infra-hold (D3). in-progress -> infra-hold is
        # workflow.py::TaskWorkflow._mark_blocked(block_status='infra-hold'),
        # from _execute_verify_review_loop's verify-infra stamp; infra-hold ->
        # pending is the infra resume in
        # harness.py::Harness._cascade_unblock_member. The other infra-hold
        # edges are forward-compat, with no enumerated writer.
        (TaskStatus.IN_PROGRESS, TaskStatus.INFRA_HOLD),
        (TaskStatus.BLOCKED, TaskStatus.INFRA_HOLD),
        (TaskStatus.INFRA_HOLD, TaskStatus.IN_PROGRESS),
        (TaskStatus.INFRA_HOLD, TaskStatus.PENDING),
        (TaskStatus.INFRA_HOLD, TaskStatus.DONE),
        (TaskStatus.INFRA_HOLD, TaskStatus.BLOCKED),
        (TaskStatus.INFRA_HOLD, TaskStatus.CANCELLED),
        # soak-validated human/agent-driven review + deferred edges (D6).
        (TaskStatus.IN_PROGRESS, TaskStatus.REVIEW),
        (TaskStatus.REVIEW, TaskStatus.IN_PROGRESS),
        (TaskStatus.REVIEW, TaskStatus.PENDING),
        (TaskStatus.REVIEW, TaskStatus.DONE),
        (TaskStatus.REVIEW, TaskStatus.BLOCKED),
        (TaskStatus.REVIEW, TaskStatus.CANCELLED),
        (TaskStatus.PENDING, TaskStatus.DEFERRED),
        (TaskStatus.IN_PROGRESS, TaskStatus.DEFERRED),
        (TaskStatus.BLOCKED, TaskStatus.DEFERRED),
    }
)

# Per-actor transition table. ORCHESTRATOR/ESCALATION/DETERMINISTIC/HUMAN all
# get the full union (safe-open default, D5).
TRANSITIONS: dict[ActorClass, frozenset[tuple[TaskStatus, TaskStatus]]] = {
    ActorClass.ORCHESTRATOR: _UNION,
    ActorClass.ESCALATION: _UNION,
    ActorClass.DETERMINISTIC: _UNION,
    ActorClass.HUMAN: _UNION,
    # The ONLY actor-specific restriction (task-1655, D5): reconciliation
    # must never mutate a live/claimed in-progress task. The
    # live_workflow_status_write guard (task_knowledge_sync.py:521/2996)
    # actively blocks recon status writes on an in-progress task at
    # runtime — this is that restriction's pure-table mirror: the union
    # minus every (in-progress, *) pair. Finer actor granularity beyond
    # this one rule is deferred to the production soak (Open Question 4).
    ActorClass.RECONCILIATION: frozenset(p for p in _UNION if p[0] != TaskStatus.IN_PROGRESS),
}


def is_legal_transition(
    frm: TaskStatus | str,
    to: TaskStatus | str,
    actor: ActorClass | str,
    *,
    reopen: bool = False,
) -> bool:
    """Is the ``frm -> to`` status write legal for ``actor``? (Table A, C1/D1)

    Evaluation order:

    1. Coerce ``frm``/``to`` via ``TaskStatus(...)`` — raises ``ValueError``
       on an out-of-vocabulary status. (Rejecting bad vocabulary is this
       function's job; a *dedicated* vocabulary gate is owned separately by
       tau1/rho1a.)
    2. Same-status is always legal (a no-op write) — mirrors the
       interceptor's ``status == old_status`` early-return
       (task_interceptor.py:645).
    3. A transition FROM a terminal status (``done``/``cancelled``) is legal
       only when ``reopen`` is true — mirrors the terminal-exit gate
       (task_interceptor.py:702), which requires a non-empty
       ``reopen_reason`` to leave a terminal status.
    4. Otherwise, legality is a membership test against this actor's
       transition set, defaulting to the full ``_UNION`` for any actor with
       no entry in ``TRANSITIONS`` (safe-open, D5) — an unattributed or
       unrecognized actor is never blocked by an actor-specific restriction,
       only by an actually-illegal ``(frm, to)`` pair or an unknown status.
    """
    frm = TaskStatus(frm)
    to = TaskStatus(to)
    if frm == to:
        return True
    if frm in TERMINAL:
        return bool(reopen)
    # Unlike frm/to, an unrecognized actor is deliberately NOT coerced via
    # ActorClass(actor) (which would raise ValueError) — safe-open (D5) means
    # an unattributed/unrecognized actor must fall back to _UNION, not be
    # rejected. ActorClass is a StrEnum, so TRANSITIONS.get(...) matches a raw
    # string actor against the real ActorClass keys by value (e.g. "human" ==
    # ActorClass.HUMAN) and otherwise falls through to the _UNION default;
    # the cast only satisfies the type checker; it changes no runtime lookup.
    return (frm, to) in TRANSITIONS.get(cast(ActorClass, actor), _UNION)


# ---------------------------------------------------------------------------
# outcome_allows_status — the WorkflowOutcome -> TaskStatus consistency map.
# Keys MIRROR the 8 WorkflowOutcome string values
# (orchestrator/src/orchestrator/workflow_types.py::WorkflowOutcome) as inline
# literals, since shared/ must NOT import orchestrator (layering).
#
# test-outcome's test_recognized_outcome_keys_match_mirrored_workflow_outcomes
# only pins these _OUTCOME_ALLOWED keys against that test's own
# _WORKFLOW_OUTCOME_VALUES copy — both are hand-maintained mirrors of the real
# orchestrator.workflow_types.WorkflowOutcome enum, so that assertion catches
# module-vs-test drift between the two copies, NOT drift from the actual
# source of truth (shared/ cannot import orchestrator to check that
# directly). The genuine cross-layer guard lives in orchestrator/tests:
# orchestrator/tests/test_workflow_state_machine.py::TestExitContractCoversEveryOutcome.
#
# Each row is exactly the spec §5 exit contract
# (docs/task-escalation-state-spec.md): the status rows the outcome's proven
# producers leave at a run() exit (task 3542, divergence E10). Producers named
# bare below are methods of
# orchestrator/src/orchestrator/workflow.py::TaskWorkflow. Relaxation 1 (a
# row already terminal is reported AS that terminal) is producer-side, in
# TaskWorkflow._observed_terminal_outcome. Relaxation 2 (the exit's own status
# write failed) is applied by the consumer,
# orchestrator/src/orchestrator/exit_contract.py::judge_exit, not by this
# pure table.
# ---------------------------------------------------------------------------

_OUTCOME_ALLOWED: dict[str, frozenset[TaskStatus]] = {
    'done': frozenset({TaskStatus.DONE}),
    # An internal sub-phase outcome consumed inside _drive, never a run()
    # exit. The key stays so a PLANNED exit is a named violation rather than
    # an unknown outcome.
    'planned': frozenset(),
    # _mark_blocked writes 'blocked' / 'infra-hold', or a steward terminal
    # decision (WORKFLOW_PRESERVE minus done) is preserved through
    # _honour_steward_terminal_decision and _mark_blocked's
    # StewardTerminalDecision branch.
    'blocked': frozenset(
        {
            TaskStatus.BLOCKED,
            TaskStatus.INFRA_HOLD,
            TaskStatus.CANCELLED,
            TaskStatus.DEFERRED,
            TaskStatus.MERGE_DEFERRED,
        }
    ),
    # _mark_blocked writes the row (its entry gate, or _park_merge_phase_row)
    # before its StewardReescalatedL1 exit. The sole block_status='infra-hold'
    # caller passes escalate_to_human=True, so it always exits BLOCKED.
    'escalated': frozenset({TaskStatus.BLOCKED}),
    # Every slot-exiting writer re-pends first: _repend_for_requeue (for the
    # WarmLaneRequeue clause, _handle_soft_cancel's fallback and
    # _requeue_on_server_error), _mark_blocked's _requeue, _plan,
    # _handle_blocking_dep_report, and
    # orchestrator/src/orchestrator/scheduler.py::Scheduler.handle_blast_radius_expansion.
    'requeued': frozenset({TaskStatus.PENDING}),
    'cancelled': frozenset({TaskStatus.CANCELLED}),
    'merge-deferred': frozenset({TaskStatus.MERGE_DEFERRED}),
    # escalation/src/escalation/server.py::release_workflow parks
    # 'in-progress' -> 'blocked' only after the slot clears;
    # orchestrator/src/orchestrator/harness.py::Harness._action_teardown_and_set_status
    # writes the park / restart row before the kill; a parked train member
    # stays 'merge-deferred'. Keeping IN_PROGRESS / MERGE_DEFERRED is a
    # reviewed divergence from spec §5's "never wherever it was": the park is
    # written after the exit, not before it (still open in the spec's §8-E10).
    'soft-cancelled': frozenset(
        {
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.PENDING,
            TaskStatus.MERGE_DEFERRED,
        }
    ),
}


def outcome_allows_status(outcome: object, status: TaskStatus | str) -> bool:
    """Is ``status`` a consistent run()-exit row for ``outcome``?

    ``outcome`` may be a plain string or any object exposing a ``.value``
    attribute (e.g. a ``WorkflowOutcome`` instance) — normalized via
    ``getattr(outcome, 'value', outcome)`` so callers never need to unwrap
    it themselves. Raises ``ValueError`` loudly on an unrecognized outcome
    or an out-of-vocabulary status (no silent fallthrough, matching the
    repo's loud-escalation norm).
    """
    key = str(getattr(outcome, 'value', outcome))
    if key not in _OUTCOME_ALLOWED:
        raise ValueError(f'unknown workflow outcome: {outcome!r}')
    return TaskStatus(status) in _OUTCOME_ALLOWED[key]
