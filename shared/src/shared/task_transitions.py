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
    "ActorClass",
    "derive_actor_class",
    "TRANSITIONS",
    "is_legal_transition",
    "outcome_allows_status",
]


class ActorClass(enum.StrEnum):
    """The taxonomy of write actors recognized by the transition authority.

    Members are genuine ``str`` instances (``enum.StrEnum``), matching the
    ``TaskStatus`` idiom in ``shared.task_statuses``.
    """

    ORCHESTRATOR = "orchestrator"
    RECONCILIATION = "reconciliation"
    ESCALATION = "escalation"
    DETERMINISTIC = "deterministic"
    HUMAN = "human"


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
    if agent_id.startswith(("recon-stage-", "reconciliation-stage")):
        return ActorClass.RECONCILIATION
    if agent_id.startswith("orchestrator-deterministic"):
        return ActorClass.DETERMINISTIC
    if agent_id.startswith(("orchestrator", "harness", "steward")):
        return ActorClass.ORCHESTRATOR
    if agent_id.startswith("escalation"):
        return ActorClass.ESCALATION
    return ActorClass.HUMAN


# ---------------------------------------------------------------------------
# TRANSITIONS — the enumerated (from, to) legality union, derived by
# enumerating every live status-write call site (grep across orchestrator/ +
# fused-memory/ on 2026-07-06; see the task-2168 plan analysis for the full
# derivation). Each pair carries its call-site anchor as an inline comment,
# grouped by kind. This is the LOAD-BEARING artifact — is_legal_transition
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
        (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),  # workflow.py:1510
        # completion
        (TaskStatus.IN_PROGRESS, TaskStatus.DONE),  # merged workflow.py:1380; found_on_main workflow.py:3809/7396/harness.py:3623
        (TaskStatus.MERGE_DEFERRED, TaskStatus.DONE),  # workflow.py:1015/6424, harness.py:619/646
        (TaskStatus.BLOCKED, TaskStatus.DONE),  # train attribution workflow.py:6424
        # park
        (TaskStatus.IN_PROGRESS, TaskStatus.MERGE_DEFERRED),  # workflow.py:867/896
        # requeue
        (TaskStatus.IN_PROGRESS, TaskStatus.PENDING),  # workflow.py:2464/3755, blast-radius scheduler.py:4484, stranded-revert harness.py:3487/3567
        (TaskStatus.BLOCKED, TaskStatus.PENDING),  # steward re-pend workflow.py:8007, escalation resume harness.py:8739
        (TaskStatus.MERGE_DEFERRED, TaskStatus.PENDING),  # re-drive harness.py:682
        (TaskStatus.DEFERRED, TaskStatus.PENDING),  # stage2 commit_planning
        # block
        (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),  # _mark_blocked workflow.py:7758, retry-cap scheduler.py:4659, substrate harness.py:4687, dep harness.py:3905
        (TaskStatus.MERGE_DEFERRED, TaskStatus.BLOCKED),  # train failer workflow.py:6464
        (TaskStatus.DEFERRED, TaskStatus.BLOCKED),  # recon targeted.py:1041
        # cancel — recon's any-non-terminal->cancelled (targeted.py:1001)
        # covers pending/blocked/deferred; harness.py:8417 covers the
        # orchestrator abandon path from in-progress/blocked.
        (TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED),  # abandon harness.py:8417
        (TaskStatus.PENDING, TaskStatus.CANCELLED),  # recon targeted.py:1001
        (TaskStatus.BLOCKED, TaskStatus.CANCELLED),  # harness.py:8417, recon targeted.py:1001
        (TaskStatus.DEFERRED, TaskStatus.CANCELLED),  # recon targeted.py:1001
        # infra resume-at-verify (D3 migrates this to blocked->infra-hold)
        (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),  # harness.py:8713
        # forward-compat D3 infra-hold edges (C7/workflow.py:4842) — live
        # write sites still use in-progress + metadata.infra_hold today;
        # omega4 migrates them before the Gamma enforcement gate flips.
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
# Keys MIRROR the 8 WorkflowOutcome string values (orchestrator/workflow.py:
# 337-345) as inline literals, since shared/ must NOT import orchestrator
# (layering).
#
# test-outcome's test_recognized_outcome_keys_match_mirrored_workflow_outcomes
# only pins these _OUTCOME_ALLOWED keys against that test's own
# _WORKFLOW_OUTCOME_VALUES copy — both are hand-maintained mirrors of the real
# orchestrator.workflow.WorkflowOutcome enum, so that assertion catches
# module-vs-test drift between the two copies, NOT drift from the actual
# source of truth (shared/ cannot import orchestrator to check that
# directly). A genuine cross-layer guard would need to live in
# orchestrator/tests, asserting against the real WorkflowOutcome enum.
#
# Several rows are intentionally loose sets rather than a single dominant
# status: W9's SM-2 invariant checks outcome_allows_status(report.outcome,
# last_status_row) holds at EVERY run() exit, and the traced code shows each
# of these outcomes can legitimately coexist with more than one status row
# at exit.
# ---------------------------------------------------------------------------

_OUTCOME_ALLOWED: dict[str, frozenset[TaskStatus]] = {
    "done": frozenset({TaskStatus.DONE}),
    # Internal-only sub-phase — the task stays claimed (in-progress).
    "planned": frozenset({TaskStatus.IN_PROGRESS}),
    "blocked": frozenset({TaskStatus.BLOCKED}),
    # Self-repend / deferred-to-stranded-sweep window / requeue-cap
    # exhausted -> blocked.
    "requeued": frozenset({TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED}),
    # Dominant _mark_blocked + open L1 / merge-gating bail preserves
    # in-progress.
    "escalated": frozenset({TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS}),
    "cancelled": frozenset({TaskStatus.CANCELLED}),
    "merge-deferred": frozenset({TaskStatus.MERGE_DEFERRED}),
    # preserved / release_workflow park->blocked / stranded-sweep->pending.
    "soft-cancelled": frozenset({TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.PENDING}),
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
    key = str(getattr(outcome, "value", outcome))
    if key not in _OUTCOME_ALLOWED:
        raise ValueError(f"unknown workflow outcome: {outcome!r}")
    return TaskStatus(status) in _OUTCOME_ALLOWED[key]
