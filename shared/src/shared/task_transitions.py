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

__all__ = [
    "ActorClass",
    "derive_actor_class",
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
