"""Server-side gate rejecting recon-stage writes to tasks they must not touch.

W5-ζ (task 2224). Consulted from
:class:`fused_memory.middleware.task_interceptor.TaskInterceptor` ONLY when
the caller's ``agent_id`` starts with ``'recon-stage-'`` — enforcing this
gate server-side means the post-hoc reconciliation guards
(``reconciliation/stages/task_knowledge_sync.py``'s
``_check_stall_guard_freshness`` and related) become redundant; deleting
them is downstream consumer W5-μ's job, NOT this module's.

Three independent early-return gates in :func:`check`:

1. ``op == 'update_task'`` AND ``live_status`` is terminal (done/cancelled)
   -> ``ReconTerminalWriteRejected``.
2. ``op == 'set_task_status'`` AND a live workflow is detected for the task
   -> ``ReconLiveWorkflowWriteRejected``.
3. ``snapshot_token is not None`` AND it disagrees with ``live_status``
   (op-agnostic) -> ``ReconStaleSnapshotRejected``.

Design note: this module is a SEPARATE gate from
``shared.task_transitions.is_legal_transition`` (the (from, to, actor)
transition-legality table) — never a third transition table. That table
answers "is old->new legal for this actor class"; this module answers "may
this recon-stage caller touch this task at all". The two compose
independently; this module never imports/extends ``shared.task_transitions``.

The recon-stage scoping (``agent_id.startswith('recon-stage-')``) lives at
the call site in ``task_interceptor.py``, not here — :func:`check` is a pure
function that always runs its gates when called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """Outcome of a :func:`check` call.

    Mirrors :class:`fused_memory.middleware.path_scope_guard.PathGuardVerdict`'s
    shape (``is_rejection`` property; ``to_error_dict()`` -> ``{}`` on ok /
    structured dict on rejection) so callers branch on ``error_type``
    uniformly across guards.

    Fields:
        outcome: ``'ok'`` or ``'rejection'``.
        op: The operation checked (``'update_task'`` or ``'set_task_status'``).
        task_id: The task identifier checked.
        agent_id: The caller's agent_id.
        error_type: Stable error-type string for callers to branch on
            (``'ReconTerminalWriteRejected'`` / ``'ReconLiveWorkflowWriteRejected'``
            / ``'ReconStaleSnapshotRejected'``). Empty on ok.
        reason: Human-readable rejection message. Empty on ok.
        hint: Actionable follow-up guidance. Empty on ok.
        live_status: The task's live status as observed by the caller.
        target_status: The status the caller was attempting to set
            (``set_task_status`` only; ``None`` for ``update_task``).
        snapshot_token: The snapshot status token the caller's write was
            based on, when supplied.
    """

    outcome: Literal['ok', 'rejection']
    op: str = ''
    task_id: str = ''
    agent_id: str = ''
    error_type: str = ''
    reason: str = ''
    hint: str = ''
    live_status: str = ''
    target_status: str | None = None
    snapshot_token: str | None = None

    @property
    def is_rejection(self) -> bool:
        return self.outcome == 'rejection'

    def to_error_dict(self) -> dict:
        """Return a structured MCP error dict, or ``{}`` on ok."""
        if not self.is_rejection:
            return {}
        return {
            'error': self.reason,
            'error_type': self.error_type,
            'agent_id': self.agent_id,
            'task_id': self.task_id,
            'op': self.op,
            'live_status': self.live_status,
            'target_status': self.target_status,
            'snapshot_token': self.snapshot_token,
            'hint': self.hint,
        }


def _reject(
    *,
    op: str,
    task_id: str,
    agent_id: str,
    error_type: str,
    reason: str,
    hint: str,
    live_status: str,
    target_status: str | None = None,
    snapshot_token: str | None = None,
) -> Verdict:
    """Small internal factory for a rejection :class:`Verdict`."""
    return Verdict(
        outcome='rejection',
        op=op,
        task_id=task_id,
        agent_id=agent_id,
        error_type=error_type,
        reason=reason,
        hint=hint,
        live_status=live_status,
        target_status=target_status,
        snapshot_token=snapshot_token,
    )
