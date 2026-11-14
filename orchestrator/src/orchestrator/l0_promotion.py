"""Promote a level-0 escalation to level 1 without losing its finding.

The L0's finding is copied verbatim into a new level-1 record, which is
written first; only then is the L0 dismissed, with a resolution naming the
L1 that now carries it. The two records cite each other, and a failure
between the writes leaves the L0 pending rather than the finding nowhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from escalation.models import Escalation

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue


def promote_l0_to_l1(
    queue: EscalationQueue,
    l0: Escalation,
    *,
    agent_role: str,
    why: str,
    detail_suffix: str,
    filing_claimant_run_id: str | None,
    resolved_by: str,
) -> Escalation:
    if l0.level != 0 or l0.status != 'pending':
        raise ValueError(
            'only a pending level-0 escalation can be promoted: '
            f'{l0.id} has level={l0.level} status={l0.status!r}'
        )
    l1 = Escalation(
        id=queue.make_id(l0.task_id),
        task_id=l0.task_id,
        agent_role=agent_role,
        severity=l0.severity,
        category=l0.category,
        summary=l0.summary,
        detail=f'{l0.detail}\n\n[promoted from {l0.id}: {why}]{detail_suffix}',
        suggested_action=l0.suggested_action or 'manual_intervention',
        worktree=None,
        workflow_state=l0.workflow_state,
        level=1,
        options=list(l0.options),
        evidence=list(l0.evidence),
        root_cause=l0.root_cause,
        project_id=l0.project_id,
        filing_claimant_run_id=filing_claimant_run_id,
    )
    queue.submit(l1)
    queue.resolve(
        l0.id,
        f'Auto-promoted to level 1 as {l1.id} — {why}',
        dismiss=True,
        resolved_by=resolved_by,
    )
    return l1
