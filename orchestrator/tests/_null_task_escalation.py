"""The pending reconciliation-backlog escalation that crashed startup step 2c3 on 2026-11-10.

``fused_memory.reconciliation.backlog_policy.BacklogPolicy`` files its whole
family (backlog, judge_halt, wedge) with ``task_id`` null.
"""

from __future__ import annotations

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def submit_null_task_escalation(
    queue: EscalationQueue, *, agent_role: str = 'fused-memory',
) -> Escalation:
    """Submit the 2026-11-10 trigger record, filed as *agent_role*, to *queue* and return it."""
    record = Escalation.from_dict({
        'id': 'esc-reconciliation-backlog-dark_factory-2026-11-10T225816_4098180000',
        'task_id': None,
        'agent_role': agent_role,
        'severity': 'blocking',
        'category': 'infra_issue',
        'summary': 'Reconciliation backlog over threshold for dark_factory',
        'level': 1,
        'workflow_state': 'infra',
        'project_id': 'dark_factory',
    })
    queue.submit(record)
    return record
