"""Info-L0 escalation records shared by the classify-level and route-level tests.

One copy of each filing shape, so a change to a discriminator's shape reaches
every test that classifies or routes it.
"""

from __future__ import annotations

from typing import Any

from escalation.models import Escalation

NOTE_DETAIL = (
    'Helper X in orchestrator/src/orchestrator/foo.py::helper_x has no test\n'
    'covering its empty-input branch; a regression there would go unnoticed.'
)


def info_l0_note(**overrides: Any) -> Escalation:
    """An agent-filed info L0 that carries follow-up work; *overrides* replace fields."""
    fields: dict[str, Any] = {
        'id': 'esc-77-3',
        'task_id': '77',
        'agent_role': 'implementer',
        'severity': 'info',
        'level': 0,
        'category': 'design_concern',
        'summary': 'Helper X lacks a regression test',
        'detail': NOTE_DETAIL,
        'suggested_action': 'Add a regression test for helper X',
    }
    fields.update(overrides)
    return Escalation(**fields)


def done_step_tripwire() -> Escalation:
    """The done-step tripwire, as filed by
    orchestrator/src/orchestrator/workflow.py::TaskWorkflow._escalate_unreconciled_done_step."""
    return info_l0_note(
        agent_role='orchestrator',
        category='infra_issue',
        summary='Done step commit not reachable from HEAD',
        suggested_action='verify_wip_reconciliation',
    )


def info_scope_divergence() -> Escalation:
    """The scope-divergence shape at info severity; its filer,
    orchestrator/src/orchestrator/workflow.py::TaskWorkflow._escalate_scope_invariant_violation,
    files it as blocking."""
    return info_l0_note(
        agent_role='orchestrator',
        category='infra_issue',
        summary='plan.files/metadata.files divergence detected for task 77',
        suggested_action='investigate_and_retry',
    )
