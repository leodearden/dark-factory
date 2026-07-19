"""ζ boundary-integration-gate matrix — operational-routing two-way boundary
(PRD ``plans/operational-ask-routing-prd.md``, task ζ/2806), orchestrator half.

This module realizes matrix row 4b — the CONSUMER half of the
``operational_mode='llm'`` pure gate (the producer half, matrix row 4a, lives
in ``fused-memory/tests/test_operational_routing_boundary_matrix.py``, which
proves the submit boundary stamps the ``x_operational_llm_gate`` marker).
Row 4b proves the orchestrator-side consequence of that marker end-to-end
through real orchestrator collaborators:

(a) the llm-marked pure-gate task is NEVER routed to the architect — it is
    always sent to ``_run_deterministic_slot`` via
    ``Scheduler.is_deterministic`` (γ/2803's routing predicate,
    scheduler.py:1996-2012);
(b) ``DeterministicRunner.run`` over that task files a born-at-L2 escalation
    (read back via a REAL ``EscalationQueue.get_by_task``, not a mock) whose
    detail carries the ``operational_llm_needs_lane`` token (γ/2803).

All production substrate here is already landed on main; a RED result is a
genuine integration gap, not a synthetic-input failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from orchestrator.deterministic_runner import OPERATIONAL_LLM_NEEDS_LANE_TOKEN


def _gate_task(
    task_id: str = '99',
    title: str = 'Ship feature gate',
    description: str = 'Gate that guards the feature launch',
    deps: list | None = None,
    gate_options: list | None = None,
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a deterministic pure-gate task dict.

    Mirrors test_deterministic_runner.py's ``_gate_task`` helper verbatim.
    """
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
    }
    if gate_options is not None:
        metadata['gate_options'] = gate_options
    if gate_escalated_at is not None:
        metadata['gate_escalated_at'] = gate_escalated_at
    task = {
        'id': task_id,
        'title': title,
        'description': description,
        'metadata': metadata,
    }
    if deps is not None:
        task['dependencies'] = deps
    return task


def _llm_gate_task(
    task_id: str = '99',
    title: str = 'Ship feature gate',
    description: str = 'Gate that guards the feature launch',
    deps: list | None = None,
    gate_options: list | None = None,
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a deterministic pure-gate task dict carrying β's llm-gate marker.

    Mirrors ``_gate_task`` verbatim, then additionally stamps
    ``metadata['x_operational_llm_gate'] = True`` and
    ``metadata['operational_mode'] = 'llm'`` — the exact shape β's
    ``inject_operational_routing`` produces for an
    ``execution_class='operational'`` + ``operational_mode='llm'``
    submission (fused_memory.middleware.operational_routing_guard.
    OPERATIONAL_LLM_GATE_MARKER_KEY). Mirrors
    test_deterministic_runner.py:49-76's ``_llm_gate_task``.
    """
    task = _gate_task(
        task_id=task_id,
        title=title,
        description=description,
        deps=deps,
        gate_options=gate_options,
        gate_escalated_at=gate_escalated_at,
    )
    task['metadata']['x_operational_llm_gate'] = True
    task['metadata']['operational_mode'] = 'llm'
    return task


def _make_assignment(task: dict):
    """Build a minimal TaskAssignment-like object for the runner.

    Deterministic tasks hold an empty modules list (I4/B12: no module lock).
    Mirrors test_deterministic_runner.py's ``_make_assignment``.
    """
    from orchestrator.scheduler import TaskAssignment
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


def _mock_scheduler(task: dict):
    """Return a MagicMock scheduler with async set_task_status/update_task/get_task.

    Mirrors test_deterministic_runner.py's ``_mock_scheduler``.
    """
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler
