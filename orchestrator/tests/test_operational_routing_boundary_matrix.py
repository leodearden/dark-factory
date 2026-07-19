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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import OPERATIONAL_LLM_NEEDS_LANE_TOKEN

# NOTE (review follow-up, task 2806 amendment pass): _gate_task /
# _llm_gate_task / _make_assignment / _mock_scheduler below are
# intentionally duplicated from test_deterministic_runner.py rather than
# imported from one shared conftest/test-support module, so the two copies
# could in principle drift if the pure-gate metadata shape or the
# TaskAssignment/mock-scheduler contract changes and only one copy is
# updated. Extracting a shared module is the right fix, but this task's
# lock scope covers only this file and
# fused-memory/tests/test_operational_routing_boundary_matrix.py —
# test_deterministic_runner.py is out of scope for this pass. Flagged as a
# follow-up: extract these four helpers into a shared test-support module
# imported by both files so the pure-gate shape is defined exactly once.


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


# ---------------------------------------------------------------------------
# Scenario 4b — llm escalation + never-routed-to-architect (matrix row 4,
# consumer half; orchestrator side). Two standalone functions (not a shared
# test class) — one sync, one async — so no class-level @pytest.mark.asyncio
# is needed (this project's pyproject.toml does NOT set asyncio_mode="auto";
# a class-level mark over a MIXED sync/async class is a collection-time
# ERROR here, per test_background_service.py / test_merge_queue_resource_audit.py).
# ---------------------------------------------------------------------------


def test_llm_gate_task_is_deterministic_never_routed_to_architect():
    """(a) Never-to-architect: an llm-marked pure-gate task is routed by
    ``Scheduler.is_deterministic`` — the predicate (scheduler.py:1996-2012)
    that sends a task to ``_run_deterministic_slot``, never the architect.
    ``is_deterministic`` reads only ``metadata['task_kind']``, which the
    llm-gate task shares with a plain pure gate (β's marker is an
    additional, independent key) — so this also confirms the llm-gate
    shape does not accidentally opt back into the architect path.
    """
    from orchestrator.scheduler import Scheduler

    task = _llm_gate_task()
    assert Scheduler.is_deterministic(task) is True


@pytest.mark.asyncio
async def test_llm_gate_escalation_carries_token_via_real_escalation_queue(
    tmp_path: Path,
):
    """(b) Escalation read path: ``DeterministicRunner.run`` over the
    llm-marked task files a born-at-L2 escalation, read back through a REAL
    ``EscalationQueue.get_by_task`` (not a mock), whose detail carries
    ``OPERATIONAL_LLM_NEEDS_LANE_TOKEN``, ``level==2``,
    ``agent_role=='orchestrator-deterministic'``, and the scheduler was set
    to ``'blocked'``. Ties β's marker (fused-memory scenario 4a,
    ``x_operational_llm_gate``) to γ's token emission through the real
    escalation API — the ζ integration matrix's distinct value over the γ
    leaf test (test_deterministic_runner.py::TestOperationalLlmGateMarker),
    which asserts the same token but not the never-to-architect routing half
    proven alongside it here.
    """
    from orchestrator.deterministic_runner import DeterministicRunner

    task = _llm_gate_task(task_id='99')
    assignment = _make_assignment(task)
    queue = EscalationQueue(tmp_path)
    scheduler = _mock_scheduler(task)

    runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
    await runner.run(assignment)

    pending = queue.get_by_task('99', status='pending')
    assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'
    esc = pending[0]

    assert OPERATIONAL_LLM_NEEDS_LANE_TOKEN in esc.detail, f'got {esc.detail!r}'
    assert esc.level == 2, f'got {esc.level!r}'
    assert esc.agent_role == 'orchestrator-deterministic', f'got {esc.agent_role!r}'

    scheduler.set_task_status.assert_awaited_once_with('99', 'blocked')
