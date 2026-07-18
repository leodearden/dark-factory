"""Unit + tripwire tests for the shared ``build_workflow()`` factory.

PRD eval-framework-revival §α, Contract C2, Invariant P2, Boundary test B2.

``build_workflow()`` is the SINGLE construction point for ``TaskWorkflow``
across production dispatch (``harness.py``) and eval dispatch
(``evals/runner.py``).  Routing both sites through one factory means a new
MANDATORY ``TaskWorkflow`` constructor parameter is either acquired by both
call sites automatically or breaks BOTH at once — never drifts silently.

The grep-guard tripwire (added in a later step) mechanises that invariant by
source inspection: exactly one production-source file may construct
``TaskWorkflow(`` and both dispatch sites must route through
``build_workflow(``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, build_workflow


def test_build_workflow_forwards_params():
    """``build_workflow(...)`` returns a ``TaskWorkflow`` forwarding every param by identity."""
    assignment = MagicMock()
    assignment.task_id = '2464'
    assignment.task = {'id': '2464', 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0

    git_ops = MagicMock()
    scheduler = MagicMock()
    briefing = MagicMock()
    mcp = MagicMock()

    # Distinct sentinels for the optional pass-through params — assert the
    # factory forwards the *identical* objects, not equal copies.
    cost_store = object()
    event_store = object()
    initial_plan = object()
    usage_gate = object()
    escalation_queue = object()

    wf = build_workflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=mcp,
        escalation_queue=escalation_queue,
        usage_gate=usage_gate,
        initial_plan=initial_plan,
        event_store=event_store,
        cost_store=cost_store,
    )

    assert isinstance(wf, TaskWorkflow)
    # Required params forwarded by identity.
    assert wf.assignment is assignment
    assert wf.config is config
    assert wf.git_ops is git_ops
    assert wf.scheduler is scheduler
    assert wf.briefing is briefing
    assert wf.mcp is mcp
    # Optional pass-through params forwarded by identity.
    assert wf.cost_store is cost_store
    assert wf.event_store is event_store
    assert wf.initial_plan is initial_plan
    assert wf.usage_gate is usage_gate
    assert wf.escalation_queue is escalation_queue
