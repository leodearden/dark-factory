"""Tests for TaskWorkflow._invoke's routing_decision telemetry + metadata
mirror (plans/adaptive-model-routing-prd.md task γ, task 2533).

``_invoke`` (workflow.py:7853) is the shared per-invocation chokepoint.  Task
γ adds: (1) a ``routing_decision`` event emitted once per invocation with the
resolved model/effort/budget_usd/max_turns, source_layer, rule_id,
routing_tier, and an inputs digest; (2) a best-effort ``metadata.routing``
mirror (latest decision + bounded history) written via
``scheduler.update_task(metadata_mode='merge')`` and mirrored in-memory onto
``self.task['metadata']['routing']``.

RED phase (step-9): ``TaskWorkflow._record_routing_decision`` does not exist
yet and ``_invoke`` never calls it, so no ``routing_decision`` event is
emitted and ``scheduler.update_task`` is never awaited for a ``'routing'``
key — all four test bodies below fail today.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow

# Distinct, concrete values (not the role defaults) so a GREEN pass proves the
# routing_decision payload reflects the actually-resolved config, not a
# hardcoded fallback.
_BUDGET_USD = 10.0
_MAX_TURNS = 80


def _make_workflow(*, event_store: _RecordingEventStore) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for ``_invoke`` routing-decision tests.

    Mirrors ``test_workflow_invocation_end_truthful.py``'s ``_make_workflow``
    (pydantic_spec spec_set MagicMock cfg, patched ``invoke_with_cap_retry`` +
    ``_build_agent_env``), plus concrete cfg.models/budgets/max_turns/effort/
    timeouts/backends.implementer values — needed here because the
    routing_decision payload asserts on the actual resolved model/effort/
    budget_usd/max_turns (a bare MagicMock attribute would satisfy the
    precedent test's assertions but not these, and a MagicMock timeout_val
    would blow up the ``'timeout=%.0fs'`` log-format call in ``_invoke``).
    ``scheduler.update_task`` is replaced with an ``AsyncMock`` so the
    metadata-mirror write is awaitable and assertable.
    """
    assignment = MagicMock()
    assignment.task_id = '2533'
    assignment.task = {'id': '2533', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    cfg = MagicMock(spec_set=_spec)
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.timeouts.working_idle_secs = 999.0
    cfg.invocation_timeout = 8888.0

    cfg.models.implementer = 'sonnet'
    cfg.budgets.implementer = _BUDGET_USD
    cfg.max_turns.implementer = _MAX_TURNS
    cfg.effort.implementer = 'high'
    cfg.timeouts.implementer = 1200.0
    cfg.backends.implementer = 'claude'

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
    )
    wf.scheduler.update_task = AsyncMock(return_value=True)
    return wf


def _stub_agent_result() -> AgentResult:
    return AgentResult(
        success=True,
        output='ok',
        timed_out=False,
        turns=3,
        cost_usd=1.0,
        duration_ms=1_000,
        transcript_turns=3,
    )


def _routing_decision_entries(rec: _RecordingEventStore) -> list[dict]:
    return [entry for (etype, entry) in rec.events if etype == EventType.routing_decision]


async def _invoke_implementer(wf: TaskWorkflow, cwd: Path, *, prompt: str = 'x') -> None:
    with (
        patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new=AsyncMock(return_value=_stub_agent_result()),
        ),
        patch.object(wf, '_build_agent_env', return_value=None),
    ):
        await wf._invoke(IMPLEMENTER, prompt=prompt, cwd=cwd)


@pytest.mark.asyncio
class TestInvokeRecordsRoutingDecisionEvent:
    """``_invoke`` emits exactly one ``routing_decision`` event per call."""

    async def test_config_layer_decision_is_recorded(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1, (
            f'expected exactly one routing_decision event; got {rec.events!r}'
        )
        data = entries[0]['data']
        assert data['model'] == 'sonnet'
        assert data['effort'] == 'high'
        assert data['budget_usd'] == _BUDGET_USD
        assert data['max_turns'] == _MAX_TURNS
        assert data['source_layer'] == 'config'
        assert data['rule_id'] is None
        assert data['routing_tier'] == 0
        assert isinstance(data['inputs_digest'], str) and data['inputs_digest']

    async def test_rust_upgrade_case_records_policy_rule(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        wf.modules = ['crates/a', 'crates/b', 'crates/c']
        wf.plan = {'steps': [{}] * 12}

        await _invoke_implementer(wf, tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1
        data = entries[0]['data']
        assert data['model'] == 'opus'
        assert data['source_layer'] == 'policy_rule'
        assert data['rule_id'] == 'rust-large-plan-implementer'


@pytest.mark.asyncio
class TestInvokeMirrorsRoutingMetadata:
    """``_invoke`` best-effort mirrors the decision onto ``metadata.routing``."""

    async def test_mirror_written_via_scheduler_and_in_memory(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path)

        wf.scheduler.update_task.assert_awaited_once()
        call_args = wf.scheduler.update_task.call_args
        assert call_args.kwargs.get('metadata_mode') == 'merge'
        payload = call_args.args[1]
        assert payload['routing']['latest']['model'] == 'sonnet'

        assert wf.task['metadata']['routing']['latest']['model'] == 'sonnet'

    async def test_successive_invocations_accumulate_history(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path, prompt='first')
        await _invoke_implementer(wf, tmp_path, prompt='second')

        routing = wf.task['metadata']['routing']
        assert len(routing['history']) == 2
        assert routing['latest'] == routing['history'][-1]
