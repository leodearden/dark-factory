"""Tests for truthful invocation_end telemetry + progress-extension param wiring
in ``TaskWorkflow._invoke`` (task 2360 steps 13/14).

``_invoke`` (workflow.py:7107) is the single chokepoint every agent role's
invocation flows through.  Task 2360 fix #3 (truthful reporting) requires its
``invocation_end`` event to carry ``transcript_turns``/``timed_out`` so
telemetry consumers can distinguish a productive kill from a wedge.  Task
2360 fix #1 (working-regime progress extension) requires ``_invoke`` to pass
``working_idle_secs``/``absolute_cap_secs`` down to ``invoke_with_cap_retry``
so the shared-layer extension (already wired end-to-end as of step-6) actually
engages for every role dispatched through the workflow.

RED phase (step-13): both assertions fail today —
- the ``invocation_end`` event's ``data`` dict omits ``transcript_turns`` and
  ``timed_out``;
- the ``invoke_with_cap_retry`` call omits ``working_idle_secs`` and
  ``absolute_cap_secs`` entirely.
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

# Distinct sentinel values (far from both the legacy 1200/600 defaults and the
# task-2360 config defaults 1800/7200) so a GREEN pass proves the values are
# actually READ from config, not coincidentally matching a hardcoded default.
_WORKING_IDLE_SECS_SENTINEL = 999.0
_INVOCATION_TIMEOUT_SENTINEL = 8888.0


def _make_workflow(*, event_store: _RecordingEventStore) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for ``_invoke`` telemetry/param-wiring tests.

    Mirrors ``test_workflow_escalation_warning.py:_make_workflow`` — MagicMock
    for all dependencies; only the fields under test (``event_store`` and the
    two progress-extension config knobs) are given concrete values.
    """
    assignment = MagicMock()
    assignment.task_id = '2360'
    assignment.task = {'id': '2360', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    cfg = MagicMock(spec_set=_spec)
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.timeouts.working_idle_secs = _WORKING_IDLE_SECS_SENTINEL
    cfg.invocation_timeout = _INVOCATION_TIMEOUT_SENTINEL

    return TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
    )


def _progress_agent_result() -> AgentResult:
    """AgentResult satisfying is_timed_out_with_progress(): timed_out + transcript_turns>0."""
    return AgentResult(
        success=False,
        output='',
        timed_out=True,
        turns=0,
        cost_usd=1.5,
        duration_ms=1_200_000,
        transcript_turns=42,
    )


@pytest.mark.asyncio
class TestInvocationEndTruthfulTelemetry:
    """``_invoke``'s ``invocation_end`` event data carries ``transcript_turns``/``timed_out``."""

    async def test_invocation_end_data_includes_transcript_turns_and_timed_out(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        stub_result = _progress_agent_result()

        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=stub_result),
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        invocation_end_entries = [
            entry for (etype, entry) in rec.events if etype == EventType.invocation_end
        ]
        assert len(invocation_end_entries) == 1, (
            f'expected exactly one invocation_end event; got {rec.events!r}'
        )
        data = invocation_end_entries[0]['data']
        assert data.get('transcript_turns') == stub_result.transcript_turns, (
            f'expected transcript_turns={stub_result.transcript_turns!r} in '
            f'invocation_end data; got {data!r}'
        )
        assert data.get('timed_out') == stub_result.timed_out, (
            f'expected timed_out={stub_result.timed_out!r} in invocation_end '
            f'data; got {data!r}'
        )


@pytest.mark.asyncio
class TestInvokeForwardsProgressExtensionParams:
    """``_invoke`` passes ``working_idle_secs``/``absolute_cap_secs`` to ``invoke_with_cap_retry``."""

    async def test_invoke_with_cap_retry_receives_progress_extension_kwargs(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        stub_result = _progress_agent_result()
        mock_invoke_with_cap_retry = AsyncMock(return_value=stub_result)

        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=mock_invoke_with_cap_retry,
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        mock_invoke_with_cap_retry.assert_awaited_once()
        call_kwargs = mock_invoke_with_cap_retry.call_args.kwargs
        assert call_kwargs.get('working_idle_secs') == _WORKING_IDLE_SECS_SENTINEL, (
            f'expected working_idle_secs={_WORKING_IDLE_SECS_SENTINEL!r}; '
            f'got {call_kwargs.get("working_idle_secs")!r}'
        )
        assert call_kwargs.get('absolute_cap_secs') == _INVOCATION_TIMEOUT_SENTINEL, (
            f'expected absolute_cap_secs={_INVOCATION_TIMEOUT_SENTINEL!r}; '
            f'got {call_kwargs.get("absolute_cap_secs")!r}'
        )
