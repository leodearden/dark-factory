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
from _workflow_helpers import fixed_prompt

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.routing import RoutingDecision
from orchestrator.workflow import TaskWorkflow

# Distinct sentinel values (far from both the legacy 1200/600 defaults and the
# task-2360 config defaults 1800/7200) so a GREEN pass proves the values are
# actually READ from config, not coincidentally matching a hardcoded default.
_WORKING_IDLE_SECS_SENTINEL = 999.0
_INVOCATION_TIMEOUT_SENTINEL = 8888.0
# Concrete routing-resolved ceilings for the cap-attribution tests.
_BUDGET_CEILING = 5.0
_TURN_CEILING = 50


def _make_workflow(
    *, event_store: _RecordingEventStore, cost_store: MagicMock | None = None,
) -> TaskWorkflow:
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
    # Numeric ceilings: _invoke compares a failed run's cost/turns against
    # them (classify_cap_kill), which a MagicMock ceiling cannot support.
    cfg.budgets.implementer = _BUDGET_CEILING
    cfg.max_turns.implementer = _TURN_CEILING
    # classify_cap_kill applies its numeric fallback only to the claude backend.
    cfg.backends.implementer = 'claude'
    # These tests pass a plain tmp_path as cwd (not a real linked worktree) and
    # do not assert on sandbox wiring; disable the sandbox block so _invoke does
    # not call compute_write_set(cwd) on a non-worktree path (task 2905 α3).
    cfg.sandbox.enabled = False

    return TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
        cost_store=cost_store,  # type: ignore[arg-type]
    )


def _recording_cost_store() -> MagicMock:
    cost_store = MagicMock()
    cost_store.save_invocation = AsyncMock()
    cost_store.model_cost_in_window = AsyncMock(return_value=0.0)
    return cost_store


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
            await wf._invoke(IMPLEMENTER, build_prompt=fixed_prompt('x'), cwd=tmp_path)

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

    async def _invocation_end_data(self, tmp_path: Path, stub_result: AgentResult) -> dict:
        """Drive ``_invoke`` with *stub_result* and return the sole
        ``invocation_end`` event's ``data`` dict."""
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=stub_result),
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, build_prompt=fixed_prompt('p'), cwd=tmp_path)
        entries = [entry for (etype, entry) in rec.events if etype == EventType.invocation_end]
        assert len(entries) == 1, f'expected exactly one invocation_end event; got {rec.events!r}'
        return entries[0]['data']

    async def test_invocation_end_data_includes_ended_awaiting_background_true(
        self, tmp_path: Path,
    ) -> None:
        """The flag that DECIDES the success verdict for a downgraded run must
        appear in telemetry.  The stub is the exact impossible-looking triple
        this class produces: success=False with subtype='success' and
        timed_out=False (task 3639)."""
        stub_result = AgentResult(
            success=False,
            output='done',
            subtype='success',
            turns=19,
            duration_ms=681_000,
            timed_out=False,
            ended_awaiting_background=True,
        )
        data = await self._invocation_end_data(tmp_path, stub_result)
        assert data.get('ended_awaiting_background') is True, (
            f'expected ended_awaiting_background=True in invocation_end data; got {data!r}'
        )

    async def test_invocation_end_data_includes_ended_awaiting_background_false(
        self, tmp_path: Path,
    ) -> None:
        """PRESENT-and-False on an ordinary success, not absent: absence is
        exactly what made the class unqueryable (0 of 3,340 rows), leaving the
        false-positive rate's denominator unknowable."""
        stub_result = AgentResult(
            success=True, output='ok', subtype='success', turns=3, duration_ms=1_000,
        )
        data = await self._invocation_end_data(tmp_path, stub_result)
        assert 'ended_awaiting_background' in data, (
            f'expected ended_awaiting_background key PRESENT in invocation_end data; got {data!r}'
        )
        assert data['ended_awaiting_background'] is False, (
            f'expected ended_awaiting_background=False; got {data["ended_awaiting_background"]!r}'
        )


@pytest.mark.asyncio
class TestInvocationRecordsModelIdAndCeilingKills:
    """``_invoke`` records the exact served ``model_id`` and which configured
    ceiling (if any) ended the run — in BOTH the ``invocation_end`` event and
    the ``invocations`` row (task 4826).

    Like ``ended_awaiting_background``, both cap keys are asserted PRESENT on
    a normal run, not only when true, so the saturation rate is computable.

    Routing is pinned to a real ``RoutingDecision`` (the
    ``test_workflow_routing_decision.py`` idiom): this module's config double
    would otherwise resolve the ceilings to MagicMocks, and a cap classifier
    compared against a MagicMock proves nothing.
    """

    _DECISION = RoutingDecision(
        model='opus',
        effort='high',
        budget_usd=_BUDGET_CEILING,
        max_turns=_TURN_CEILING,
        source_layer='config',
        rule_id=None,
    )

    async def _drive(
        self, tmp_path: Path, result: AgentResult, *, backend: str = 'claude',
    ) -> tuple[dict, dict]:
        """Run ``_invoke`` once on *backend* returning *result*; give back
        ``(invocation_end data, save_invocation kwargs)``."""
        rec = _RecordingEventStore()
        cost_store = _recording_cost_store()
        wf = _make_workflow(event_store=rec, cost_store=cost_store)
        wf.config.backends.implementer = backend
        mock_invoke = AsyncMock(return_value=result)

        with (
            patch('orchestrator.workflow.resolve_route', return_value=self._DECISION),
            patch('orchestrator.workflow.invoke_with_cap_retry', new=mock_invoke),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, build_prompt=fixed_prompt('p'), cwd=tmp_path)

        invoke_kw = mock_invoke.call_args.kwargs
        assert invoke_kw['max_budget_usd'] == _BUDGET_CEILING
        assert invoke_kw['max_turns'] == _TURN_CEILING
        entries = [entry for (etype, entry) in rec.events if etype == EventType.invocation_end]
        assert len(entries) == 1, f'expected exactly one invocation_end event; got {rec.events!r}'
        cost_store.save_invocation.assert_awaited_once()
        return entries[0]['data'], cost_store.save_invocation.call_args.kwargs

    async def test_exact_model_id_rides_beside_the_alias(self, tmp_path: Path) -> None:
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(success=True, output='ok', subtype='success', model_id='claude-opus-5'),
        )
        assert data['model'] == 'opus'
        assert data['model_id'] == 'claude-opus-5'
        assert save_kw['model'] == 'opus'
        assert save_kw['model_id'] == 'claude-opus-5'

    async def test_model_id_none_is_emitted_not_omitted(self, tmp_path: Path) -> None:
        data, save_kw = await self._drive(
            tmp_path, AgentResult(success=True, output='ok', subtype='success'),
        )
        assert 'model_id' in data, f'expected model_id key PRESENT; got {data!r}'
        assert data['model_id'] is None
        assert 'model_id' in save_kw
        assert save_kw['model_id'] is None

    async def test_normal_run_emits_capped_false_and_reason_none(self, tmp_path: Path) -> None:
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(success=True, output='ok', subtype='success', turns=3, cost_usd=0.2),
        )
        assert 'capped' in data, f'expected capped key PRESENT; got {data!r}'
        assert data['capped'] is False
        assert 'capped_reason' in data, f'expected capped_reason key PRESENT; got {data!r}'
        assert data['capped_reason'] is None
        assert save_kw['capped'] is False
        assert save_kw['capped_reason'] is None

    async def test_turn_ceiling_kill_is_recorded_as_turns(self, tmp_path: Path) -> None:
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(
                success=False, output='', subtype='error_max_turns',
                turns=_TURN_CEILING, cost_usd=0.7,
            ),
        )
        assert data['capped'] is True
        assert data['capped_reason'] == 'turns'
        assert save_kw['capped'] is True
        assert save_kw['capped_reason'] == 'turns'

    async def test_budget_ceiling_kill_is_recorded_as_budget(self, tmp_path: Path) -> None:
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(
                success=False, output='', subtype='error_max_budget_usd',
                turns=12, cost_usd=_BUDGET_CEILING + 0.03,
            ),
        )
        assert data['capped'] is True
        assert data['capped_reason'] == 'budget'
        assert save_kw['capped'] is True
        assert save_kw['capped_reason'] == 'budget'

    async def test_resolved_budget_reaches_the_numeric_fallback(self, tmp_path: Path) -> None:
        """A failed run AT the routing-resolved budget with an inconclusive
        subtype is attributed to the budget ceiling — proving the resolved
        ceiling, not a constant, is what the classifier compares against."""
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(
                success=False, output='', subtype='error_during_execution',
                turns=1, cost_usd=_BUDGET_CEILING,
            ),
        )
        assert data['capped_reason'] == 'budget'
        assert save_kw['capped_reason'] == 'budget'

    async def test_non_claude_failed_run_past_the_budget_is_not_capped(
        self, tmp_path: Path,
    ) -> None:
        """codex enforces no budget ceiling, so a failed run whose estimated
        cost passed the routed budget was not ended by one."""
        data, save_kw = await self._drive(
            tmp_path,
            AgentResult(
                success=False, output='', subtype='',
                turns=1, cost_usd=_BUDGET_CEILING + 2.0,
            ),
            backend='codex',
        )
        assert data['capped'] is False
        assert data['capped_reason'] is None
        assert save_kw['capped'] is False
        assert save_kw['capped_reason'] is None


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
            await wf._invoke(IMPLEMENTER, build_prompt=fixed_prompt('x'), cwd=tmp_path)

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
