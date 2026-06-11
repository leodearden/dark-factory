"""Tests for the zero-output hang circuit breaker in TaskWorkflow.

Covers:
- Circuit breaker in _execute_iterations: fast-fail to BLOCKED after
  max_consecutive_zero_output_timeouts consecutive zero-output CLI timeouts.
- Counter reset: a non-zero-output result clears the consecutive counter.
- Caller routing in _execute_verify_review_loop: when _zero_output_hang_info
  is set the infra-typed block reason is used (NOT 'Execution iterations
  exhausted') and category='infra_issue'.

All tests are RED until step-8 (circuit breaker + routing) is implemented.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared harness
# ---------------------------------------------------------------------------


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '1739',
    max_execute_iterations: int = 10,
    max_consecutive_zero_output_timeouts: int = 2,
    recycle_config_dir_on_zero_output: bool = False,
) -> TaskWorkflow:
    """Minimal TaskWorkflow harness for zero-output hang tests.

    Mirrors test_workflow.py:_make_workflow; adds the zero-output-hang config
    attributes (max_consecutive_zero_output_timeouts, recycle_config_dir_on_zero_output).
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.merge_train_former_enabled = False
    config.merge_train_max_members = 3
    # Zero-output hang knobs
    config.max_execute_iterations = max_execute_iterations
    config.max_consecutive_zero_output_timeouts = max_consecutive_zero_output_timeouts
    config.recycle_config_dir_on_zero_output = recycle_config_dir_on_zero_output
    # Disable per-iteration extras so only the zero-output branch matters
    config.judge_after_each_iteration = False
    config.inter_iteration_rebase = False

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    wf.plan = {
        'files': [],
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'status': 'pending'}],
    }
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    return wf


def _zero_output_agent_result() -> AgentResult:
    """Return an AgentResult that satisfies is_zero_output_timeout()."""
    return AgentResult(
        success=False,
        output='Agent produced no output',
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=1_200_000,
    )


def _nonzero_agent_result() -> AgentResult:
    """Return an AgentResult with turns=3 — does NOT satisfy is_zero_output_timeout()."""
    return AgentResult(
        success=False,
        output='partial output',
        timed_out=False,
        turns=3,
        cost_usd=0.25,
        duration_ms=5_000,
    )


def _stub_iteration_helpers(wf: TaskWorkflow, invoke_result: AgentResult) -> AsyncMock:
    """Stub per-iteration helpers so _execute_iterations runs without real I/O.

    Returns the mocked _invoke AsyncMock so callers can configure side effects.
    """
    # artifacts helpers
    wf.artifacts.get_pending_steps = MagicMock(  # type: ignore[method-assign]
        return_value=[{'id': 'step-1'}],
    )
    wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
    wf.artifacts.read_plan = MagicMock(  # type: ignore[method-assign]
        return_value={
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        },
    )
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))  # type: ignore[method-assign]
    wf.artifacts.append_iteration_log = MagicMock()  # type: ignore[method-assign]
    wf.artifacts.stamp_plan_provenance = MagicMock()  # type: ignore[method-assign]

    # workflow methods
    mock_invoke = AsyncMock(return_value=invoke_result)
    wf._invoke = mock_invoke  # type: ignore[method-assign]
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value='abc123')  # type: ignore[method-assign]
    wf.briefing.build_implementer_prompt = AsyncMock(return_value='test-prompt')

    return mock_invoke


# ---------------------------------------------------------------------------
# Step-7 RED tests: circuit breaker in _execute_iterations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCircuitBreakerFastFail:
    """Circuit breaker in _execute_iterations fast-fails to BLOCKED after
    max_consecutive_zero_output_timeouts consecutive zero-output CLI timeouts.

    Fails RED because _execute_iterations has no circuit breaker logic yet.
    """

    async def test_blocks_after_threshold_zero_output_timeouts(self, tmp_path: Path):
        """BLOCKED after exactly max_consecutive_zero_output_timeouts=2 zero-output calls.

        Without the circuit breaker, the loop would run max_execute_iterations=10 times.
        The breaker must short-circuit to BLOCKED after exactly 2 calls.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
        )
        mock_invoke = _stub_iteration_helpers(wf, _zero_output_agent_result())

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        # Circuit breaker: must stop at 2 calls, NOT run all 10
        assert mock_invoke.call_count == 2, (
            f'Expected 2 _invoke calls (threshold), got {mock_invoke.call_count}'
        )

    async def test_zero_output_hang_info_set_on_trip(self, tmp_path: Path):
        """_zero_output_hang_info is populated with reason+detail when the breaker trips.

        Fails RED because the attribute is not declared in __init__ nor set by the breaker.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        await wf._execute_iterations()

        info = wf._zero_output_hang_info
        assert info is not None, '_zero_output_hang_info must be set on circuit-breaker trip'
        assert 'reason' in info, f'Expected "reason" key in _zero_output_hang_info: {info!r}'
        assert 'detail' in info, f'Expected "detail" key in _zero_output_hang_info: {info!r}'
        # reason must be distinct from the generic iteration-exhausted message
        assert 'infra' in info['reason'].lower() or 'zero-output' in info['reason'].lower(), (
            f'Expected infra/zero-output in reason, got: {info["reason"]!r}'
        )

    async def test_preserve_config_dir_set_on_trip(self, tmp_path: Path):
        """_preserve_config_dir=True when the breaker trips (forensic evidence kept).

        Fails RED because the attribute does not exist yet.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        await wf._execute_iterations()

        assert wf._preserve_config_dir is True, (
            'Expected _preserve_config_dir=True when circuit breaker trips'
        )


@pytest.mark.asyncio
class TestCircuitBreakerCounterReset:
    """A non-zero-output result between two zero-output results resets the counter.

    Sequence: zero-output, non-zero-output, zero-output, zero-output → trips at call 4.
    Without the reset the breaker would trip at call 2.

    Fails RED because the circuit breaker (and its reset logic) do not exist.
    """

    async def test_nonzero_result_resets_consecutive_counter(self, tmp_path: Path):
        """Counter resets after a non-zero-output result so breaker requires threshold
        NEW consecutive failures after the reset before tripping.

        With threshold=2, sequence [zero, nonzero, zero, zero] trips at iteration 4
        (not 2, which would fire if the counter were NOT reset).

        The side_effect is padded to max_execute_iterations=10 so the test does not
        crash with StopAsyncIteration when the circuit breaker is absent (RED path
        fails on assert call_count==4, not on an unexpected exception).
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
        )
        mock_invoke = _stub_iteration_helpers(wf, _zero_output_agent_result())
        # Sequence: zero, nonzero, zero, zero → tripping pair starts fresh after reset
        # Remaining 6 items let the loop reach max_execute_iterations when breaker absent.
        mock_invoke.side_effect = [
            _zero_output_agent_result(),   # iter 1: count=1 (< 2)
            _nonzero_agent_result(),       # iter 2: count reset to 0
            _zero_output_agent_result(),   # iter 3: count=1 (< 2)
            _zero_output_agent_result(),   # iter 4: count=2 → BLOCKED (circuit breaker)
            # extra items consumed only when circuit breaker is absent (RED path)
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
        ]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        # If counter resets correctly, breaker trips at iter 4 (not iter 2)
        assert mock_invoke.call_count == 4, (
            f'Expected 4 _invoke calls (reset counter before threshold), '
            f'got {mock_invoke.call_count}'
        )


# ---------------------------------------------------------------------------
# Step-7 RED test: _execute_verify_review_loop routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteVerifyReviewLoopZeroOutputRouting:
    """_execute_verify_review_loop routes to the infra block reason when
    _zero_output_hang_info is set, not the generic 'Execution iterations exhausted'.

    Fails RED because _execute_verify_review_loop does not check
    _zero_output_hang_info yet.
    """

    async def test_zero_output_hang_info_routes_to_infra_block(self, tmp_path: Path):
        """When _zero_output_hang_info is set and _execute_iterations returns BLOCKED,
        _mark_blocked is called with the infra reason and category='infra_issue',
        NOT with 'Execution iterations exhausted'.
        """
        wf = _make_workflow(tmp_path=tmp_path)

        # Pre-set the info dict (as the circuit breaker will do in step-8)
        infra_reason = 'infra: zero-output CLI hang (consecutive fresh-invocation timeouts)'
        wf._zero_output_hang_info = {  # type: ignore[attr-defined]
            'reason': infra_reason,
            'detail': 'iterations=2 duration_ms=2400000',
        }

        # Stub _execute_iterations to return BLOCKED
        wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        # Capture _mark_blocked calls
        mark_blocked_calls: list[tuple] = []

        async def fake_mark_blocked(reason, *, detail='', category='task_failure', **_kw):
            mark_blocked_calls.append((reason, detail, category))
            return WorkflowOutcome.BLOCKED

        wf._mark_blocked = fake_mark_blocked  # type: ignore[method-assign]

        result = await wf._execute_verify_review_loop()

        assert result == WorkflowOutcome.BLOCKED
        assert mark_blocked_calls, '_mark_blocked must have been called'
        called_reason, called_detail, called_category = mark_blocked_calls[0]

        # Must NOT fall through to the generic message
        assert called_reason != 'Execution iterations exhausted', (
            f'Expected infra-specific reason, got generic: {called_reason!r}'
        )
        # Must be the infra reason
        assert 'infra' in called_reason.lower() or 'zero-output' in called_reason.lower(), (
            f'Expected infra/zero-output in reason, got: {called_reason!r}'
        )
        # Must be classified as infra_issue
        assert called_category == 'infra_issue', (
            f'Expected category="infra_issue", got: {called_category!r}'
        )
