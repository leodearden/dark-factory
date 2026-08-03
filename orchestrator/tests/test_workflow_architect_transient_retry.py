"""DEFECT B (task 3143): the architect's transient-glitch retry must not be
gated on ``result.success``.

``_plan``'s anomalous-premature-exit heuristic (turns <= 2, cost < $0.20, no
plan.json) sits BELOW the ``not result.success`` terminal ``_mark_blocked``
return, so a failure carrying that exact signature can never reach it.  The
2026-07-28 ~16:31Z incident payload -- turns=0, cost_usd=0.0, no plan.json --
satisfies every clause of the heuristic and would have been retried had it been
reachable; instead the planning task was terminally blocked on a sub-second
transport glitch that did no work and billed nothing.

Kept in its own module (no ``orchestrator/tests/conftest.py`` edit): editing
that conftest makes verify.py's ``has_conftest`` fall back to the full
owning-package suite at merge-verify time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from shared.cli_invoke import AgentResult

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    mark_blocked: AsyncMock
    handle_no_plan: AsyncMock
    invoke: AsyncMock


def _empty_output_failure() -> AgentResult:
    """The observed payload: zero turns, zero cost, no plan, NOT timed out."""
    return AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_empty_output',
        turns=0,
        cost_usd=0.0,
        duration_ms=17_331,
    )


def _cli_input_rejected_failure() -> AgentResult:
    """Same shape, now carrying the distinct pre-turn-rejection subtype."""
    return AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_cli_input_rejected',
        turns=0,
        cost_usd=0.0,
        duration_ms=2_100,
        stderr=(
            'Error: Input must be provided either through stdin or as a '
            'prompt argument when using --print\n'
        ),
    )


def _succeeded() -> AgentResult:
    return AgentResult(
        success=True,
        output='wrote plan',
        cost_usd=1.20,
        duration_ms=90_000,
        turns=14,
    )


def _make(
    *,
    worktree: Path,
    project_root: Path,
    invoke_side_effect,
    task_id: str = '3112',
) -> _Fixture:
    """Build a TaskWorkflow ready for ``_plan()``.

    Mirrors test_workflow_architect_l0_promotion.py's fixture; the only
    difference is that ``_invoke`` is scripted per-call so dispatch COUNT is
    the observable.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd', 'metadata': {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='currentmain')

    escalation_queue = MagicMock()
    escalation_queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=escalation_queue,
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    invoke = AsyncMock(side_effect=invoke_side_effect)
    wf._invoke = invoke  # type: ignore[method-assign]

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]
    handle_no_plan = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._handle_no_plan_failure = handle_no_plan  # type: ignore[method-assign]

    wf.briefing.build_architect_prompt = AsyncMock(return_value='prompt')
    wf.briefing.build_revalidation_prompt = AsyncMock(return_value='prompt')

    return _Fixture(
        wf=wf,
        artifacts=artifacts,
        mark_blocked=mark_blocked,
        handle_no_plan=handle_no_plan,
        invoke=invoke,
    )


def _write_finalized_plan(artifacts: TaskArtifacts, *, steps: int = 2) -> None:
    artifacts.write_plan({
        '_finalized_at': '2026-08-03T12:00:00Z',
        'analysis': 'a',
        'steps': [
            {'id': f'step-{i}', 'description': 'do it', 'type': 'impl',
             'status': 'pending', 'commit': None}
            for i in range(1, steps + 1)
        ],
        'files': [],
        'prerequisites': [],
        'design_decisions': [],
        'reuse': [],
    })


@pytest.mark.asyncio
class TestTransientFailureIsRetriedOnce:
    """THE RED CORE: a zero-work FAILURE with the transient-glitch signature
    must buy the same one retry the success path already gets.

    Nothing was billed and no turns ran, so the retry is free of
    double-billing -- and it is the only thing standing between a sub-second
    timing glitch and a terminally-blocked planning task.
    """

    async def _drive(self, tmp_path: Path, first: AgentResult) -> _Fixture:
        calls = {'n': 0}

        async def _scripted(*_args, **_kwargs) -> AgentResult:
            calls['n'] += 1
            if calls['n'] == 1:
                return first
            # The retry succeeds and leaves a finalized plan on disk, exactly
            # as a healthy architect run does.
            _write_finalized_plan(f.artifacts)
            return _succeeded()

        f = _make(
            worktree=tmp_path / 'wt',
            project_root=tmp_path / 'proj',
            invoke_side_effect=_scripted,
        )
        await f.wf._plan()
        return f

    async def test_empty_output_failure_is_retried_once(self, tmp_path: Path):
        f = await self._drive(tmp_path, _empty_output_failure())

        assert f.invoke.await_count == 2, (
            f'expected the transient failure to buy exactly one retry, got '
            f'{f.invoke.await_count} dispatch(es) -- today the heuristic sits '
            f'BELOW the terminal _mark_blocked return and is unreachable'
        )
        f.mark_blocked.assert_not_called()

    async def test_cli_input_rejected_failure_is_retried_once(self, tmp_path: Path):
        f = await self._drive(tmp_path, _cli_input_rejected_failure())

        assert f.invoke.await_count == 2
        f.mark_blocked.assert_not_called()


@pytest.mark.asyncio
class TestTransientRetryIsBounded:
    """BOUNDEDNESS: exactly one extra dispatch, never a loop.  A second
    consecutive zero-work failure is not a glitch and must go terminal."""

    async def test_two_consecutive_failures_block_after_two_dispatches(self, tmp_path: Path):
        f = _make(
            worktree=tmp_path / 'wt',
            project_root=tmp_path / 'proj',
            invoke_side_effect=[_empty_output_failure(), _empty_output_failure()],
        )

        await f.wf._plan()

        assert f.invoke.await_count == 2, (
            f'no third Opus dispatch may be bought, got {f.invoke.await_count}'
        )
        f.mark_blocked.assert_awaited_once()
        assert f.mark_blocked.await_args.args[0].startswith('Planning failed: ')


@pytest.mark.asyncio
class TestNonTransientKindsStayTerminal:
    """NARROWNESS (passes today, must keep passing): a failure that either
    burned the full wall clock or is deterministic buys NO second Opus
    dispatch."""

    @pytest.mark.parametrize(
        'result',
        [
            pytest.param(
                AgentResult(
                    success=False, output='', subtype='error_max_turns',
                    turns=50, cost_usd=4.80, output_tokens=900,
                ),
                id='max_turns',
            ),
            pytest.param(
                AgentResult(
                    success=False, output='', subtype='error_empty_output',
                    turns=0, cost_usd=0.0, api_error_status=503,
                ),
                id='api_error',
            ),
            pytest.param(
                AgentResult(
                    success=False, output='model not found', subtype='',
                    turns=0, cost_usd=0.0, api_error_status=404,
                ),
                id='model_not_found',
            ),
            pytest.param(
                AgentResult(
                    success=False, output='', subtype='error_empty_output',
                    turns=0, cost_usd=0.0, duration_ms=600_000, timed_out=True,
                ),
                id='timed_out',
            ),
        ],
    )
    async def test_kind_goes_straight_to_mark_blocked(self, tmp_path: Path, result):
        f = _make(
            worktree=tmp_path / 'wt',
            project_root=tmp_path / 'proj',
            invoke_side_effect=[result, _succeeded()],
        )

        await f.wf._plan()

        assert f.invoke.await_count == 1, (
            f'a non-transient failure must not buy a second expensive '
            f'dispatch, got {f.invoke.await_count}'
        )
        f.mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
class TestSalvagePrecedenceUnchanged:
    """The salvage path still outranks the retry: a failed run that
    nevertheless left a finalized plan on disk is salvaged, not re-dispatched
    and not blocked."""

    async def test_finalized_plan_on_disk_is_salvaged_without_retry(self, tmp_path: Path):
        worktree = tmp_path / 'wt'
        f = _make(
            worktree=worktree,
            project_root=tmp_path / 'proj',
            invoke_side_effect=[_empty_output_failure(), _succeeded()],
        )
        _write_finalized_plan(f.artifacts, steps=3)

        await f.wf._plan()

        assert f.invoke.await_count == 1, 'a salvageable run must not be retried'
        f.mark_blocked.assert_not_called()
        assert len(f.wf.plan.get('steps', [])) == 3


@pytest.mark.asyncio
class TestSuccessPathHeuristicUnchanged:
    """REGRESSION PIN for the pre-existing success=True heuristic, which no
    existing test covers -- extracting it into a shared predicate must not
    change it."""

    async def test_anomalous_success_still_retries_exactly_once(self, tmp_path: Path):
        anomalous = AgentResult(
            success=True, output='', turns=1, cost_usd=0.05, duration_ms=800,
        )
        calls = {'n': 0}

        async def _scripted(*_args, **_kwargs) -> AgentResult:
            calls['n'] += 1
            if calls['n'] == 1:
                return anomalous
            _write_finalized_plan(f.artifacts)
            return _succeeded()

        f = _make(
            worktree=tmp_path / 'wt',
            project_root=tmp_path / 'proj',
            invoke_side_effect=_scripted,
        )

        await f.wf._plan()

        assert f.invoke.await_count == 2
        f.mark_blocked.assert_not_called()
