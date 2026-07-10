"""Tests for is_timed_out_with_progress resume wiring in TaskWorkflow._execute_iterations.

Covers:
- Progress-timeout (timed_out=True, transcript_turns>0) sets _pending_resume_session_id
  and _pending_resume_role='implementer' in _execute_iterations, enabling the existing
  _invoke resume lifecycle (workflow.py:6240-6253) on the next iteration.
- Zero-output and success results do NOT set the pending-resume fields (regression guards).
- A progress-timeout resets consecutive_zero_output (proof-of-life), preventing
  premature zero-output circuit-breaker trips in mixed wedge/progress/wedge sequences.

RED phase (step-1): TestResumeOnProgressWiring.test_progress_timeout_sets_pending_resume
fails because _execute_iterations has no is_timed_out_with_progress branch yet —
pending fields remain None.

RED phase (step-3): TestProgressResetsZeroOutputCounter.test_progress_resets_consecutive_counter
fails because the step-2 elif `continue`s without resetting the counter —
circuit breaker trips at call_count==3 instead of 4.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import ZERO_OUTPUT_HANG_REASON, TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Shared harness (mirrors test_workflow_zero_output_hang.py)
# ---------------------------------------------------------------------------


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '1780',
    max_execute_iterations: int = 10,
    max_consecutive_zero_output_timeouts: int = 2,
    max_progress_resume_iterations: int = 20,
) -> TaskWorkflow:
    """Minimal TaskWorkflow harness for resume-on-progress tests.

    Mirrors test_workflow_zero_output_hang.py:_make_workflow.  Includes the
    zero-output-hang config attributes required by _execute_iterations.
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
    config.max_execute_iterations = max_execute_iterations
    config.max_consecutive_zero_output_timeouts = max_consecutive_zero_output_timeouts
    config.max_progress_resume_iterations = max_progress_resume_iterations
    config.recycle_config_dir_on_zero_output = False
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


def _progress_timeout_agent_result() -> AgentResult:
    """AgentResult satisfying is_timed_out_with_progress() — transcript_turns=5 > 0."""
    return AgentResult(
        success=False,
        output='',
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=1_200_000,
        transcript_turns=5,  # > 0 ⇒ is_timed_out_with_progress() == True
    )


def _zero_output_agent_result() -> AgentResult:
    """AgentResult satisfying is_zero_output_timeout() — transcript_turns=0."""
    return AgentResult(
        success=False,
        output='Agent produced no output',
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=1_200_000,
        transcript_turns=0,  # == 0 ⇒ is_zero_output_timeout() == True
    )


def _success_agent_result() -> AgentResult:
    """A successful (non-timeout) AgentResult."""
    return AgentResult(
        success=True,
        output='All done!',
        timed_out=False,
        turns=3,
        cost_usd=0.25,
        duration_ms=5_000,
    )


def _stub_iteration_helpers(wf: TaskWorkflow, invoke_result: AgentResult) -> AsyncMock:
    """Stub per-iteration helpers so _execute_iterations runs without real I/O.

    Returns the mocked _invoke AsyncMock so callers can configure side effects.
    """
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

    mock_invoke = AsyncMock(return_value=invoke_result)
    wf._invoke = mock_invoke  # type: ignore[method-assign]
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value='abc123')  # type: ignore[method-assign]
    wf.briefing.build_implementer_prompt = AsyncMock(return_value='test-prompt')

    return mock_invoke


# ---------------------------------------------------------------------------
# Step-1 RED tests: pending-resume fields set on progress timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResumeOnProgressWiring:
    """is_timed_out_with_progress in _execute_iterations sets pending-resume fields.

    (a) test_progress_timeout_sets_pending_resume — RED driver:
        no elif branch yet, so fields remain None after a progress-timeout.
    (b) test_zero_output_does_not_set_pending_resume — green-from-start guard:
        zero-output hits the existing circuit-breaker branch, not the new elif.
    (c) test_success_does_not_set_pending_resume — green-from-start guard:
        success falls into else (counter reset), never touches pending fields.
    """

    async def test_progress_timeout_sets_pending_resume(self, tmp_path: Path):
        """A progress-timeout sets _pending_resume_session_id and _pending_resume_role.

        max_execute_iterations=1 so the loop runs one iteration, the elif branch
        sets the fields and `continue`s, then the cap check fires → BLOCKED.
        The fields must survive the BLOCKED exit so _invoke can consume them next run.

        Fails RED: no is_timed_out_with_progress branch ⇒ fields stay None.
        """
        wf = _make_workflow(tmp_path=tmp_path, max_execute_iterations=1)
        _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        # Pre-set the session-id stash (_invoke normally sets this; mocked _invoke skips it)
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert wf._pending_resume_session_id == 'killed-sid', (
            f'Expected _pending_resume_session_id="killed-sid", '
            f'got {wf._pending_resume_session_id!r}'
        )
        assert wf._pending_resume_role == 'implementer', (
            f'Expected _pending_resume_role="implementer", '
            f'got {wf._pending_resume_role!r}'
        )

    async def test_zero_output_does_not_set_pending_resume(self, tmp_path: Path):
        """Zero-output timeout does NOT set pending-resume fields.

        max_consecutive_zero_output_timeouts=1 so the circuit breaker trips on the
        first zero-output timeout → BLOCKED.  is_timed_out_with_progress is False
        for transcript_turns=0, so the elif is never entered.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=1,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert wf._pending_resume_session_id is None, (
            f'Expected _pending_resume_session_id=None for zero-output, '
            f'got {wf._pending_resume_session_id!r}'
        )
        assert wf._pending_resume_role is None, (
            f'Expected _pending_resume_role=None for zero-output, '
            f'got {wf._pending_resume_role!r}'
        )

    async def test_success_does_not_set_pending_resume(self, tmp_path: Path):
        """A successful result does NOT set pending-resume fields.

        max_execute_iterations=1 → loop runs once, cap fires → BLOCKED.
        success=True means timed_out=False, so is_timed_out_with_progress is False
        and the elif is never entered.
        """
        wf = _make_workflow(tmp_path=tmp_path, max_execute_iterations=1)
        _stub_iteration_helpers(wf, _success_agent_result())

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert wf._pending_resume_session_id is None, (
            f'Expected _pending_resume_session_id=None for success, '
            f'got {wf._pending_resume_session_id!r}'
        )
        assert wf._pending_resume_role is None, (
            f'Expected _pending_resume_role=None for success, '
            f'got {wf._pending_resume_role!r}'
        )


# ---------------------------------------------------------------------------
# Step-3 RED test: progress-timeout resets the consecutive-zero-output counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProgressResetsZeroOutputCounter:
    """A progress-timeout resets consecutive_zero_output (proof-of-life).

    Before γ, a progress-timeout fell into the `else` branch which reset the counter.
    The new `elif … continue` bypasses that else, so without an explicit reset a mixed
    wedge/progress/wedge sequence trips the zero-output circuit breaker prematurely.

    Sequence: zero_output, progress, zero_output, zero_output
    With reset    (step-4): counter: 1 → 0 → 1 → 2 → BLOCKED, call_count == 4 ✓
    Without reset (step-2): counter: 1 → 1 → 2 → BLOCKED,     call_count == 3 ✗

    Fails RED at step-2 because the elif does not reset consecutive_zero_output yet.
    """

    async def test_progress_resets_consecutive_counter(self, tmp_path: Path):
        """Progress-timeout resets the zero-output counter so breaker requires
        max_consecutive_zero_output_timeouts=2 NEW consecutive failures after reset.

        Sequence [zero, progress, zero, zero] trips at call_count==4.
        Without the reset the breaker would trip at call_count==3.
        Padding items let the loop reach max_execute_iterations when breaker absent (RED path).
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
        )
        mock_invoke = _stub_iteration_helpers(wf, _zero_output_agent_result())
        # Pre-set session-id stash (real _invoke sets this; mocked _invoke skips it)
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        mock_invoke.side_effect = [
            _zero_output_agent_result(),    # iter 1: count=1 (< 2, continue)
            _progress_timeout_agent_result(),  # iter 2: reset count=0 + set fields + continue
            _zero_output_agent_result(),    # iter 3: count=1 (< 2, continue)
            _zero_output_agent_result(),    # iter 4: count=2 → BLOCKED (circuit breaker)
            # Padding: consumed only when circuit breaker is absent (RED path)
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
            _zero_output_agent_result(),
        ]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert mock_invoke.call_count == 4, (
            f'Expected 4 _invoke calls (counter resets after progress), '
            f'got {mock_invoke.call_count}'
        )


# ---------------------------------------------------------------------------
# task 2360 step-17: progress-resume iteration accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProgressResumeIterationAccounting:
    """A progress-timeout+resume pair counts as ONE against max_execute_iterations
    (excluded from the cap check), independently bounded by
    config.max_progress_resume_iterations so a non-converging kill/resume
    loop cannot run forever (task 2360, reify-4827).

    (a) test_progress_resume_pairs_excluded_from_iteration_cap — PAIR-COUNTS-AS-ONE:
        two progress-timeouts followed by a successful, plan-clearing iteration
        must NOT trip a max_execute_iterations=2 cap (both progress-timeouts
        are excluded from the count), and the third (successful) invocation
        must actually run.
    (b) test_progress_resume_churn_bound_blocks_with_distinct_reason — CHURN-BOUND:
        an endless progress-timeout stream trips the independent
        max_progress_resume_iterations=3 bound and returns BLOCKED with a
        reason distinct from both ZERO_OUTPUT_HANG_REASON and the generic
        'Execution iterations exhausted' path.

    Fails RED: today every progress-timeout increments execute_iterations
    toward the cap with no exclusion, and there is no independent churn bound
    — (a) would BLOCK after 2 invocations (never reaching the 3rd), and (b)
    has no ``_progress_resume_churn_info`` attribute at all.
    """

    async def test_progress_resume_pairs_excluded_from_iteration_cap(self, tmp_path: Path):
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=2,
            max_progress_resume_iterations=100,
        )
        mock_invoke = _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        # Pending steps truthy for the first 3 while-checks, cleared after the
        # 3rd (successful) invocation so the loop exits with DONE.
        wf.artifacts.get_pending_steps.side_effect = [  # type: ignore[attr-defined]
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [],
        ]
        mock_invoke.side_effect = [
            _progress_timeout_agent_result(),  # iter 1: progress-timeout (excluded)
            _progress_timeout_agent_result(),  # iter 2: progress-timeout (excluded)
            _success_agent_result(),           # iter 3: succeeds, plan clears
        ]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE (cap not tripped by excluded progress-resumes), '
            f'got {outcome!r}'
        )
        assert mock_invoke.call_count == 3, (
            f'Expected all 3 invocations to run (2 excluded progress-resumes '
            f'+ 1 successful), got {mock_invoke.call_count}'
        )

    async def test_progress_resume_churn_bound_blocks_with_distinct_reason(self, tmp_path: Path):
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=100,
            max_progress_resume_iterations=3,
        )
        mock_invoke = _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]
        # Endless progress-timeout stream — pending steps never clear
        # (_stub_iteration_helpers' default get_pending_steps stays truthy).

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert mock_invoke.call_count == 3, (
            f'Expected exactly 3 progress-resume invocations before the churn '
            f'bound trips, got {mock_invoke.call_count}'
        )
        info = wf._progress_resume_churn_info
        assert info is not None, (
            '_progress_resume_churn_info must be set when the churn bound trips'
        )
        assert 'reason' in info and 'detail' in info
        assert info['reason'] != ZERO_OUTPUT_HANG_REASON, (
            'progress-resume churn must use a reason distinct from the '
            'zero-output-hang reason'
        )
        assert info['reason'] != 'Execution iterations exhausted', (
            'progress-resume churn must use a reason distinct from the '
            'generic iteration-cap-exhausted reason'
        )
        # The generic zero-output hang info must NOT be set by this path.
        assert wf._zero_output_hang_info is None


# ---------------------------------------------------------------------------
# reify comprehensive review (task 2360): progress_resume_total must be
# cumulative across _execute_iterations re-entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProgressResumeTotalSurvivesReentry:
    """progress_resume_total (the max_execute_iterations cap-exclusion
    counter) must be cumulative across separate _execute_iterations calls,
    mirroring self.metrics.execute_iterations' own never-reset lifetime.

    _execute_verify_review_loop re-enters _execute_iterations after a review
    cycle (_replan) or an amendment round (_amend) — see the `continue`
    statements in that loop. If progress_resume_total resets to 0 at the top
    of _execute_iterations instead of living on self.metrics, a second (or
    later) execute phase loses the exclusion earned by progress-resumes in
    earlier phases: the cap check then compares the FULL cumulative
    execute_iterations against a freshly-zeroed exclusion, so a task that
    passed through review/amendment cycles can trip max_execute_iterations
    immediately on re-entry even though every phase individually stayed well
    under budget (reviewer_comprehensive.json finding #1).
    """

    async def test_progress_resume_total_survives_execute_phase_reentry(self, tmp_path):
        """Phase 1: 4 progress-timeouts (excluded) + 1 success -> DONE.
        Phase 2 (re-entry, as if from a replan/amend `continue`): 1 more
        success should be enough to finish -- the phase-1 exclusion must
        still hold so the cap isn't seen as already exhausted.

        Fails RED: today progress_resume_total resets to 0 on re-entry, so
        phase 2's cap check sees (execute_iterations=5 - 0 >= 5) and returns
        BLOCKED before running a single phase-2 iteration.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=5,
            max_progress_resume_iterations=100,
        )
        mock_invoke = _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        # Phase 1: 4 progress-timeouts then a success that clears pending steps.
        wf.artifacts.get_pending_steps.side_effect = [  # type: ignore[attr-defined]
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [{'id': 'step-1'}],
            [],
        ]
        mock_invoke.side_effect = [
            _progress_timeout_agent_result(),
            _progress_timeout_agent_result(),
            _progress_timeout_agent_result(),
            _progress_timeout_agent_result(),
            _success_agent_result(),
        ]

        phase1_outcome = await wf._execute_iterations()

        assert phase1_outcome == WorkflowOutcome.DONE
        assert mock_invoke.call_count == 5

        # Phase 2 (re-entry): as if a review/amendment cycle just added more
        # pending steps and looped back to EXECUTE. One more success should
        # be enough -- the phase-1 exclusion must still hold.
        wf.artifacts.get_pending_steps.side_effect = [  # type: ignore[attr-defined]
            [{'id': 'step-2'}],
            [],
        ]
        mock_invoke.side_effect = [_success_agent_result()]

        phase2_outcome = await wf._execute_iterations()

        assert phase2_outcome == WorkflowOutcome.DONE, (
            f'Expected DONE (phase-1 progress-resume exclusion must survive '
            f're-entry), got {phase2_outcome!r}'
        )
        assert mock_invoke.call_count == 6, (
            f'Expected phase 2 to actually run its 1 iteration, '
            f'got {mock_invoke.call_count} total calls'
        )


# ---------------------------------------------------------------------------
# Amendment: test that _invoke actually consumes the pending-resume fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInvokeResumesConsumesFields:
    """_invoke consumes _pending_resume_session_id/_pending_resume_role on match.

    The γ comment states: '_invoke resume lifecycle picks these up because
    _pending_resume_role == role.name == implementer'.  These tests call _invoke
    directly (not through _execute_iterations), patching only the subprocess
    layer (invoke_with_cap_retry) so the real resume-selection block
    (workflow.py:6262-6273) runs.

    A refactor that broke the role-name match or the field-clear would be
    caught here; the existing _execute_iterations tests cannot catch it because
    they mock _invoke entirely.
    """

    async def test_invoke_resume_lifecycle_consumes_pending_fields(self, tmp_path: Path):
        """_invoke passes resume_session_id='killed-sid' to invoke_with_cap_retry
        and clears both pending fields afterwards.

        Pre-sets _pending_resume_session_id='killed-sid' / _pending_resume_role='implementer'
        exactly as γ's elif branch does after a wall-killed progress-timeout, then calls
        _invoke(IMPLEMENTER, ...) with only the subprocess layer patched.

        Assertions:
        - invoke_with_cap_retry receives resume_session_id='killed-sid'
        - _pending_resume_session_id is None after the call
        - _pending_resume_role is None after the call
        """
        from unittest.mock import patch

        from orchestrator.agents.roles import IMPLEMENTER

        wf = _make_workflow(tmp_path=tmp_path)
        # Exactly what γ's elif branch sets after a progress-timeout
        wf._pending_resume_session_id = 'killed-sid'  # type: ignore[attr-defined]
        wf._pending_resume_role = 'implementer'  # type: ignore[attr-defined]

        resume_result = AgentResult(
            success=True, output='resumed ok', timed_out=False, turns=2, cost_usd=0.0,
        )

        with patch('orchestrator.workflow.invoke_with_cap_retry', new_callable=AsyncMock) as mock_iwcr:
            mock_iwcr.return_value = resume_result
            # Stub _build_agent_env to avoid traversing config.env_overrides/jobserver
            # (both are MagicMock under the spec_set config, but dict.update semantics
            # on non-dict mocks are undefined — bypass to keep the test focused).
            with patch.object(wf, '_build_agent_env', return_value=None):
                await wf._invoke(IMPLEMENTER, 'test-prompt', tmp_path)

        # The resume-selection block must have passed the stashed session id
        call_kwargs = mock_iwcr.call_args.kwargs
        assert call_kwargs.get('resume_session_id') == 'killed-sid', (
            f"Expected resume_session_id='killed-sid' in invoke_with_cap_retry call, "
            f"got {call_kwargs.get('resume_session_id')!r}"
        )

        # Pending fields must be cleared so the next _invoke starts a fresh session
        assert wf._pending_resume_session_id is None, (
            f'Expected _pending_resume_session_id=None after _invoke, '
            f'got {wf._pending_resume_session_id!r}'
        )
        assert wf._pending_resume_role is None, (
            f'Expected _pending_resume_role=None after _invoke, '
            f'got {wf._pending_resume_role!r}'
        )

    async def test_invoke_without_pending_fields_does_not_resume(self, tmp_path: Path):
        """When no pending-resume fields are set, _invoke passes resume_session_id=None.

        Guard: verifies the resume-selection block's 'else' path allocates a fresh
        UUID session without passing resume_session_id.
        """
        from unittest.mock import patch

        from orchestrator.agents.roles import IMPLEMENTER

        wf = _make_workflow(tmp_path=tmp_path)
        # No pending fields set — the 'else' path in the resume-selection block fires

        resume_result = AgentResult(
            success=True, output='fresh start', timed_out=False, turns=1, cost_usd=0.0,
        )

        with patch('orchestrator.workflow.invoke_with_cap_retry', new_callable=AsyncMock) as mock_iwcr:
            mock_iwcr.return_value = resume_result
            with patch.object(wf, '_build_agent_env', return_value=None):
                await wf._invoke(IMPLEMENTER, 'test-prompt', tmp_path)

        call_kwargs = mock_iwcr.call_args.kwargs
        assert call_kwargs.get('resume_session_id') is None, (
            f'Expected resume_session_id=None (fresh session), '
            f'got {call_kwargs.get("resume_session_id")!r}'
        )
