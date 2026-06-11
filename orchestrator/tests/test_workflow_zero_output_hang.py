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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from shared.config_dir import TaskConfigDir

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


# ---------------------------------------------------------------------------
# Step-9 RED test: _capture_zero_output_evidence
# ---------------------------------------------------------------------------


class TestCaptureZeroOutputEvidence:
    """_capture_zero_output_evidence writes a JSON evidence file under artifacts.root.

    Fails RED because _capture_zero_output_evidence does not exist yet (step-10
    will implement it).
    """

    def test_evidence_file_created_under_artifacts_root(self, tmp_path: Path):
        """Evidence JSON file is written at artifacts.root/zero_output_evidence-iter{N}.json.

        Builds a workflow with a real TaskConfigDir containing sentinel files, calls
        _capture_zero_output_evidence, and asserts the file exists and contains the
        expected fields: stderr_tail, proc_tree, config_dir path, config_dir_listing
        with the sentinel files.
        """
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None

        # Build a real TaskConfigDir with sentinel files
        config_dir = TaskConfigDir(wf.task_id, base_dir=wf.artifacts.root)
        sentinel_a = config_dir.path / 'sentinel_a.txt'
        sentinel_a.write_text('hello')
        sentinel_b = config_dir.path / 'sentinel_b.json'
        sentinel_b.write_text('{}')
        wf._config_dir = config_dir  # type: ignore[attr-defined]

        # AgentResult carrying evidence fields
        result = AgentResult(
            success=False,
            output='Agent produced no output',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            stderr='boom-tail error log line\n',
            proc_tree='snapshot_process_group(99999): 1 process(es) in group:\n  pid=99999 ppid=1 state=D wchan=hrtimer_nanosleep comm=sleep\n',
        )

        iteration = 3
        wf._capture_zero_output_evidence(result, iteration)  # type: ignore[attr-defined]

        evidence_path = wf.artifacts.root / f'zero_output_evidence-iter{iteration}.json'
        assert evidence_path.exists(), (
            f'Expected evidence file at {evidence_path} but it does not exist'
        )

        data = json.loads(evidence_path.read_text())

        # Core scalar fields
        assert data.get('iteration') == iteration, f'iteration mismatch: {data!r}'
        assert data.get('timed_out') is True, f'timed_out mismatch: {data!r}'
        assert data.get('turns') == 0, f'turns mismatch: {data!r}'

        # stderr_tail and proc_tree propagated
        assert 'boom-tail' in data.get('stderr_tail', ''), (
            f'Expected "boom-tail" in stderr_tail: {data!r}'
        )
        assert 'hrtimer_nanosleep' in data.get('proc_tree', ''), (
            f'Expected proc_tree content in evidence: {data!r}'
        )

        # Config dir path and listing
        assert data.get('config_dir'), f'Expected config_dir path in evidence: {data!r}'
        listing = data.get('config_dir_listing', [])
        assert listing, f'Expected non-empty config_dir_listing in evidence: {data!r}'
        listing_str = str(listing)
        assert 'sentinel_a.txt' in listing_str or any(
            'sentinel_a' in entry for entry in listing
        ), f'Expected sentinel_a.txt in config_dir_listing: {listing!r}'


# ---------------------------------------------------------------------------
# Step-11 RED tests: _cleanup_config_dir
# ---------------------------------------------------------------------------


class TestCleanupConfigDir:
    """_cleanup_config_dir is a preserve-aware wrapper around config_dir.cleanup().

    Case A: _preserve_config_dir=True  → cleanup is SKIPPED  → path still exists.
    Case B: _preserve_config_dir=False → cleanup runs         → path is gone.
    Case C: _config_dir is None        → no-op, no exception.

    All tests fail RED because _cleanup_config_dir does not exist yet (step-12
    implements it).
    """

    def test_preserve_flag_true_skips_cleanup(self, tmp_path: Path):
        """When _preserve_config_dir=True, _cleanup_config_dir() must NOT delete the dir."""
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None
        config_dir = TaskConfigDir(wf.task_id, base_dir=wf.artifacts.root)
        sentinel = config_dir.path / 'keep_me.txt'
        sentinel.write_text('preserve test')
        wf._config_dir = config_dir  # type: ignore[attr-defined]
        wf._preserve_config_dir = True  # type: ignore[attr-defined]

        wf._cleanup_config_dir()  # type: ignore[attr-defined]

        assert config_dir.path.exists(), (
            f'_cleanup_config_dir() must skip cleanup when _preserve_config_dir=True; '
            f'path was deleted: {config_dir.path}'
        )
        assert sentinel.exists(), 'Sentinel file must survive when cleanup is skipped'

    def test_preserve_flag_false_performs_cleanup(self, tmp_path: Path):
        """When _preserve_config_dir=False, _cleanup_config_dir() deletes the dir."""
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None
        config_dir = TaskConfigDir(wf.task_id, base_dir=wf.artifacts.root)
        sentinel = config_dir.path / 'remove_me.txt'
        sentinel.write_text('cleanup test')
        wf._config_dir = config_dir  # type: ignore[attr-defined]
        wf._preserve_config_dir = False  # type: ignore[attr-defined]

        wf._cleanup_config_dir()  # type: ignore[attr-defined]

        assert not config_dir.path.exists(), (
            f'_cleanup_config_dir() must remove config dir when _preserve_config_dir=False; '
            f'path still exists: {config_dir.path}'
        )

    def test_none_config_dir_is_noop(self, tmp_path: Path):
        """When _config_dir is None, _cleanup_config_dir() is a no-op and never raises."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf._config_dir = None  # type: ignore[attr-defined]
        wf._preserve_config_dir = False  # type: ignore[attr-defined]

        # Must not raise
        wf._cleanup_config_dir()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Step-13 RED tests: _recycle_config_dir + loop wiring
# ---------------------------------------------------------------------------


class TestRecycleConfigDir:
    """_recycle_config_dir() cleans the current TaskConfigDir and creates a fresh one.

    Unit case: sentinel file is gone after recycle; self._config_dir is a new
    instance pointing to a fresh (now-empty) directory.

    Loop wiring cases: with recycle_config_dir_on_zero_output=True,
    _recycle_config_dir is called exactly threshold-1 times (sub-threshold only);
    with recycle_config_dir_on_zero_output=False it is never called.

    All tests fail RED because _recycle_config_dir does not exist yet (step-14
    implements it).
    """

    def test_recycle_removes_sentinel_and_creates_fresh_dir(self, tmp_path: Path):
        """After _recycle_config_dir(), the old sentinel is gone and self._config_dir
        points to a fresh (existing) TaskConfigDir directory.
        """
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None
        old_config_dir = TaskConfigDir(wf.task_id, base_dir=wf.artifacts.root)
        sentinel = old_config_dir.path / 'old_sentinel.txt'
        sentinel.write_text('stale session state')
        old_path = old_config_dir.path
        wf._config_dir = old_config_dir  # type: ignore[attr-defined]

        wf._recycle_config_dir()  # type: ignore[attr-defined]

        # Old sentinel must be gone (old dir cleaned up)
        assert not sentinel.exists(), (
            f'Sentinel must be removed after recycle, but still exists: {sentinel}'
        )

        # self._config_dir must be a new instance
        new_config_dir = wf._config_dir  # type: ignore[attr-defined]
        assert new_config_dir is not old_config_dir, (
            'self._config_dir must be replaced with a new TaskConfigDir instance'
        )

        # New dir must exist (TaskConfigDir creates it on __init__)
        assert new_config_dir.path.exists(), (
            f'New config dir must exist after recycle: {new_config_dir.path}'
        )

        # Both must share the same task_id in the directory name
        assert wf.task_id in str(new_config_dir.path), (
            f'New config dir path must contain task_id {wf.task_id!r}: '
            f'{new_config_dir.path}'
        )

        # Old path must differ from new (or at minimum the sentinel is gone)
        # TaskConfigDir uses a fixed name derived from task_id so paths match —
        # the important invariant is that the OLD CONTENTS are gone.
        assert not old_path.exists() or not sentinel.exists(), (
            'Old sentinel must be gone after recycle'
        )


@pytest.mark.asyncio
class TestRecycleConfigDirLoopWiring:
    """_recycle_config_dir is called on sub-threshold zero-output timeouts when
    recycle_config_dir_on_zero_output=True, and NOT called on the tripping iteration
    or when the flag is False.

    threshold=3 → sub-threshold iterations are 1 and 2 (2 recycles expected),
    iteration 3 trips the breaker (no recycle on trip).
    """

    async def test_recycle_called_on_subthreshold_iterations(self, tmp_path: Path):
        """With threshold=3 and recycle=True, _recycle_config_dir is called exactly
        threshold-1 = 2 times (on sub-threshold timeouts), NOT on the tripping one.
        """
        from unittest.mock import patch

        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=3,
            recycle_config_dir_on_zero_output=True,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        with patch.object(wf, '_recycle_config_dir') as mock_recycle:
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        # 3 zero-output calls: call 1 → recycle, call 2 → recycle, call 3 → trip (no recycle)
        assert mock_recycle.call_count == 2, (
            f'Expected 2 recycle calls (threshold-1), got {mock_recycle.call_count}'
        )

    async def test_recycle_not_called_when_flag_false(self, tmp_path: Path):
        """With recycle_config_dir_on_zero_output=False, _recycle_config_dir is
        never called regardless of how many zero-output timeouts occur.
        """
        from unittest.mock import patch

        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
            recycle_config_dir_on_zero_output=False,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        with patch.object(wf, '_recycle_config_dir') as mock_recycle:
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert mock_recycle.call_count == 0, (
            f'Expected 0 recycle calls when flag=False, got {mock_recycle.call_count}'
        )


# ---------------------------------------------------------------------------
# Amendment tests: evidence-file loop wiring (Suggestion 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCaptureEvidenceLoopWiring:
    """_capture_zero_output_evidence is called on EVERY zero-output timeout inside
    _execute_iterations (before the threshold check), so distinct per-iteration
    evidence files are written for both sub-threshold and tripping occurrences.

    This verifies the wiring between the circuit-breaker counter and the evidence
    helper — the property explicitly called out in the design: 'captured on EVERY
    zero-output timeout, before the threshold check'.
    """

    async def test_evidence_files_written_for_every_zero_output_timeout(
        self, tmp_path: Path
    ):
        """With threshold=2, two zero-output timeouts must produce two distinct
        evidence files (zero_output_evidence-iter1.json and iter2.json) under
        artifacts.root.

        Evidence files are created before the circuit breaker trips so that even
        the sub-threshold timeout (iteration 1) leaves a forensic record.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=10,
            max_consecutive_zero_output_timeouts=2,
            recycle_config_dir_on_zero_output=False,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())
        # Wire up a real TaskConfigDir so _capture_zero_output_evidence can write
        from shared.config_dir import TaskConfigDir as _TCD
        wf._config_dir = _TCD(wf.task_id, base_dir=wf.artifacts.root)  # type: ignore[attr-defined]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert wf.artifacts is not None

        # Two iterations ran (threshold=2), so two evidence files must exist.
        iter1_path = wf.artifacts.root / 'zero_output_evidence-iter1.json'
        iter2_path = wf.artifacts.root / 'zero_output_evidence-iter2.json'
        assert iter1_path.exists(), (
            f'Evidence file for iteration 1 missing: {iter1_path}\n'
            f'artifacts.root contents: {list(wf.artifacts.root.iterdir())}'
        )
        assert iter2_path.exists(), (
            f'Evidence file for iteration 2 missing: {iter2_path}\n'
            f'artifacts.root contents: {list(wf.artifacts.root.iterdir())}'
        )

        # Each file must contain valid JSON with iteration numbers matching
        data1 = json.loads(iter1_path.read_text())
        data2 = json.loads(iter2_path.read_text())
        assert data1.get('iteration') == 1, f'iter1 evidence has wrong iteration: {data1!r}'
        assert data2.get('iteration') == 2, f'iter2 evidence has wrong iteration: {data2!r}'

        # Both must record a zero-output timeout
        assert data1.get('timed_out') is True
        assert data2.get('timed_out') is True
