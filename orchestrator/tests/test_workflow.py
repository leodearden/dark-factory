"""Workflow tests for the architect plan-tightening retry.

The plan-files-not-touched gate (``merge_queue._check_plan_files_touched_in_branch``)
refuses to merge when the architect's plan declares files the branch never
touched.  Before this fix, the caller short-circuited straight to L1 with
``escalate_to_human=True`` — busywork when the architect honestly
over-declared files that turned out not to be needed.

These tests pin the bounded one-shot retry: the architect gets a single
chance to ``update_plan_metadata(files=[narrowed list])``.  Lenient
semantics — only forbid NEW files; the gate's re-check is the source of
truth for pass/fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import PlanFilesTouchedResult
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState


def _make_workflow(*, tmp_path: Path, task_id: str = '2656') -> TaskWorkflow:
    """Minimal TaskWorkflow harness — mirrors test_workflow_worktree_missing."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'

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
    wf.plan = {'files': ['a.py', 'b.py', 'c.py']}
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    return wf


class _CheckSequence:
    """Helper: produce a sequence of PlanFilesTouchedResult on successive calls."""

    def __init__(self, results: list[PlanFilesTouchedResult]) -> None:
        self.results = list(results)
        self.calls = 0

    async def __call__(self, *args, **kwargs) -> PlanFilesTouchedResult:
        idx = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[idx]


@pytest.mark.asyncio
class TestSubmitToMergeQueuePlanTightening:
    """Architect plan-tightening retry inside ``_submit_to_merge_queue``."""

    async def test_narrow_succeeds_gate_passes_on_recheck(
        self, tmp_path: Path, monkeypatch,
    ):
        """Architect drops the genuinely-unneeded file; re-check passes;
        merge proceeds.  Verifies the ``plan_files_narrowed`` event fires
        and no L1 is submitted.
        """
        wf = _make_workflow(tmp_path=tmp_path)

        check_seq = _CheckSequence([
            PlanFilesTouchedResult(not_touched=['a.py']),  # first call
            PlanFilesTouchedResult(not_touched=[]),         # after narrowing
        ])
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            check_seq,
        )

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_head_sha\n', ''
        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        emits: list[str] = []

        def fake_emit(event_store, task_id, outcome, **kwargs):  # noqa: ARG001
            emits.append(outcome)

        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_attempt', fake_emit,
        )

        # Stub architect: write a narrowed plan to disk so read_plan() returns it.
        async def fake_invoke(role, prompt, cwd, **_kw):  # noqa: ARG001
            assert wf.artifacts is not None
            wf.artifacts.write_plan({'files': ['b.py', 'c.py']})
            return AgentResult(
                success=True, output='', cost_usd=0.0, turns=1,
            )
        wf._invoke = fake_invoke  # type: ignore[method-assign]

        # Stub briefing helper used by _try_narrow_plan.
        wf.briefing.build_plan_tightening_prompt = AsyncMock(return_value='PROMPT')

        # Stub enqueue: resolve future with a 'merged' outcome so we exit the
        # happy path quickly.  We only assert the event emit list / no-L1,
        # not the post-merge SHA pipeline.
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        async def fake_enqueue(queue, req: MergeRequest, event_store):  # noqa: ARG001
            req.result.set_result(MergeOutcome('done', merge_sha='deadbeef'))
        monkeypatch.setattr(
            'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
        )

        mark_blocked = AsyncMock()
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        # Stub _finalise_merged_done so success path returns DONE cleanly.
        wf._finalise_merged_done = AsyncMock(  # type: ignore[method-assign]
            return_value=WorkflowOutcome.DONE,
        )

        outcome = await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert outcome == WorkflowOutcome.DONE
        assert check_seq.calls == 2, (
            f'gate must re-check after narrowing; got {check_seq.calls} calls'
        )
        assert 'plan_files_narrowed' in emits
        assert 'plan_files_not_touched' not in emits
        mark_blocked.assert_not_awaited()

    async def test_architect_refuses_to_narrow_falls_through_to_l1(
        self, tmp_path: Path, monkeypatch,
    ):
        """Architect returns plan unchanged; re-check still flags the
        same not_touched file; workflow escalates to L1.
        """
        wf = _make_workflow(tmp_path=tmp_path)

        check_seq = _CheckSequence([
            PlanFilesTouchedResult(not_touched=['a.py']),
            PlanFilesTouchedResult(not_touched=['a.py']),  # unchanged after narrow
        ])
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            check_seq,
        )

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_head_sha\n', ''
        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        emits: list[str] = []

        def fake_emit(event_store, task_id, outcome, **kwargs):  # noqa: ARG001
            emits.append(outcome)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_attempt', fake_emit,
        )

        # Architect leaves plan.json untouched.
        async def fake_invoke(role, prompt, cwd, **_kw):  # noqa: ARG001
            assert wf.artifacts is not None
            wf.artifacts.write_plan({'files': ['a.py', 'b.py', 'c.py']})
            return AgentResult(
                success=True, output='', cost_usd=0.0, turns=1,
            )
        wf._invoke = fake_invoke  # type: ignore[method-assign]
        wf.briefing.build_plan_tightening_prompt = AsyncMock(return_value='PROMPT')

        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert outcome == WorkflowOutcome.BLOCKED
        assert check_seq.calls == 2
        mark_blocked.assert_awaited_once()
        # Caller must pass escalate_to_human=True.
        _, kwargs = mark_blocked.call_args
        assert kwargs.get('escalate_to_human') is True
        assert 'plan_files_not_touched' in emits

    async def test_architect_adds_new_file_rejected_no_recheck(
        self, tmp_path: Path, monkeypatch,
    ):
        """Architect drops A but introduces a new file Z; the subset check
        rejects the narrowing pass.  Workflow does NOT re-check the gate
        (narrowed==False) and falls through to L1 on the original
        not_touched=[a.py].
        """
        wf = _make_workflow(tmp_path=tmp_path)

        check_seq = _CheckSequence([
            PlanFilesTouchedResult(not_touched=['a.py']),
        ])
        monkeypatch.setattr(
            'orchestrator.merge_queue._check_plan_files_touched_in_branch',
            check_seq,
        )

        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_head_sha\n', ''
        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        emits: list[str] = []

        def fake_emit(event_store, task_id, outcome, **kwargs):  # noqa: ARG001
            emits.append(outcome)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_attempt', fake_emit,
        )

        async def fake_invoke(role, prompt, cwd, **_kw):  # noqa: ARG001
            assert wf.artifacts is not None
            wf.artifacts.write_plan({'files': ['b.py', 'c.py', 'z.py']})
            return AgentResult(
                success=True, output='', cost_usd=0.0, turns=1,
            )
        wf._invoke = fake_invoke  # type: ignore[method-assign]
        wf.briefing.build_plan_tightening_prompt = AsyncMock(return_value='PROMPT')

        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = await wf._submit_to_merge_queue('task/2656', pre_rebased=False)

        assert outcome == WorkflowOutcome.BLOCKED
        # Single call: narrow rejected → no re-check.
        assert check_seq.calls == 1
        mark_blocked.assert_awaited_once()
        _, kwargs = mark_blocked.call_args
        assert kwargs.get('escalate_to_human') is True
        assert 'plan_files_not_touched' in emits

    async def test_one_shot_guard_skips_architect_on_second_call(
        self, tmp_path: Path,
    ):
        """``_try_narrow_plan`` is one-shot: the second invocation must
        return False without re-invoking the architect.  Protects against
        loops if the workflow ever re-enters _submit_to_merge_queue on
        the same TaskWorkflow instance.
        """
        wf = _make_workflow(tmp_path=tmp_path)

        invocations: list[int] = []

        async def fake_invoke(role, prompt, cwd, **_kw):  # noqa: ARG001
            invocations.append(1)
            assert wf.artifacts is not None
            wf.artifacts.write_plan({'files': ['b.py', 'c.py']})
            return AgentResult(
                success=True, output='', cost_usd=0.0, turns=1,
            )
        wf._invoke = fake_invoke  # type: ignore[method-assign]
        wf.briefing.build_plan_tightening_prompt = AsyncMock(return_value='PROMPT')

        first = await wf._try_narrow_plan(['a.py'])
        second = await wf._try_narrow_plan(['a.py'])

        assert first is True
        assert second is False
        # Architect invoked exactly once across both calls.
        assert len(invocations) == 1


_PASSING_VERIFY_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all green',
)


@pytest.mark.asyncio
class TestVerifyDebugfixLoopForceWorkspace:
    """_verify_debugfix_loop forwards force_workspace based on train membership (γ₁).

    When self._train is not None the call to run_scoped_verification must receive
    force_workspace=True so the workspace-wide command runs.  Non-train tasks must
    receive force_workspace=False — byte-identical existing behaviour.
    """

    def _setup_wf(self, tmp_path: Path) -> TaskWorkflow:
        wf = _make_workflow(tmp_path=tmp_path)
        # Point _task_files (property reads from self.plan['files']) at a known file.
        wf.plan['files'] = ['orchestrator/x.py']
        # Skip the pre-verify rebase step to keep the loop tight.
        wf.config.rebase_before_verify = False
        return wf

    async def test_train_member_passes_force_workspace_true(
        self, tmp_path: Path, monkeypatch,
    ):
        """A task with metadata.train set must call run_scoped_verification with
        force_workspace=True so the workspace-wide command runs instead of the
        per-module scoped command.
        """
        wf = self._setup_wf(tmp_path)
        wf.task['metadata'] = {'train': {'id': 'T1', 'order': 0, 'members': ['1523']}}

        spy = AsyncMock(return_value=_PASSING_VERIFY_RESULT)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', spy)

        outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE
        spy.assert_awaited_once()
        _, kwargs = spy.call_args
        assert kwargs.get('force_workspace') is True, (
            f'Expected force_workspace=True for train member; got call_args={spy.call_args}'
        )

    async def test_non_train_task_passes_force_workspace_false(
        self, tmp_path: Path, monkeypatch,
    ):
        """A task without metadata.train must call run_scoped_verification with
        force_workspace=False — byte-identical behaviour to the pre-γ₁ baseline.
        """
        wf = self._setup_wf(tmp_path)
        # No train metadata — plain task.

        spy = AsyncMock(return_value=_PASSING_VERIFY_RESULT)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', spy)

        outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE
        spy.assert_awaited_once()
        _, kwargs = spy.call_args
        assert kwargs.get('force_workspace') is False, (
            f'Expected force_workspace=False for non-train task; got call_args={spy.call_args}'
        )


@pytest.mark.asyncio
class TestEnterMergeDeferred:
    """_enter_merge_deferred parks the task in the merge-deferred holding state (γ₁).

    The helper must:
    1. Return WorkflowOutcome.MERGE_DEFERRED.
    2. Await scheduler.set_task_status(task_id, 'merge-deferred') exactly once.
    """

    async def test_returns_merge_deferred_outcome_and_calls_scheduler(
        self, tmp_path: Path,
    ):
        """_enter_merge_deferred parks the train member and returns MERGE_DEFERRED."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.task['metadata'] = {'train': {'id': 'T1', 'order': 1, 'members': ['a', '1523']}}
        wf.scheduler.set_task_status = AsyncMock()
        wf.scheduler.clear_requeue_count = MagicMock()

        result = await wf._enter_merge_deferred()

        assert result == WorkflowOutcome.MERGE_DEFERRED, (
            f'Expected WorkflowOutcome.MERGE_DEFERRED; got {result!r}'
        )
        wf.scheduler.set_task_status.assert_awaited_once_with(wf.task_id, 'merge-deferred')
        assert wf.state == WorkflowState.MERGE_DEFERRED, (
            f'Expected wf.state==MERGE_DEFERRED (worktree preserved); got {wf.state!r}'
        )
        # Requeue counter must be cleared: the task is workspace-green; any
        # counter accumulated from prior failed attempts is no longer relevant
        # (harness MERGE_DEFERRED falls through with requeued=False and won't
        # call clear_requeue_count itself).
        wf.scheduler.clear_requeue_count.assert_called_once_with(wf.task_id)


def _make_run_wf(
    tmp_path: Path,
    *,
    with_train: bool = False,
) -> TaskWorkflow:
    """Build a TaskWorkflow stubbed to reach the post-loop merge block in run().

    Stubs the preamble (setup/plan) and the merge helpers so the tests can
    assert on the train guard without running git or the real merge pipeline.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    if with_train:
        wf.task['metadata'] = {'train': {'id': 'T1', 'order': 0, 'members': ['1523']}}

    # Preamble stubs
    wf._setup_worktree_and_artifacts = AsyncMock()  # type: ignore[method-assign]
    wf._recover_if_already_merged = AsyncMock(return_value=None)  # type: ignore[method-assign]
    wf.config.simple_task_enabled = False
    # wf.plan is already set by _make_workflow → skips the planner entirely.
    wf._check_branch_on_main = AsyncMock(return_value=None)  # type: ignore[method-assign]
    wf._execute_verify_review_loop = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.DONE,
    )
    wf._worktree_external = False

    # Merge-path stubs (train members never reach these; non-train do)
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf.git_ops.get_main_sha = AsyncMock(return_value='deadbeef00')
    wf.git_ops.is_ancestor = AsyncMock(return_value=False)  # not already merged
    wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)
    wf.config.max_merge_retries = 1
    wf.config.max_pre_merge_retries = 1

    # SUCCESS-block stubs (after _submit_to_merge_queue returns DONE)
    wf._write_completion_to_memory = AsyncMock()  # type: ignore[method-assign]
    wf._ensure_steward_started = AsyncMock()  # type: ignore[method-assign]
    wf._await_steward_completion = AsyncMock()  # type: ignore[method-assign]
    wf._finalise_merged_done = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.DONE,
    )

    # Train-guard target + merge-queue (only one of these is called per case)
    wf._enter_merge_deferred = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.MERGE_DEFERRED,
    )
    wf._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.DONE,
    )

    return wf


@pytest.mark.asyncio
class TestRunMergeDeferredGuard:
    """run() routes train members to merge-deferred instead of the merge phase (γ₁).

    The train guard fires immediately after `if not self._worktree_external:` in
    the post-loop merge block, before any git calls or merge-queue submission.
    Non-train tasks must see byte-identical behaviour.
    """

    async def test_train_member_parks_in_merge_deferred(
        self, tmp_path: Path, monkeypatch,
    ):
        """A task with metadata.train set must return MERGE_DEFERRED without
        submitting to the merge queue.  The execute/verify/review pipeline
        still runs (stubbed via _execute_verify_review_loop).
        """
        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_sha\n', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=_PASSING_VERIFY_RESULT),
        )

        wf = _make_run_wf(tmp_path, with_train=True)
        outcome = await wf.run()

        assert outcome == WorkflowOutcome.MERGE_DEFERRED, (
            f'Expected MERGE_DEFERRED for train member; got {outcome!r}'
        )
        cast(AsyncMock, wf._enter_merge_deferred).assert_awaited_once()
        cast(AsyncMock, wf._submit_to_merge_queue).assert_not_called()

    async def test_non_train_task_proceeds_to_merge(
        self, tmp_path: Path, monkeypatch,
    ):
        """A task without metadata.train must NOT call _enter_merge_deferred and
        must proceed to _submit_to_merge_queue — the pre-γ₁ merge path unchanged.
        """
        async def fake_run(cmd, **kwargs):  # noqa: ARG001
            return 0, 'fake_sha\n', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=_PASSING_VERIFY_RESULT),
        )

        wf = _make_run_wf(tmp_path, with_train=False)
        outcome = await wf.run()

        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE for non-train task merge path; got {outcome!r}'
        )
        cast(AsyncMock, wf._enter_merge_deferred).assert_not_called()
        cast(AsyncMock, wf._submit_to_merge_queue).assert_awaited_once()
