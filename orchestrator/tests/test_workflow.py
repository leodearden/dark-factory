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

        async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):  # noqa: ARG001
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
        # δ₂: _maybe_enqueue_group_merge calls tasks_by_train; return [] so
        # the trigger parks (all members not deferred) → MERGE_DEFERRED path.
        wf.scheduler.tasks_by_train = AsyncMock(return_value=[])

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


# ---------------------------------------------------------------------------
# TestWorkflowMergeInflightRegistry — merge_inflight_registry wiring in __init__
# ---------------------------------------------------------------------------


class TestWorkflowMergeInflightRegistry:
    """Tests for the merge_inflight_registry parameter in TaskWorkflow.__init__."""

    def test_registry_stored_when_provided(self, tmp_path: Path):
        """Constructor stores the injected registry on self.merge_inflight_registry."""
        from orchestrator.merge_queue import InFlightMergeRegistry
        sentinel = InFlightMergeRegistry()
        _make_workflow(tmp_path=tmp_path)
        # Directly re-construct with the kwarg to check storage
        assignment = MagicMock()
        assignment.task_id = '999'
        assignment.task = {'id': '999', 'title': 'T', 'description': 'd'}
        assignment.modules = []
        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wf2 = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_inflight_registry=sentinel,
        )
        assert wf2.merge_inflight_registry is sentinel

    def test_registry_defaults_to_none(self, tmp_path: Path):
        """Without the kwarg, merge_inflight_registry is None."""
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.merge_inflight_registry is None


@pytest.mark.asyncio
class TestSubmitToMergeQueueRegistersInRegistry:
    """_submit_to_merge_queue registers its branch in the injected registry."""

    async def test_branch_is_inflight_at_dequeue_time(self, tmp_path: Path, monkeypatch):
        """Branch is registered in the registry before the merge worker dequeues.

        The workflow-path enqueue must call register_and_enqueue_merge_request
        (not plain enqueue_merge_request) so the registry slot is held while
        the request sits in the queue.  A background worker captures
        is_inflight at dequeue time; the test asserts it was True.

        RED until step-8 impl switches the enqueue call.
        """
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}          # empty → _task_files=None → gate skipped
        wf._base_commit = None
        wf._module_configs = []

        inflight_seen: list[bool] = []

        async def _worker():
            req = await real_queue.get()
            inflight_seen.append(registry.is_inflight(req.branch))
            req.result.set_result(MergeOutcome(status='done', merge_sha='sha'))

        worker_task = asyncio.create_task(_worker())
        outcome = await wf._submit_to_merge_queue('B')
        await worker_task

        assert inflight_seen == [True], (
            f'Branch must be registered in registry at dequeue time; '
            f'inflight_seen={inflight_seen}'
        )
        assert outcome == WorkflowOutcome.DONE


@pytest.mark.asyncio
class TestSubmitToMergeQueueAttachesAsPeer:
    """_submit_to_merge_queue attaches as a peer waiter when an entry is in-flight."""

    def _make_attach_workflow(self, tmp_path, real_queue, registry):
        """Build a workflow wired to real_queue + registry (no plan gate)."""
        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []
        return wf

    async def test_peer_completion_no_duplicate_enqueue(
        self, tmp_path, monkeypatch,
    ):
        """Workflow attaches as peer, primary resolve flows through, no enqueue."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        TIP = 'abc123def456abc1'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        wf = self._make_attach_workflow(tmp_path, real_queue, registry)

        # Stub _run so rev-parse returns TIP (same as snapshot → SAME relation)
        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        coalesced_calls: list[int] = []
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: coalesced_calls.append(1),
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        # Let the task run to its blocking await
        for _ in range(10):
            await asyncio.sleep(0)

        # (1) No duplicate enqueue
        assert real_queue.qsize() == 0, (
            f'workflow must not enqueue a duplicate; qsize={real_queue.qsize()}'
        )

        # (2) Two waiters; 2nd belongs to the workflow
        entry = registry.entry('B')
        assert entry is not None
        assert len(entry.waiters) == 2
        assert entry.waiters[1].source == 'workflow'

        # (3) Resolving the primary fans the outcome to the workflow's waiter
        P.set_result(MergeOutcome(status='done', merge_sha='sha'))
        outcome = await submit_task

        assert outcome == WorkflowOutcome.DONE

        # (4) merge_coalesced event emitted
        assert len(coalesced_calls) == 1

    async def test_superset_tip_resnapshots_before_attach(
        self, tmp_path, monkeypatch,
    ):
        """SUPERSET new tip → re_snapshot before attach; no classify → RED."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        OLD = 'oldtip000000000001'
        NEW = 'newtip000000000001'

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=OLD, snapshot_tip=OLD,
        )

        wf = self._make_attach_workflow(tmp_path, real_queue, registry)
        # Stub is_ancestor: OLD is ancestor of NEW → NEW is SUPERSET
        wf.git_ops.is_ancestor = AsyncMock(side_effect=lambda a, b: a == OLD and b == NEW)

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, NEW + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        re_snapshot_calls: list[tuple] = []
        real_re_snapshot = registry.re_snapshot
        registry.re_snapshot = lambda branch, tip: (  # type: ignore[method-assign]
            re_snapshot_calls.append((branch, tip)) or real_re_snapshot(branch, tip)
        )

        # Pin this test to the verifying=False (RESNAPSHOT) path.
        _pre_entry = registry.entry('B')
        assert _pre_entry is not None and _pre_entry.verifying is False, (
            'test_superset_tip_resnapshots_before_attach exercises the verifying=False '
            'RESNAPSHOT path; set_verifying must NOT be called here'
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        # re_snapshot must be called before attach
        assert ('B', NEW) in re_snapshot_calls, (
            f're_snapshot not called with (B, {NEW!r}); calls={re_snapshot_calls}'
        )
        # Waiter is still attached
        _entry = registry.entry('B')
        assert _entry is not None
        assert len(_entry.waiters) == 2

        P.set_result(MergeOutcome(status='done', merge_sha='sha'))
        await submit_task

    async def test_attach_and_chain_reenqueues_superset_delta(
        self, tmp_path, monkeypatch,
    ):
        """ATTACH_AND_CHAIN (verifying=True + SUPERSET) → independent re-enqueue, no peer-attach.

        RED today: the code logs a warning then peer-attaches (qsize stays 0, waiters==2,
        merge_coalesced IS emitted).  GREEN after the fix leaves attached=False so control
        falls through to register_and_enqueue_merge_request (qsize==1, waiters==1, no event).
        """
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        OLD = 'oldtip000000000002'
        NEW = 'newtip000000000002'

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=OLD, snapshot_tip=OLD,
        )
        # Flip verifying=True so decide_attach_action maps SUPERSET → ATTACH_AND_CHAIN
        registry.set_verifying('B', True)

        wf = self._make_attach_workflow(tmp_path, real_queue, registry)
        # Stub is_ancestor: OLD is ancestor of NEW → classify_tip_relation returns SUPERSET
        wf.git_ops.is_ancestor = AsyncMock(side_effect=lambda a, b: a == OLD and b == NEW)

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, NEW + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)

        coalesced_calls: list[int] = []
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: coalesced_calls.append(1),
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        # (1) Must re-enqueue independently — NOT attach as a peer
        assert real_queue.qsize() == 1, (
            f'ATTACH_AND_CHAIN must enqueue independently; qsize={real_queue.qsize()}'
        )

        # (2) Primary entry must NOT have gained a second (peer) waiter
        _entry = registry.entry('B')
        assert _entry is not None
        assert len(_entry.waiters) == 1, (
            f'no peer waiter must be added for ATTACH_AND_CHAIN; '
            f'waiters={_entry.waiters!r}'
        )

        # (3) No merge_coalesced event for an independent re-enqueue
        assert len(coalesced_calls) == 0, (
            f'_emit_merge_coalesced must not be called for ATTACH_AND_CHAIN; '
            f'calls={coalesced_calls}'
        )

        # (4) The independently enqueued request is for branch 'B'
        enqueued_req = real_queue.get_nowait()
        assert enqueued_req.branch == 'B'

        # Complete the workflow: resolve the independently enqueued request's own future.
        # (The primary P is unresolved — the workflow must be awaiting its OWN future.)
        enqueued_req.result.set_result(MergeOutcome(status='done', merge_sha='sha'))
        outcome = await submit_task
        assert outcome == WorkflowOutcome.DONE

    async def test_divergent_tip_resolves_and_attaches(
        self, tmp_path, monkeypatch,
    ):
        """DIVERGENT tip → resolve_divergent yields SUBSET → attach without re_snapshot."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        OLD = 'divold000000000001'
        NEW = 'divnew000000000001'

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=OLD, snapshot_tip=OLD,
        )

        wf = self._make_attach_workflow(tmp_path, real_queue, registry)
        # Neither is ancestor → DIVERGENT
        wf.git_ops.is_ancestor = AsyncMock(return_value=False)
        wf.git_ops.project_root = tmp_path

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, NEW + '\n', ''
            if 'cherry' in cmd:
                # no '+' lines → all commits in NEW are in OLD → SUBSET
                return 0, '- abc\n- def\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr('orchestrator.merge_queue._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        # Attached (ATTACH_CONTAINMENT action for SUBSET); no ValueError escaped
        assert real_queue.qsize() == 0
        entry = registry.entry('B')
        assert entry is not None
        assert len(entry.waiters) == 2

        P.set_result(MergeOutcome(status='done', merge_sha='sha'))
        await submit_task

    async def test_attach_race_fallback_enqueues(
        self, tmp_path, monkeypatch,
    ):
        """attach() returning False (entry released mid-await) → enqueue fallback."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        TIP = 'racetip00000000001'
        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        wf = self._make_attach_workflow(tmp_path, real_queue, registry)
        wf.git_ops.is_ancestor = AsyncMock(return_value=False)

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            if 'cherry' in cmd:
                return 0, '- abc\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr('orchestrator.merge_queue._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        # Force attach() to return False (entry released during classify await)
        registry.attach = lambda branch, waiter: False  # type: ignore[method-assign]

        async def _worker():
            req = await real_queue.get()
            req.result.set_result(MergeOutcome(status='done', merge_sha='sha'))

        worker_task = asyncio.create_task(_worker())
        outcome = await wf._submit_to_merge_queue('B', merge_phase=True)
        await worker_task

        assert outcome == WorkflowOutcome.DONE
        assert real_queue.qsize() == 0  # worker consumed it


@pytest.mark.asyncio
class TestAttachedWaiterOutcomeMapping:
    """I8 guard: conflict/blocked from primary flows through existing outcome mapping."""

    async def _setup_attached_workflow(self, tmp_path, monkeypatch):
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        TIP = 'guardtip0000000001'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        return wf, P, registry

    async def test_conflict_outcome_routes_to_resolve_and_resubmit(
        self, tmp_path, monkeypatch,
    ):
        """Conflict from primary → _resolve_and_resubmit (as if own merge failed)."""
        import asyncio

        from orchestrator.merge_queue import MergeOutcome

        wf, P, registry = await self._setup_attached_workflow(tmp_path, monkeypatch)

        resolve_calls: list[tuple] = []
        wf._resolve_and_resubmit = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda branch, details, **kw: (
                resolve_calls.append((branch, details)) or WorkflowOutcome.BLOCKED
            ),
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        P.set_result(MergeOutcome(status='conflict', conflict_details='conflict_X'))
        await submit_task

        assert resolve_calls == [('B', 'conflict_X')]

    async def test_blocked_outcome_routes_to_mark_blocked(
        self, tmp_path, monkeypatch,
    ):
        """Blocked from primary → _mark_blocked (as if own merge failed)."""
        import asyncio

        from orchestrator.merge_queue import MergeOutcome

        wf, P, registry = await self._setup_attached_workflow(tmp_path, monkeypatch)

        wf._write_merge_failure_review = MagicMock()
        mark_calls: list[str] = []
        wf._mark_blocked = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda reason, **kw: (
                mark_calls.append(reason) or WorkflowOutcome.BLOCKED
            ),
        )

        submit_task = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        P.set_result(MergeOutcome(status='blocked', reason='verification failed'))
        await submit_task

        assert len(mark_calls) == 1
        assert 'verification failed' in mark_calls[0]


@pytest.mark.asyncio
class TestSubmitToMergeQueueSoftCancelDetaches:
    """Soft-cancel detaches the workflow waiter instead of cancelling the primary."""

    async def test_soft_cancel_detaches_waiter_primary_lives(
        self, tmp_path, monkeypatch,
    ):
        """Soft-cancel removes workflow waiter; primary stays alive for the MCP waiter."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        TIP = 'tip000000000000001'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        scheduler = MagicMock()
        scheduler.get_status = AsyncMock(return_value='in-progress')

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        # Pre-set cancel_event so the workflow soft-cancels after attaching
        wf._cancel_event.set()

        outcome = await wf._submit_to_merge_queue('B', merge_phase=True)

        # (1) Workflow waiter was detached; only MCP waiter 'mr-mcp' remains
        entry = registry.entry('B')
        assert entry is not None, 'entry must still be in-flight (primary not cancelled)'
        assert len(entry.waiters) == 1
        assert entry.waiters[0].request_id == 'mr-mcp'

        # (2) Primary P is NOT cancelled
        assert not P.cancelled()

        # (3) Outcome is SOFT_CANCELLED (scheduler says in-progress + cancel_event set)
        assert outcome == WorkflowOutcome.SOFT_CANCELLED

    async def test_re_attach_coalesces_after_soft_cancel(
        self, tmp_path, monkeypatch,
    ):
        """A second _submit_to_merge_queue after soft-cancel re-attaches (2 waiters, no enqueue)."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        TIP = 'tip000000000000002'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        scheduler = MagicMock()
        scheduler.get_status = AsyncMock(return_value='in-progress')

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        # First call: soft-cancel → detach
        wf._cancel_event.set()
        await wf._submit_to_merge_queue('B', merge_phase=True)
        _entry = registry.entry('B')
        assert _entry is not None
        assert len(_entry.waiters) == 1  # detached

        # Second call: re-attach (cancel_event still set, but entry still in-flight)
        # Clear the cancel event so the second call can actually attach and await
        wf._cancel_event.clear()
        submit_task = asyncio.create_task(wf._submit_to_merge_queue('B', merge_phase=True))
        for _ in range(10):
            await asyncio.sleep(0)

        # Re-attached: 2 waiters again, no enqueue
        assert real_queue.qsize() == 0
        _entry2 = registry.entry('B')
        assert _entry2 is not None
        assert len(_entry2.waiters) == 2

        # Clean up: resolve P so the task can finish
        P.set_result(MergeOutcome(status='done', merge_sha='sha2'))
        await submit_task


@pytest.mark.asyncio
class TestAwaitCancellableSoftCancelHook:
    """_await_cancellable on_soft_cancel hook: detach instead of cancel."""

    async def test_hook_called_future_not_cancelled_on_cancel_win(
        self, tmp_path: Path,
    ):
        """When the cancel event wins the race, hook is called and future is NOT cancelled."""
        import asyncio

        wf = _make_workflow(tmp_path=tmp_path)
        wf._cancel_event.set()  # cancel wins immediately

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        hook_calls: list[int] = []

        result = await wf._await_cancellable(fut, on_soft_cancel=lambda: hook_calls.append(1))

        assert result is None
        assert hook_calls == [1], 'hook must be called exactly once'
        assert not fut.cancelled(), 'future must NOT be cancelled when hook is provided'

    async def test_no_hook_future_is_cancelled_on_cancel_win(
        self, tmp_path: Path,
    ):
        """Without a hook (default None), cancel-win still cancels the future."""
        import asyncio

        wf = _make_workflow(tmp_path=tmp_path)
        wf._cancel_event.set()

        fut: asyncio.Future = asyncio.get_event_loop().create_future()

        result = await wf._await_cancellable(fut)

        assert result is None
        assert fut.cancelled(), 'future must be cancelled when no hook is provided'

    async def test_hook_not_called_when_future_resolves_first(
        self, tmp_path: Path,
    ):
        """When the awaitable resolves first, hook is NOT called and its result is returned."""
        import asyncio

        wf = _make_workflow(tmp_path=tmp_path)
        # cancel_event is NOT set

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result('the_result')
        hook_calls: list[int] = []

        result = await wf._await_cancellable(fut, on_soft_cancel=lambda: hook_calls.append(1))

        assert result == 'the_result'
        assert hook_calls == [], 'hook must NOT be called when future resolves first'


@pytest.mark.asyncio
class TestGroupMergePathUnchangedByGamma3:
    """D9 guard: train group-merge soft-cancel cancels future; no registry.attach/detach."""

    async def test_group_merge_soft_cancel_cancels_future_not_detaches(
        self, tmp_path, monkeypatch,
    ):
        """GroupMergeRequest future is cancelled on soft-cancel; attach/detach never called."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        attach_calls: list = []
        detach_calls: list = []
        _real_attach = registry.attach
        _real_detach = registry.detach
        registry.attach = lambda *a, **kw: (  # type: ignore[method-assign]
            attach_calls.append(a) or _real_attach(*a, **kw)
        )
        registry.detach = lambda *a, **kw: (  # type: ignore[method-assign]
            detach_calls.append(a) or _real_detach(*a, **kw)
        )

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {
            'id': 'B', 'title': 'T', 'description': 'd',
            'metadata': {'train': {'id': 'T1', 'order': 1, 'members': ['A', 'B']}},
        }
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        scheduler = MagicMock()
        # Both members merge-deferred; 'B' is tip (order=1, highest)
        scheduler.tasks_by_train = AsyncMock(return_value=[
            {'id': 'A', 'status': 'merge-deferred'},
            {'id': 'B', 'status': 'merge-deferred'},
        ])
        scheduler.get_status = AsyncMock(return_value='merge-deferred')
        scheduler.get_statuses = AsyncMock(return_value=({'A': 'merge-deferred', 'B': 'merge-deferred'}, None))

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []

        # Soft-cancel fires immediately
        wf._cancel_event.set()

        outcome = await wf._maybe_enqueue_group_merge()

        # D9: the GroupMergeRequest future was cancelled (blanket fut.cancel(), no on_soft_cancel).
        # fut.cancel() schedules the done_callback (which releases the registry slot) for the
        # next event-loop iteration — yield a few ticks so the callback fires before asserting.
        for _ in range(5):
            await asyncio.sleep(0)

        # Registry slot (for 'B') is released by the done_callback on cancellation
        assert registry.entry('B') is None, 'entry must be released after cancel'

        # No attach or detach calls — train path uses acquire/release, not attach/detach
        assert attach_calls == [], f'registry.attach must not be called for train; got {attach_calls}'
        assert detach_calls == [], f'registry.detach must not be called for train; got {detach_calls}'

        # Soft-cancel → SOFT_CANCELLED (non-terminal scheduler status + cancel_event set)
        assert outcome == WorkflowOutcome.SOFT_CANCELLED


@pytest.mark.asyncio
class TestSubmitToMergeQueueEnqueuePathEdgeCases:
    """Coverage for soft-cancel on the enqueue (non-attach) path and rev-parse failures."""

    def _make_wf(self, tmp_path, *, registry):
        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        scheduler = MagicMock()
        scheduler.get_status = AsyncMock(return_value='in-progress')

        import asyncio
        real_queue: asyncio.Queue = asyncio.Queue()

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []
        return wf, real_queue

    async def test_enqueue_path_soft_cancel_cancels_future(
        self, tmp_path, monkeypatch,
    ):
        """When registry is present but branch slot is free, acquire succeeds and
        soft-cancel routes through detach() → count-0 → primary_future.cancel()."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry

        registry = InFlightMergeRegistry()
        wf, real_queue = self._make_wf(tmp_path, registry=registry)

        monkeypatch.setattr('orchestrator.merge_queue._emit_merge_coalesced', lambda *a, **kw: None)

        # Pre-set cancel_event; soft-cancel fires as soon as _await_cancellable runs.
        wf._cancel_event.set()

        outcome = await wf._submit_to_merge_queue('B', merge_phase=True)

        # Enqueued (branch slot was free).
        assert real_queue.qsize() == 1

        # The sole registry waiter was detached (count→0) → primary_future cancelled.
        # The done_callback fires async — yield a few ticks.
        for _ in range(5):
            await asyncio.sleep(0)
        assert registry.entry('B') is None, 'slot must be released after cancel'

        # Outcome is SOFT_CANCELLED (scheduler non-terminal + cancel_event set).
        assert outcome == WorkflowOutcome.SOFT_CANCELLED

    async def test_rev_parse_failure_falls_through_to_enqueue(
        self, tmp_path, monkeypatch,
    ):
        """When rev-parse fails (new_tip=None), coalesce is skipped and the
        workflow enqueues independently rather than attaching blind."""
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        registry = InFlightMergeRegistry()

        TIP = 'tip000000000000003'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        wf, real_queue = self._make_wf(tmp_path, registry=registry)

        # Stub rev-parse to fail so new_tip=None.
        async def fake_run_fail(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 1, '', 'fatal: not a git repository'
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run_fail)
        monkeypatch.setattr('orchestrator.merge_queue._emit_merge_coalesced', lambda *a, **kw: None)

        # Run without cancel so the test can drive completion normally.
        submit_task = asyncio.create_task(wf._submit_to_merge_queue('B', merge_phase=True))
        for _ in range(10):
            await asyncio.sleep(0)

        # Fell through to enqueue (did NOT attach as peer).
        assert real_queue.qsize() == 1, 'must have enqueued independently'
        entry = registry.entry('B')
        assert entry is not None
        assert len(entry.waiters) == 1, 'only the original MCP waiter; workflow not attached'

        # Resolve both futures so the task can finish cleanly.
        P.set_result(MergeOutcome(status='done', merge_sha='sha'))
        # The enqueued workflow request needs its result future resolved too.
        queued_req = real_queue.get_nowait()
        if not queued_req.result.done():
            queued_req.result.set_result(MergeOutcome(status='done', merge_sha='sha2'))
        await submit_task


# ---------------------------------------------------------------------------
# TestBoundaryTableWorkflow — PRD §8 row 10 (soft-cancel detach)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBoundaryTableWorkflow:
    """PRD §8 row 10: soft-cancel detach at the workflow seam.

    Extends TestSubmitToMergeQueueSoftCancelDetaches with:
    - The remaining-MCP-waiter-completes assertion (P resolves to 'done')
    - The re-attach-coalesces assertion (entry waiter count back to 2)
    """

    def _make_wf_with_peer(self, tmp_path, real_queue, registry, *, tip: str):
        """Build a workflow and pre-seed branch B with mcp-source primary P."""
        from orchestrator.merge_queue import (  # noqa: F401 (local)
            InFlightMergeRegistry,
            MergeOutcome,
        )

        assignment = MagicMock()
        assignment.task_id = 'B'
        assignment.task = {'id': 'B', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        config = MagicMock()
        config.fused_memory.project_id = 'dark_factory'
        config.fused_memory.url = 'http://localhost:8002'
        config.max_review_cycles = 2
        config.max_amendment_rounds = 1
        config.lock_depth = 2
        config.steward_completion_timeout = 300.0
        config.project_root = tmp_path

        wt = tmp_path / 'wt'
        wt.mkdir(parents=True, exist_ok=True)

        scheduler = MagicMock()
        scheduler.get_status = AsyncMock(return_value='in-progress')

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
            merge_queue=real_queue,
            merge_inflight_registry=registry,
        )
        wf.worktree = wt
        wf.plan = {}
        wf._base_commit = None
        wf._module_configs = []
        return wf

    async def test_scenario_10_soft_cancel_detach_remaining_waiter_completes(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Row 10: soft-cancel detaches workflow waiter; primary MCP waiter still completes.

        Acquire an mcp-source primary P on branch B; have the workflow attach
        as a second waiter, then soft-cancel.  Assert:
        (1) Workflow waiter detached — only mr-mcp remains (P NOT cancelled).
        (2) Workflow outcome is REQUEUED.
        (3) On retry, re-attach coalesces (entry waiter count = 2, no enqueue).
        (4) Resolve P to 'done' — the remaining MCP waiter P is still done.
        Extends TestSubmitToMergeQueueSoftCancelDetaches with (3) and (4).
        """
        import asyncio

        from orchestrator.merge_queue import InFlightMergeRegistry, MergeOutcome

        real_queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()

        TIP = 'tip-bt10-000000001'
        P: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        registry.acquire(
            'B', 'mcp-task', P,
            request_id='mr-mcp', source='mcp',
            submitted_tip=TIP, snapshot_tip=TIP,
        )

        wf = self._make_wf_with_peer(tmp_path, real_queue, registry, tip=TIP)

        async def fake_run(cmd, cwd=None, timeout=None):
            if 'rev-parse' in cmd:
                return 0, TIP + '\n', ''
            return 0, '', ''

        monkeypatch.setattr('orchestrator.workflow._run', fake_run)
        monkeypatch.setattr(
            'orchestrator.merge_queue._emit_merge_coalesced',
            lambda *a, **kw: None,
        )

        # ── (1)+(2) First call: soft-cancel → detach ───────────────────────
        wf._cancel_event.set()
        outcome1 = await wf._submit_to_merge_queue('B', merge_phase=True)

        entry = registry.entry('B')
        assert entry is not None, 'entry must remain in-flight after detach'
        assert len(entry.waiters) == 1, (
            f'Only mcp waiter must remain after detach, got {len(entry.waiters)} waiters'
        )
        assert entry.waiters[0].request_id == 'mr-mcp', (
            f'Remaining waiter must be mr-mcp, got: {entry.waiters[0].request_id!r}'
        )
        assert not P.cancelled(), 'Primary P must NOT be cancelled after workflow soft-cancel'
        assert outcome1 == WorkflowOutcome.SOFT_CANCELLED, (
            f'Expected SOFT_CANCELLED outcome after soft-cancel, got: {outcome1}'
        )

        # ── (3) Re-attach: coalesces back to 2 waiters ──────────────────────
        wf._cancel_event.clear()
        submit_task2 = asyncio.create_task(
            wf._submit_to_merge_queue('B', merge_phase=True)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        entry2 = registry.entry('B')
        assert entry2 is not None
        assert len(entry2.waiters) == 2, (
            f'Re-attach must coalesce to 2 waiters, got {len(entry2.waiters)}'
        )
        assert real_queue.qsize() == 0, (
            f'Re-attach must NOT enqueue a duplicate, qsize={real_queue.qsize()}'
        )

        # ── (4) Resolve P → both waiters complete ────────────────────────────
        P.set_result(MergeOutcome(status='done', merge_sha='sha-bt10'))
        outcome2 = await asyncio.wait_for(submit_task2, timeout=5.0)

        assert P.done() and not P.cancelled(), 'MCP waiter P must be done (not cancelled)'
        assert P.result().status == 'done', (
            f'MCP waiter P must resolve to done, got: {P.result().status!r}'
        )
        assert outcome2 == WorkflowOutcome.DONE, (
            f'Workflow re-attach outcome must be DONE, got: {outcome2}'
        )


# ---------------------------------------------------------------------------
# TestHandleSoftCancelOutcome — unit tests for the 3-way decision in
# _handle_soft_cancel: terminal → DONE, cancel_event set → SOFT_CANCELLED,
# cancel cleared (spurious) → REQUEUED.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleSoftCancelOutcome:
    """Three-way decision contract for _handle_soft_cancel."""

    def _make_wf(self, tmp_path: Path, *, status: str) -> TaskWorkflow:
        assignment = MagicMock()
        assignment.task_id = '99'
        assignment.task = {'id': '99', 'title': 'T', 'description': 'd'}
        assignment.modules = []

        from _orch_helpers import pydantic_spec  # noqa: PLC0415

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
        scheduler.get_status = AsyncMock(return_value=status)

        wf = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=MagicMock(),
            scheduler=scheduler,
            briefing=MagicMock(),
            mcp=MagicMock(),
        )
        return wf

    async def test_terminal_status_returns_done(self, tmp_path: Path):
        """Scheduler status 'done' → DONE regardless of cancel_event."""
        wf = self._make_wf(tmp_path, status='done')
        wf._cancel_event.set()  # even with event set, terminal wins
        outcome = await wf._handle_soft_cancel('merge')
        assert outcome == WorkflowOutcome.DONE

    async def test_non_terminal_cancel_set_returns_soft_cancelled(self, tmp_path: Path):
        """Scheduler status non-terminal + cancel_event.is_set() → SOFT_CANCELLED."""
        wf = self._make_wf(tmp_path, status='in-progress')
        wf._cancel_event.set()
        outcome = await wf._handle_soft_cancel('merge')
        assert outcome == WorkflowOutcome.SOFT_CANCELLED

    async def test_non_terminal_cancel_not_set_returns_requeued(self, tmp_path: Path):
        """Scheduler status non-terminal + cancel_event NOT set → REQUEUED (spurious)."""
        wf = self._make_wf(tmp_path, status='in-progress')
        # cancel_event is NOT set — simulates a spurious/cleared cancel
        assert not wf._cancel_event.is_set()
        outcome = await wf._handle_soft_cancel('merge')
        assert outcome == WorkflowOutcome.REQUEUED
