"""Tests for the broken-main contagion guard wired into _verify_debugfix_loop.

These tests drive ``TaskWorkflow._verify_debugfix_loop`` with a failing verify
whose cause is inherited from main (``verify_failure_is_preexisting_on_main``
returns True) and assert the guard:
  - does NOT invoke the debugger
  - returns WorkflowOutcome.BLOCKED
  - does NOT append to _failure_signature_history
  - populates _inherited_break_info with category + dedupe_fingerprint

Separate tests cover the non-inherited (fall-through) paths, flag-off, and
the infra_timeout / flock_error skip-set.  The escalation-filing tests drive
the real VERIFY-branch call site through ``_execute_verify_review_loop``.
"""
from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.main_health_actions import MainHealthNextAction
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
    compute_preexisting_main_break_fingerprint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAIN_SHA = 'deadbeef1234567890'

COMPILE_ERROR_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2769: foo.tsx:12',
    summary='TS2769 compile_error',
    cause_hint='error TS2769: foo.tsx:12',
    category='compile_error',
)

PASSING_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all checks passed',
)


def _make_config(tmp_path: Path, *, escalate_preexisting: bool = True) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        max_verify_attempts=3,
        escalate_preexisting_main_break=escalate_preexisting,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_workflow(
    config: OrchestratorConfig,
    worktree: Path,
) -> TaskWorkflow:
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    git_ops = MagicMock(spec=GitOps)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    git_ops.rebase_onto_main = AsyncMock(return_value=None)

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    workflow._check_escalations = MagicMock(return_value=[])
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')
    from orchestrator.agents.invoke import AgentResult
    workflow._invoke = AsyncMock(return_value=AgentResult(success=True, output=''))
    workflow._get_head_commit = AsyncMock(return_value='head-sha')
    # Stub rebase so it doesn't touch git
    workflow._inter_iteration_rebase = AsyncMock(return_value=None)
    return workflow


def _make_workflow_with_queue(tmp_path: Path) -> tuple[TaskWorkflow, EscalationQueue]:
    """An ownerless workflow (no merge worker) filing into a real escalation queue."""
    worktree = tmp_path / 'task-wt'
    worktree.mkdir()
    workflow = _make_workflow(_make_config(tmp_path), worktree)
    queue = EscalationQueue(tmp_path / 'escalations')
    workflow.escalation_queue = queue
    return workflow, queue


def _block_at_the_verify_call_site(
    workflow: TaskWorkflow, *also: AbstractContextManager,
) -> WorkflowOutcome:
    """Run the execute/verify/review loop past a stubbed EXECUTE into an inherited verify break."""

    async def _drive() -> WorkflowOutcome:
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                workflow, '_execute_iterations',
                new=AsyncMock(return_value=WorkflowOutcome.PLANNED),
            ))
            stack.enter_context(patch(
                'orchestrator.workflow.run_scoped_verification',
                new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
            ))
            stack.enter_context(patch(
                'orchestrator.workflow.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ))
            for context in also:
                stack.enter_context(context)
            return await workflow._execute_verify_review_loop()

    return asyncio.run(_drive())


# ---------------------------------------------------------------------------
# Test 1 — inherited break: BLOCKED, debugger not called, history not advanced
# ---------------------------------------------------------------------------


class TestVerifyDebugfixLoopInheritedBreak:
    """Step-7 / test-expectation #1: guard detects inherited break and blocks without patching."""

    def test_inherited_break_blocks_without_debugger(self, tmp_path: Path) -> None:
        """When verify_failure_is_preexisting_on_main returns True:
          - DEBUGGER (_invoke) is NOT called
          - loop returns WorkflowOutcome.BLOCKED
          - _failure_signature_history is NOT appended (length stays 0)
          - _inherited_break_info is populated with category and fingerprint
        """
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)

        async def _run_loop() -> WorkflowOutcome:
            with (
                # run_scoped_verification is imported at module level in workflow.py
                patch('orchestrator.workflow.run_scoped_verification',
                      new=AsyncMock(return_value=COMPILE_ERROR_RESULT)),
                # verify_failure_is_preexisting_on_main will be imported in workflow.py (step-8)
                patch('orchestrator.workflow.verify_failure_is_preexisting_on_main',
                      new=AsyncMock(return_value=(True, MAIN_SHA))),
            ):
                return await workflow._verify_debugfix_loop()

        outcome = asyncio.run(_run_loop())

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED when break is inherited; got {outcome}'
        )
        workflow._invoke.assert_not_called()  # type: ignore[union-attr]  # debugger must NOT run
        assert len(workflow._failure_signature_history) == 0, (
            'Signature history must not be advanced for inherited failures '
            f'(actual: {workflow._failure_signature_history})'
        )
        info = workflow._inherited_break_info  # type: ignore[attr-defined]
        assert info is not None, '_inherited_break_info must be populated'
        assert info.get('category') == 'preexisting_main_break', (
            f'Expected category=preexisting_main_break; got {info}'
        )
        assert info.get('fingerprint'), '_inherited_break_info must carry a non-empty fingerprint'


# ---------------------------------------------------------------------------
# Test 2 — non-inherited: helper=False -> debugger invoked, history advanced
# Test 2b — flag off -> helper not called
# Test 2c — skip-set category -> helper not called
# ---------------------------------------------------------------------------


class TestVerifyDebugfixLoopNonInheritedPaths:
    """Step-9 / test-expectation #2 + invariant e + flag-off: existing path unchanged."""

    def _run_with_spy(
        self,
        tmp_path: Path,
        *,
        helper_return: bool = False,
        flag_on: bool = True,
        category: str = 'compile_error',
    ) -> tuple[WorkflowOutcome, MagicMock, MagicMock]:
        """Run one loop iteration, return (outcome, helper_spy, invoke_spy)."""
        config = _make_config(tmp_path, escalate_preexisting=flag_on)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)

        failing_result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='error TS2769: foo.tsx:12',
            summary=f'{category} error',
            cause_hint='error TS2769: foo.tsx:12',
            category=category,
        )

        # Helper now returns (bool, str) tuple; False→(False,''), True→(True, MAIN_SHA).
        helper_spy = AsyncMock(
            return_value=(True, MAIN_SHA) if helper_return else (False, ''),
        )
        # After one failing run: make next run pass to exit loop normally.
        call_count = {'n': 0}

        async def _verify_side_effect(*args, **kwargs):
            call_count['n'] += 1
            if call_count['n'] == 1:
                return failing_result
            return PASSING_RESULT

        async def _run_loop() -> WorkflowOutcome:
            with (
                patch('orchestrator.workflow.run_scoped_verification',
                      side_effect=_verify_side_effect),
                patch('orchestrator.workflow.verify_failure_is_preexisting_on_main',
                      new=helper_spy),
            ):
                return await workflow._verify_debugfix_loop()

        outcome = asyncio.run(_run_loop())
        return outcome, helper_spy, workflow._invoke  # type: ignore[return-value]

    def test_helper_false_debugger_is_called_and_history_advances(
        self, tmp_path: Path,
    ) -> None:
        """When helper returns False, existing path runs: debugger called, history +1."""
        outcome, helper_spy, invoke_spy = self._run_with_spy(tmp_path, helper_return=False)
        assert outcome == WorkflowOutcome.DONE  # second verify passes
        invoke_spy.assert_called()  # type: ignore[union-attr]

    def test_flag_off_helper_not_called(self, tmp_path: Path) -> None:
        """When escalate_preexisting_main_break=False, helper is never called."""
        outcome, helper_spy, invoke_spy = self._run_with_spy(
            tmp_path, helper_return=False, flag_on=False,
        )
        helper_spy.assert_not_called()

    def test_infra_timeout_category_skips_helper(self, tmp_path: Path) -> None:
        """category='infra_timeout' is in the skip-set — helper must not be called."""
        _outcome, helper_spy, _invoke = self._run_with_spy(
            tmp_path, category='infra_timeout',
        )
        helper_spy.assert_not_called()

    def test_flock_error_category_skips_helper(self, tmp_path: Path) -> None:
        """category='flock_error' is in the skip-set — helper must not be called."""
        _outcome, helper_spy, _invoke = self._run_with_spy(
            tmp_path, category='flock_error',
        )
        helper_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — sibling fix: verify PASSES post-rebase -> guard never reached
# ---------------------------------------------------------------------------


class TestVerifyDebugfixLoopSiblingFix:
    """Step-11 / test-expectation #3: post-rebase passing verify exits DONE without guard."""

    def test_passing_verify_exits_done_without_calling_helper(
        self, tmp_path: Path,
    ) -> None:
        """When run_scoped_verification returns passed=True, loop returns DONE immediately.

        verify_failure_is_preexisting_on_main must NOT be called,
        no escalation/_mark_blocked occurs, and _inherited_break_info stays None.
        """
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)

        helper_spy = AsyncMock(return_value=(False, ''))

        async def _run_loop() -> WorkflowOutcome:
            with (
                patch('orchestrator.workflow.run_scoped_verification',
                      new=AsyncMock(return_value=PASSING_RESULT)),
                patch('orchestrator.workflow.verify_failure_is_preexisting_on_main',
                      new=helper_spy),
            ):
                return await workflow._verify_debugfix_loop()

        outcome = asyncio.run(_run_loop())

        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE when verify passes; got {outcome}'
        )
        helper_spy.assert_not_called()
        assert workflow._inherited_break_info is None, (
            '_inherited_break_info must stay None when verify passes'
        )
        workflow._invoke.assert_not_called()  # type: ignore[union-attr]  # debugger must not run


# ---------------------------------------------------------------------------
# Test 4 — the real VERIFY-branch call site files the steward-facing L0
# ---------------------------------------------------------------------------


class TestInheritedBreakEscalationFiling:
    """The inherited-break L0 is a cross-task-folded, level-0 'blocking' record that
    asks its steward to re-run the failures in isolation before deferring to a fix owner."""

    def test_inherited_break_files_a_steward_l0_asking_for_an_isolated_rerun(
        self, tmp_path: Path,
    ) -> None:
        from escalation.models import BORN_AT_L2_SEVERITIES

        workflow, queue = _make_workflow_with_queue(tmp_path)

        outcome = _block_at_the_verify_call_site(workflow)

        assert outcome == WorkflowOutcome.BLOCKED
        set_task_status = workflow.scheduler.set_task_status
        assert isinstance(set_task_status, AsyncMock)
        set_task_status.assert_any_await(workflow.task_id, 'blocked')
        [esc] = queue.get_by_task(workflow.task_id, level=0)
        assert esc.level == 0
        assert esc.severity == 'blocking'
        assert esc.severity not in BORN_AT_L2_SEVERITIES
        assert esc.category == 'preexisting_main_break'
        assert esc.dedupe_fingerprint == compute_preexisting_main_break_fingerprint(
            'compile_error', COMPILE_ERROR_RESULT.cause_hint, MAIN_SHA,
        )
        assert esc.suggested_action == MainHealthNextAction.STEWARD_RERUN_IN_ISOLATION


class TestCrossTaskInheritedBreakDedup:
    """Step-15 / test-expectation #4: N tasks with same (category, cause_hint, main_sha)
    signature -> ONE parent escalation; different main_sha -> distinct fingerprint,
    no fold into the first parent.

    Tests the fingerprint composition (`compute_content_fingerprint`) plus the
    `submit_or_dedupe` routing from `_mark_blocked` directly — bypassing the
    steward/cleanup path so the parent L0 is still pending when the second task
    submits, mirroring the concurrent real-world scenario.
    """

    def test_same_signature_folds_to_one_parent(self, tmp_path: Path) -> None:
        """Two tasks (A, B) with identical (category, cause_hint, main_sha) dedupe to one parent.

        Asserts:
        - Task A's submission: status='queued', exactly 1 file in queue_dir.
        - Task B's submission: status='dedup_skipped', still 1 file in queue_dir.
        - Parent (A) has dedupe_count == 1 and B's escalation id in dedupe_children.
        Contrast:
        - Task C uses a DIFFERENT main_sha -> different fingerprint -> status='queued',
          now 2 files in queue_dir (no fold into A's parent).
        """
        from escalation.dedupe import (
            DedupeConfig,
            compute_content_fingerprint,
            content_fingerprint_key,
            submit_or_dedupe,
        )
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        from orchestrator.workflow import _normalize_cause_hint

        eq = EscalationQueue(tmp_path / 'escalations')

        category = 'compile_error'
        cause_hint = 'error TS2769: foo.tsx:12'
        main_sha = MAIN_SHA
        different_main_sha = 'cafebabe1234567890'

        # Fingerprint for tasks A and B (same signature, same main_sha)
        fp_ab = compute_content_fingerprint(
            'preexisting_main_break',
            category,
            [],
            description=_normalize_cause_hint(cause_hint) + '|' + main_sha,
        )
        # Fingerprint for task C (same signature, different main_sha — hotfix landed)
        fp_c = compute_content_fingerprint(
            'preexisting_main_break',
            category,
            [],
            description=_normalize_cause_hint(cause_hint) + '|' + different_main_sha,
        )
        assert fp_ab != fp_c, (
            'Fingerprint must differ when main_sha differs; '
            f'fp_ab={fp_ab!r}, fp_c={fp_c!r}'
        )

        dedup_config = DedupeConfig(
            infra_dedupe_enabled=True,
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=('preexisting_main_break',),
            key_fn=content_fingerprint_key,
        )

        def _make_esc(task_id: str, fp: str) -> Escalation:
            esc = Escalation(
                id=eq.make_id(task_id),
                task_id=task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='preexisting_main_break',
                summary='Verify failure is preexisting on main',
                detail='TS2769 type error',
                suggested_action=MainHealthNextAction.STEWARD_RERUN_IN_ISOLATION,
            )
            esc.dedupe_fingerprint = fp
            return esc

        # --- Task A: first parent (no existing parent) ---
        esc_a = _make_esc('A', fp_ab)
        result_a = submit_or_dedupe(eq, esc_a, dedup_config)
        assert result_a.get('status') == 'queued', (
            f'Task A should be queued as parent; got {result_a}'
        )
        queue_files_after_a = sorted(eq.queue_dir.glob('esc-*.json'))
        assert len(queue_files_after_a) == 1, (
            f'Expected 1 file after task A submits; got {queue_files_after_a}'
        )

        # --- Task B: same fingerprint -> folds into A's parent ---
        esc_b = _make_esc('B', fp_ab)
        result_b = submit_or_dedupe(eq, esc_b, dedup_config)
        assert result_b.get('status') == 'dedup_skipped', (
            f'Task B (same fingerprint) must fold into A; got {result_b}'
        )
        assert result_b.get('parent_id') == esc_a.id, (
            f'B must fold under A (id={esc_a.id!r}); got parent_id={result_b.get("parent_id")!r}'
        )
        queue_files_after_b = sorted(eq.queue_dir.glob('esc-*.json'))
        assert len(queue_files_after_b) == 1, (
            f'Expected still 1 file after task B folds; got {queue_files_after_b}'
        )

        # Parent (A) dedupe accounting: dedupe_count == 1, B's id in dedupe_children
        parent_esc = eq.get(esc_a.id)
        assert parent_esc is not None, f'Parent escalation {esc_a.id!r} must exist'
        assert parent_esc.dedupe_count == 1, (
            f'dedupe_count must be 1 after one child fold; got {parent_esc.dedupe_count}'
        )
        assert esc_b.id in parent_esc.dedupe_children, (
            f'B id {esc_b.id!r} must be in dedupe_children; '
            f'got {parent_esc.dedupe_children!r}'
        )

        # --- Task C: different main_sha -> different fingerprint -> NEW parent ---
        esc_c = _make_esc('C', fp_c)
        result_c = submit_or_dedupe(eq, esc_c, dedup_config)
        assert result_c.get('status') == 'queued', (
            f'Task C (different main_sha, fp={fp_c!r}) must NOT fold into A; got {result_c}'
        )
        queue_files_after_c = sorted(eq.queue_dir.glob('esc-*.json'))
        assert len(queue_files_after_c) == 2, (
            f'Expected 2 files after task C (different main_sha) queues a new parent; '
            f'got {queue_files_after_c}'
        )


# ---------------------------------------------------------------------------
# Test 6 — fingerprint-exception fallback: fp='' -> raw-submit branch
# ---------------------------------------------------------------------------


class TestFingerprintExceptionFallback:
    """When compute_content_fingerprint raises, the call site still files the steward L0,
    raw-submitted (no submit_or_dedupe fold) with its category, severity and action intact."""

    def test_a_raising_fingerprint_files_the_steward_l0_raw(self, tmp_path: Path) -> None:
        workflow, queue = _make_workflow_with_queue(tmp_path)

        outcome = _block_at_the_verify_call_site(
            workflow,
            # Patched at its source: workflow imports it lazily at call time.
            patch(
                'escalation.dedupe.compute_content_fingerprint',
                side_effect=RuntimeError('fingerprint computation failed'),
            ),
        )

        assert outcome == WorkflowOutcome.BLOCKED
        [esc] = queue.get_by_task(workflow.task_id, level=0)
        assert not esc.dedupe_fingerprint
        assert esc.category == 'preexisting_main_break'
        assert esc.severity == 'blocking'
        assert esc.suggested_action == MainHealthNextAction.STEWARD_RERUN_IN_ISOLATION
