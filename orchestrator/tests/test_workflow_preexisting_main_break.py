"""Tests for the broken-main contagion guard wired into _verify_debugfix_loop.

These tests drive ``TaskWorkflow._verify_debugfix_loop`` with a failing verify
whose cause is inherited from main (``verify_failure_is_preexisting_on_main``
returns True) and assert the guard:
  - does NOT invoke the debugger
  - returns WorkflowOutcome.BLOCKED
  - does NOT append to _failure_signature_history
  - populates _inherited_break_info with category + dedupe_fingerprint

Separate tests cover the non-inherited (fall-through) paths, flag-off, and
the infra_timeout / flock_error skip-set.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


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

    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),
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
        import orchestrator.verify as verify_module

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
                      new=AsyncMock(return_value=True)),
            ):
                return await workflow._verify_debugfix_loop()

        outcome = asyncio.run(_run_loop())

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED when break is inherited; got {outcome}'
        )
        workflow._invoke.assert_not_called()  # debugger must NOT run
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
        import orchestrator.verify as verify_module

        from orchestrator.agents.invoke import AgentResult

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

        helper_spy = AsyncMock(return_value=helper_return)
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
        return outcome, helper_spy, workflow._invoke

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

        helper_spy = AsyncMock(return_value=False)

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
        workflow._invoke.assert_not_called()  # debugger must not run
