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
