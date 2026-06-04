"""Tests for verify_failure_is_preexisting_on_main — the broken-main contagion helper.

The helper detects whether a verify failure that occurred on a task branch
already exists on main (i.e. was inherited from the rebase target, not
introduced by this task).  Tests are organised as:

  1. Happy path: same signature on main -> True
  2. False cases: main passes / different signature -> False
  3. Cheapness refinement: only non-flaky categories reach the helper
  4. Lifecycle / cleanup: temp worktree create + remove in finally, no task-worktree mutation
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

MAIN_SHA = 'aabbcc1122334455'

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2769: foo.tsx:12',
    summary='TS2769 compile_error',
    cause_hint='error TS2769: foo.tsx:12',
    category='compile_error',
)

SAME_RESULT = VerifyResult(
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

DIFFERENT_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED test_bar',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='AssertionError in test_bar',
    category='test_failure',
)


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


# ---------------------------------------------------------------------------
# Test 1 — happy path: same failure signature on main -> True
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingOnMain:
    """Step-1 / test-expectation #1: same (category, cause_hint) on main -> True."""

    def test_returns_true_when_same_signature_on_main(self, tmp_path: Path) -> None:
        """When main probe reproduces the same (category, cause_hint) -> True."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        module_configs: list = []
        task_files = ['src/foo.tsx']

        # Mock git_ops: get_main_sha returns a SHA
        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)

        # run_scoped_verification on main -> same failure
        async def _fake_verify(*args, **kwargs) -> VerifyResult:
            return SAME_RESULT

        # _run (git) -> always succeed (rc=0)
        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_fake_verify),
            patch(
                'orchestrator.git_ops._run',  # lazy-imported inside the helper
                side_effect=_fake_run,
            ),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, module_configs, task_files,
                    FAILING_RESULT, mock_git_ops,
                )
            )

        assert result is True, (
            'Expected True — same (category, cause_hint) reproduced on main means inherited.'
        )
