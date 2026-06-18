"""Tests for ``verify.run_main_tip_sweep`` — task 1832.

run_main_tip_sweep(config, git_ops) -> tuple[str, VerifyResult] | None

Runs a FULL unscoped verification (all subprojects) against a throwaway
detached worktree pinned at the current main SHA.  Returns (main_sha,
VerifyResult) on success, or None on any infrastructure failure (fail-safe).

Mirrors the mock strategy of test_verify_preexisting_main_break.py:
  - MagicMock git_ops with get_main_sha (AsyncMock) and worktree_base
  - monkeypatch orchestrator.git_ops._run to simulate worktree add/remove
  - monkeypatch orchestrator.verify.run_full_verification (AsyncMock)

Test coverage:
  step-3:  test_run_main_tip_sweep_happy_path
  step-5:  test_run_main_tip_sweep_failsafe_empty_sha
           test_run_main_tip_sweep_failsafe_worktree_add_fails
           test_run_main_tip_sweep_drift_passthrough
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

MAIN_SHA = 'a' * 40

PASSING_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all checks passed',
)

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED test_x',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='FAILED test_x',
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


def _make_git_ops(tmp_path: Path, main_sha: str = MAIN_SHA) -> MagicMock:
    """Build a minimal MagicMock git_ops."""
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value=main_sha)
    worktree_base = tmp_path / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    mock.worktree_base = worktree_base
    return mock


# ---------------------------------------------------------------------------
# step-3: happy path
# ---------------------------------------------------------------------------


class TestRunMainTipSweepHappyPath:
    """step-3: run_main_tip_sweep returns (main_sha, VerifyResult) on success
    AND the worktree add + remove commands were both issued."""

    def test_run_main_tip_sweep_happy_path(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        async def _fake_full_verify(project_root, cfg, **kwargs):
            return PASSING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected a (sha, VerifyResult) tuple, got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA, f'Expected main_sha={MAIN_SHA!r}, got {swept_sha!r}'
        assert vr.passed is True

        # Verify git worktree add was called
        add_calls = [c for c in run_calls if 'worktree' in c and 'add' in c]
        assert add_calls, 'Expected at least one git worktree add call'
        add_cmd = add_calls[0]
        assert '--detach' in add_cmd, f'Expected --detach in worktree add cmd: {add_cmd}'
        assert MAIN_SHA in add_cmd, f'Expected main_sha in worktree add cmd: {add_cmd}'

        # Verify git worktree remove was called (cleanup ran)
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, 'Expected a git worktree remove --force call (cleanup)'
        remove_cmd = remove_calls[0]
        assert '--force' in remove_cmd, f'Expected --force in worktree remove cmd: {remove_cmd}'
