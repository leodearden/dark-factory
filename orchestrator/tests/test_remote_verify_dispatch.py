"""Transport tests for ``RemoteRunner._dispatch``, reached only through the
public ``run_merge_verify`` / ``run_task_verify`` / ``probe_clean``.

PRD: plans/remote-task-leg-verify-prd.md §4.3, §5.2, §6 rows 3/4/19.

The no-verdict results these tests feed the transport are PRODUCED by
executing the public ``verify.run_verification`` against real subprocesses,
so a drift in the producer's summary wording fails here, at the boundary,
rather than passing against a hand-typed marker string.
"""

import shlex
from pathlib import Path

import pytest
from _xdist_crash_fixtures import XDIST_SESSION_ABORTED_OUTPUT

from orchestrator import verify
from orchestrator.config import OrchestratorConfig
from orchestrator.verify import (
    SIGNAL_KILL_SUMMARY_MARKER,
    WORKER_DEATH_SUMMARY_MARKER,
    VerifyResult,
)


async def _executed_result(worktree: Path, test_command: str) -> VerifyResult:
    config = OrchestratorConfig(
        project_root=worktree,
        test_command=test_command,
        lint_command='true',
        type_check_command='true',
        verify_timeout_retries=0,
    )
    return await verify.run_verification(worktree, config, max_retries=0)


async def _signal_killed_result(worktree: Path) -> VerifyResult:
    return await _executed_result(worktree, 'kill -9 $$')


async def _xdist_worker_death_result(worktree: Path) -> VerifyResult:
    output = worktree / 'xdist.out'
    output.write_text(XDIST_SESSION_ABORTED_OUTPUT)
    return await _executed_result(worktree, f'cat {shlex.quote(str(output))}; exit 1')


async def _genuine_red_result(worktree: Path) -> VerifyResult:
    return await _executed_result(worktree, 'echo plain failure; exit 1')


class TestIsNoVerdictSummary:
    @pytest.mark.asyncio
    async def test_executed_signal_kill_is_no_verdict(self, tmp_path: Path):
        result = await _signal_killed_result(tmp_path)

        assert not result.passed
        assert SIGNAL_KILL_SUMMARY_MARKER in result.summary
        assert verify.is_no_verdict_summary(result.summary) is True

    @pytest.mark.asyncio
    async def test_executed_xdist_worker_death_is_no_verdict(self, tmp_path: Path):
        result = await _xdist_worker_death_result(tmp_path)

        assert not result.passed
        assert WORKER_DEATH_SUMMARY_MARKER in result.summary
        assert verify.is_no_verdict_summary(result.summary) is True

    @pytest.mark.asyncio
    async def test_executed_genuine_red_is_a_verdict(self, tmp_path: Path):
        result = await _genuine_red_result(tmp_path)

        assert not result.passed
        assert result.summary == 'Failures: tests failed'
        assert verify.is_no_verdict_summary(result.summary) is False

    def test_green_summary_is_a_verdict(self):
        assert verify.is_no_verdict_summary('All checks passed') is False

    def test_empty_summary_is_a_verdict(self):
        assert verify.is_no_verdict_summary('') is False
