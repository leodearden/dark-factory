"""runner.collect_metrics, the executor's metric collection, and its git diff-stat probe."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.evals.metrics import RoleUnderTest
from orchestrator.evals.runner import collect_metrics
from orchestrator.verify import VerifyResult


def _workflow() -> MagicMock:
    wf = MagicMock()
    for name in (
        'total_duration_ms', 'total_output_tokens', 'total_input_tokens',
        'total_cache_read_tokens', 'total_cache_create_tokens', 'total_turns',
        'execute_iterations', 'verify_attempts', 'judge_invocations',
        'judge_early_exits',
    ):
        setattr(wf.metrics, name, 0)
    wf.metrics.total_cost_usd = 0.0
    wf.metrics.judge_cost_usd = 0.0
    wf.config.env_overrides = {}
    wf.config.max_execute_iterations = 20
    wf.artifacts.read_plan.return_value = {'steps': []}
    wf.artifacts.aggregate_reviews.return_value = None
    return wf


@pytest.mark.asyncio
async def test_collect_metrics_sets_role_and_replay_frame_on_the_dataclass():
    verification = VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='',
    )
    with (
        patch('orchestrator.verify.run_verification', AsyncMock(return_value=verification)),
        patch('orchestrator.evals.runner._git_diff_stats', AsyncMock(return_value=(3, 1))),
    ):
        m = await collect_metrics(
            _workflow(), Path('/fake/wt'), {'id': 't', 'pre_task_commit': 'base'},
            role=RoleUnderTest.JUDGE, replay_frame='f',
        )

    assert m.role_under_test is RoleUnderTest.JUDGE
    assert m.replay_frame == 'f'


# ---------------------------------------------------------------------------
# Task 1809 step-15: _git_diff_stats returncode check + (-1,-1) sentinel
# ---------------------------------------------------------------------------

def _make_fake_proc(returncode: int, stdout: bytes, stderr: bytes) -> MagicMock:
    """Build a minimal fake asyncio Process mock."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.mark.asyncio
class TestGitDiffStatsReturncode:
    """Step-15 (RED): _git_diff_stats must return (-1,-1) and emit a WARNING
    when git exits non-zero, rather than returning (0,0) silently.

    Before step-16 impl: rc!=0 path is silent and returns (0,0) → RED.
    After step-16 impl: returns (-1,-1) + WARNING at 'orchestrator.evals.runner'.
    """

    async def test_nonzero_returncode_returns_sentinel_and_warns(
        self, caplog,
    ) -> None:
        from orchestrator.evals.runner import _git_diff_stats

        fake_proc = _make_fake_proc(
            returncode=128,
            stdout=b'',
            stderr=b'fatal: bad object deadbeef',
        )

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.runner'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'deadbeef')

        assert result == (-1, -1), (
            f'Expected (-1,-1) sentinel on rc=128; got {result!r}'
        )
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.evals.runner on rc!=0; '
            'got no warnings'
        )

    async def test_create_subprocess_raises_returns_sentinel_and_warns(
        self, caplog,
    ) -> None:
        from orchestrator.evals.runner import _git_diff_stats

        with (
            patch(
                'asyncio.create_subprocess_exec',
                side_effect=OSError('no such file'),
            ),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.runner'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'deadbeef')

        assert result == (-1, -1), (
            f'Expected (-1,-1) sentinel on OSError; got {result!r}'
        )
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.evals.runner on OSError; '
            'got no warnings'
        )

    async def test_zero_returncode_valid_stat_output_parsed(
        self, caplog,
    ) -> None:
        from orchestrator.evals.runner import _git_diff_stats

        stat_output = (
            b' src/foo.py | 10 +++++-----\n'
            b' src/bar.py |  5 ++---\n'
            b' 2 files changed, 15 insertions(+), 0 deletions(-)\n'
        )
        fake_proc = _make_fake_proc(returncode=0, stdout=stat_output, stderr=b'')

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.runner'),
        ):
            lines, files = await _git_diff_stats(Path('/fake/worktree'), 'abc123')

        assert files == 2, f'Expected files=2; got {files!r}'
        assert lines == 15, f'Expected lines=15; got {lines!r}'
        parse_warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not parse_warnings, (
            f'No WARNING expected for successful parse; got: {parse_warnings}'
        )

    async def test_zero_returncode_empty_stdout_returns_zero_zero(
        self, caplog,
    ) -> None:
        from orchestrator.evals.runner import _git_diff_stats

        fake_proc = _make_fake_proc(returncode=0, stdout=b'', stderr=b'')

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.runner'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'abc123')

        assert result == (0, 0), (
            f'Expected (0,0) for legitimate empty diff; got {result!r}'
        )
        parse_warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not parse_warnings, (
            f'No WARNING expected for empty-diff (no-change); got: {parse_warnings}'
        )
