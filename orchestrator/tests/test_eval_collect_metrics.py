"""runner.collect_metrics, the executor's metric collection, measured against a real git worktree."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _eval_diff_repo import repo_with_commit

from orchestrator.evals.metrics import EvalMetrics, RoleUnderTest
from orchestrator.evals.runner import collect_metrics
from orchestrator.verify import VerifyResult

_RUNNER_LOGGER = 'orchestrator.evals.runner'


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


async def _collect(
    worktree: Path,
    base_commit: str,
    *,
    role: RoleUnderTest = RoleUnderTest.IMPLEMENTER,
    replay_frame: str | None = None,
) -> EvalMetrics:
    verification = VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='',
    )
    with patch('orchestrator.verify.run_verification', AsyncMock(return_value=verification)):
        return await collect_metrics(
            _workflow(), worktree, {'id': 't', 'pre_task_commit': base_commit},
            role=role, replay_frame=replay_frame,
        )


def _runner_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _RUNNER_LOGGER and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_collect_metrics_sets_role_and_replay_frame_on_the_dataclass(tmp_path: Path):
    repo, base = repo_with_commit(tmp_path / 'repo', {})

    m = await _collect(repo, base, role=RoleUnderTest.JUDGE, replay_frame='f')

    assert m.role_under_test is RoleUnderTest.JUDGE
    assert m.replay_frame == 'f'


@pytest.mark.asyncio
class TestDiffStats:
    """lines_changed / files_changed measure the base..HEAD diff.

    A measurement that failed reads (-1, -1) with a WARNING, never a silent
    (0, 0) that would pass for a genuine no-change run (task 1809).
    """

    async def test_committed_change_counts_lines_and_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, base = repo_with_commit(
            tmp_path / 'repo',
            {'README.md': '# Changed\n', 'src/b.py': 'x = 1\ny = 2\n'},
        )

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, base)

        assert (m.lines_changed, m.files_changed) == (4, 2)
        assert not _runner_warnings(caplog)

    async def test_nothing_committed_since_base_is_zero_zero(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, base = repo_with_commit(tmp_path / 'repo', {})

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, base)

        assert (m.lines_changed, m.files_changed) == (0, 0)
        assert not _runner_warnings(caplog)

    async def test_unknown_base_commit_is_the_sentinel_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo, _base = repo_with_commit(tmp_path / 'repo', {})

        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(repo, 'deadbeef')

        assert (m.lines_changed, m.files_changed) == (-1, -1)
        assert _runner_warnings(caplog)

    async def test_absent_worktree_is_the_sentinel_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
            m = await _collect(tmp_path / 'absent', 'deadbeef')

        assert (m.lines_changed, m.files_changed) == (-1, -1)
        assert _runner_warnings(caplog)
