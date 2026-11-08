"""The executing host's half of a task-role verify: the body of `orchestrator verify-task`.

PRD: plans/remote-task-leg-verify-prd.md §4.2, §5.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _task_verify_fixtures import seed_verify_repo, task_verify_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.task_verify_run import (
    prepare_task_verify_worktree,
    run_task_verify_on_worktree,
)


class TestPrepareTaskVerifyWorktree:
    def test_creates_an_empty_task_dir(self, tmp_path: Path) -> None:
        prepare_task_verify_worktree(tmp_path)

        assert (tmp_path / '.task').is_dir()
        assert list((tmp_path / '.task').iterdir()) == []

    def test_refuses_a_pre_existing_task_dir(self, tmp_path: Path) -> None:
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        (task_dir / 'verify_warmed').write_text('')

        with pytest.raises(RuntimeError, match='cold') as excinfo:
            prepare_task_verify_worktree(tmp_path)

        assert str(task_dir) in str(excinfo.value)
        assert (task_dir / 'verify_warmed').exists()


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
@pytest.mark.asyncio
async def test_run_task_verify_on_a_fresh_worktree_persists_logs_and_strips_paths(
    tmp_path: Path,
) -> None:
    repo, head_sha = seed_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo, verify_cold_preprovision_command='')
    git_ops = GitOps(config.git, repo)
    spec = task_verify_spec(head_sha, 'true')
    wt = await git_ops.create_ephemeral_verify_worktree(head_sha)
    try:
        result = await run_task_verify_on_worktree(wt, config, spec)

        assert result.passed
        assert result.worktree_log_paths == []
        assert result.archive_log_paths == []
        persisted = sorted(p.name for p in (wt / '.task' / 'verify').iterdir())
        assert persisted, 'the cold .task/ invariant must enable attempt-log persistence'
        assert all(name.startswith('attempt-1.') for name in persisted), persisted
    finally:
        await git_ops.cleanup_merge_worktree(wt)
