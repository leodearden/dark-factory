"""`orchestrator verify-task` and the remote-verify worktree hygiene it shares.

A remote verify (``verify-merge`` or ``verify-task``) records its worktree
beside its pgid file; ``cancel-verify`` SIGKILLs the tree, which skips the
subcommand's own ``finally``, so it reaps the recorded worktree instead.
PRD: plans/remote-task-leg-verify-prd.md §4.2, §5.3.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from test_cli import _setup_verify_repo

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import (
    pgid_file,
    read_worktree_path,
    worktree_path_file,
    write_pgid_file,
    write_worktree_path,
)

NONEXISTENT_PID = 2 ** 22  # no process can hold this pid: it is at pid_max


def _invoke_cancel_verify(tmp_path: Path, request_id: str) -> Result:
    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')
    return CliRunner().invoke(
        main, ['cancel-verify', '--request-id', request_id, '--config', str(cfg_file)],
    )


def _registered_worktrees(repo: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), 'worktree', 'list', '--porcelain'],
        check=True, capture_output=True, text=True,
    ).stdout


class TestCancelVerifyReapsRecordedWorktree:
    @pytest.fixture
    def repo_setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        repo, head_sha = _setup_verify_repo(tmp_path)
        config = OrchestratorConfig(project_root=repo)
        monkeypatch.setattr(cli_module, 'load_config', lambda _: config)
        return repo, head_sha, GitOps(config.git, repo)

    def _record_dead_run(self, git_ops: GitOps, request_id: str, worktree: Path) -> Path:
        pgf = pgid_file(git_ops.worktree_base, request_id)
        write_pgid_file(pgf, NONEXISTENT_PID)
        write_worktree_path(pgf, worktree)
        return pgf

    def test_reaps_recorded_ephemeral_merge_worktree(self, tmp_path: Path, repo_setup) -> None:
        repo, head_sha, git_ops = repo_setup
        wt = asyncio.run(git_ops.create_ephemeral_verify_worktree(head_sha))
        pgf = self._record_dead_run(git_ops, 'req-reap', wt)

        r = _invoke_cancel_verify(tmp_path, 'req-reap')

        assert r.exit_code == 0, r.output
        assert not wt.exists()
        assert str(wt) not in _registered_worktrees(repo)
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()

    def test_refuses_to_reap_a_task_shaped_path(self, tmp_path: Path, repo_setup) -> None:
        _, _, git_ops = repo_setup
        task_dir = git_ops.worktree_base / '1234'
        task_dir.mkdir(parents=True)
        sentinel = task_dir / 'sentinel'
        sentinel.write_text('keep me')
        pgf = self._record_dead_run(git_ops, 'req-task', task_dir)

        r = _invoke_cancel_verify(tmp_path, 'req-task')

        assert r.exit_code == 0, r.output
        assert sentinel.read_text() == 'keep me'
        assert str(task_dir) in r.stderr
        assert read_worktree_path(pgf) is None

    def test_refuses_a_merge_named_path_outside_worktree_base(
        self, tmp_path: Path, repo_setup,
    ) -> None:
        _, _, git_ops = repo_setup
        stray = tmp_path / 'elsewhere' / '_merge-stray'
        stray.mkdir(parents=True)
        sentinel = stray / 'sentinel'
        sentinel.write_text('keep me')
        self._record_dead_run(git_ops, 'req-stray', stray)

        r = _invoke_cancel_verify(tmp_path, 'req-stray')

        assert r.exit_code == 0, r.output
        assert sentinel.read_text() == 'keep me'
        assert str(stray) in r.stderr
