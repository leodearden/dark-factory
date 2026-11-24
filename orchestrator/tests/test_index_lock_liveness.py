"""Who holds a work tree's index.lock, and what is safe to tell an operator (task 5229).

The scan tests use real processes in real repos: the property under test is
what ``/proc`` says about them, so a fake process table would only restate the
implementation.
"""
from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator
from pathlib import Path

from _blocked_commit_fixture import commit_blocked_in_pre_commit_hook
from _git_fixtures import seed_repo
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.index_lock_liveness import scan_work_tree_git_processes


def _env_fenced_at_parent_of(cwd: Path) -> dict[str, str]:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return env


@contextlib.contextmanager
def _waiting_on_stdin(argv: list[str], cwd: Path) -> Iterator[subprocess.Popen[bytes]]:
    """*argv* running in *cwd*, blocked reading stdin until the block exits.

    ``Popen`` returns only after ``exec``, so the process's comm and cwd are
    already what the scan will read: no readiness wait is needed.
    """
    proc = subprocess.Popen(
        argv, cwd=cwd, env=_env_fenced_at_parent_of(cwd),
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def _pids(work_tree: Path) -> set[int]:
    return {process.pid for process in scan_work_tree_git_processes(work_tree).processes}


def _nested_worktree(repo: Path) -> Path:
    nested = repo / '.worktrees' / 'wt'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '-b', 'nested', str(nested)],
        cwd=repo, env=_env_fenced_at_parent_of(repo), check=True, capture_output=True,
    )
    return nested


class TestScanWorkTreeGitProcesses:
    def test_a_live_git_process_in_the_work_tree_is_reported_until_it_exits(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], repo) as proc:
            scan = scan_work_tree_git_processes(repo)
            assert scan.confirmed is True
            [process] = [p for p in scan.processes if p.pid == proc.pid]
            assert 'hash-object' in process.command
        assert proc.pid not in _pids(repo)

    def test_a_commit_blocked_in_its_pre_commit_hook_is_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        (repo / 'README.md').write_text('# dirtied so --only has a change\n')
        with commit_blocked_in_pre_commit_hook(repo, 'README.md') as git_pid:
            scan = scan_work_tree_git_processes(repo)
            assert scan.confirmed is True
            [process] = [p for p in scan.processes if p.pid == git_pid]
            assert 'commit' in process.command

    def test_a_git_process_in_a_subdirectory_of_the_work_tree_is_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        subdir = repo / 'sub' / 'deeper'
        subdir.mkdir(parents=True)
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], subdir) as proc:
            assert proc.pid in _pids(repo)

    def test_a_git_process_in_a_nested_linked_worktree_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        nested = _nested_worktree(repo)
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], nested) as proc:
            assert proc.pid not in _pids(repo)
            assert proc.pid in _pids(nested)

    def test_a_git_process_below_a_nested_worktree_root_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        below_nested = _nested_worktree(repo) / 'sub'
        below_nested.mkdir()
        with _waiting_on_stdin(['git', 'hash-object', '--stdin'], below_nested) as proc:
            assert proc.pid not in _pids(repo)

    def test_a_non_git_process_in_the_work_tree_is_not_reported(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo')
        with _waiting_on_stdin(['cat'], repo) as proc:
            scan = scan_work_tree_git_processes(repo)
            assert scan.confirmed is True
            assert proc.pid not in {p.pid for p in scan.processes}

    def test_an_unenumerable_proc_root_is_unknown_not_empty(self, tmp_path):
        scan = scan_work_tree_git_processes(tmp_path, proc_root=tmp_path / 'absent')
        assert scan.confirmed is False
        assert scan.processes == ()
