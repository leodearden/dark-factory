"""Tests for orchestrator.evals.snapshots."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from orchestrator.evals import snapshots
from orchestrator.evals.snapshots import create_eval_worktree, get_diff


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in *cwd* and return stripped stdout."""
    return subprocess.run(
        ['git', *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a tiny git repo with two commits.

    Returns ``(repo_path, first_commit_sha, second_commit_sha)``.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()

    _git(['init', '-q', '-b', 'main'], cwd=repo)
    _git(['config', 'user.email', 'test@example.com'], cwd=repo)
    _git(['config', 'user.name', 'Test User'], cwd=repo)
    _git(['config', 'commit.gpgsign', 'false'], cwd=repo)

    (repo / 'README.md').write_text('first\n')
    _git(['add', 'README.md'], cwd=repo)
    _git(['commit', '-q', '-m', 'first commit'], cwd=repo)
    first = _git(['rev-parse', 'HEAD'], cwd=repo)

    (repo / 'README.md').write_text('second\n')
    _git(['add', 'README.md'], cwd=repo)
    _git(['commit', '-q', '-m', 'second commit'], cwd=repo)
    second = _git(['rev-parse', 'HEAD'], cwd=repo)

    return repo, first, second


class TestCreateEvalWorktreeHeadAssertion:
    """The defensive HEAD == pre_task_commit assertion in create_eval_worktree."""

    def test_head_matches_pre_task_commit_happy_path(
        self, tmp_repo: tuple[Path, str, str]
    ) -> None:
        repo, first, _second = tmp_repo

        worktree_path, run_id = asyncio.run(
            create_eval_worktree(repo, 'test_task', first)
        )

        try:
            assert worktree_path.exists()
            assert len(run_id) == 8
            head = _git(['rev-parse', 'HEAD'], cwd=worktree_path)
            assert head == first
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(worktree_path)],
                cwd=str(repo),
                capture_output=True,
            )

    def test_assertion_fires_on_drift(
        self,
        tmp_repo: tuple[Path, str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If git rev-parse HEAD returns the wrong SHA, raise RuntimeError."""
        repo, first, second = tmp_repo

        original_run = snapshots._run

        async def fake_run(cmd: list[str], cwd: Path) -> str:
            # Lie about HEAD: return `second` instead of whatever git says.
            if cmd[:2] == ['git', 'rev-parse'] and cmd[-1] == 'HEAD':
                return second
            return await original_run(cmd, cwd)

        monkeypatch.setattr(snapshots, '_run', fake_run)

        with pytest.raises(RuntimeError, match='HEAD mismatch for test_task'):
            asyncio.run(create_eval_worktree(repo, 'test_task', first))

        # Cleanup any worktree git left behind before the assertion fired.
        for child in (repo / '.eval-worktrees' / 'test_task').glob('run-*'):
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(child)],
                cwd=str(repo),
                capture_output=True,
            )

    def test_setup_commands_run_after_assertion(
        self, tmp_repo: tuple[Path, str, str]
    ) -> None:
        """Setup commands must run only after the HEAD assertion has passed.

        We verify this by passing a setup command that creates a marker file:
        if it ran, the file exists; if the assertion had fired, the worktree
        wouldn't exist at all.
        """
        repo, first, _second = tmp_repo

        worktree_path, _ = asyncio.run(
            create_eval_worktree(
                repo,
                'test_task',
                first,
                setup_commands=['touch SETUP_RAN'],
            )
        )

        try:
            assert (worktree_path / 'SETUP_RAN').exists(), (
                'setup_commands should have run after the HEAD assertion passed'
            )
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(worktree_path)],
                cwd=str(repo),
                capture_output=True,
            )


class TestGetDiff:
    """get_diff must diff the COMMITTED eval branch against the threaded base."""

    def test_get_diff_returns_committed_diff_vs_base(
        self, tmp_repo: tuple[Path, str, str]
    ) -> None:
        """A committed change on the eval branch shows up in the diff.

        Reproduces D1: the landed change is a COMMIT (not a working-tree
        edit), and a misleading ``<worktree>/.task/metadata.json`` points
        ``base_commit`` at HEAD. The old metadata-read + uncommitted-only
        fallback returned '' for this shape; threading the authoritative
        ``base_commit`` (here ``first``) yields the full committed diff.
        """
        repo, first, _second = tmp_repo

        worktree_path, _ = asyncio.run(
            create_eval_worktree(repo, 'gd_task', first)
        )
        try:
            # Land the change as a COMMIT on the detached eval branch, so it
            # is NOT visible to `git diff HEAD` (the old uncommitted fallback).
            (worktree_path / 'LANDED.py').write_text('X = "committed_marker"\n')
            _git(['add', 'LANDED.py'], cwd=worktree_path)
            _git(['commit', '-q', '-m', 'landed change'], cwd=worktree_path)

            # Misleading metadata.json: if get_diff still read it, base==HEAD
            # would make `git diff HEAD..HEAD` empty and mask the change.
            head = _git(['rev-parse', 'HEAD'], cwd=worktree_path)
            task_dir = worktree_path / '.task'
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / 'metadata.json').write_text(
                json.dumps({'base_commit': head})
            )

            diff = asyncio.run(get_diff(worktree_path, first))

            assert diff, 'expected a non-empty committed diff vs base'
            assert 'LANDED.py' in diff
            assert 'committed_marker' in diff
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(worktree_path)],
                cwd=str(repo),
                capture_output=True,
            )
