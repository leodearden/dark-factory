"""Tests for orchestrator.evals.snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
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


class TestEvalWorktreeRoot:
    """snapshots.eval_worktree_root must place eval worktrees OUTSIDE the repo.

    Eval worktrees nested UNDER project_root let a pytest/pyright/ruff/cargo run
    inside them walk up and collect the live main repo's ancestor config (root
    conftest.py sys.path-injects + pre-imports CURRENT-main <subproject>/src,
    shadowing the fixture's pinned pre_task_commit code — Defect B, task 2881).
    Relocating to a sibling of the repo removes that ancestor chain entirely.
    """

    def test_returns_project_name_qualified_sibling(self) -> None:
        assert snapshots.eval_worktree_root(Path('/a/b/repo')) == Path(
            '/a/b/repo-eval-worktrees'
        )

    def test_result_is_outside_the_repo(self) -> None:
        p = Path('/a/b/repo')
        assert not snapshots.eval_worktree_root(p).is_relative_to(p)

    def test_eval_worktree_substring_preserved(self) -> None:
        # Guards fused-memory's reconciliation project-scope anchor
        # (config/schema.py:457), which substring-matches 'eval-worktree'.
        p = Path('/a/b/repo')
        assert 'eval-worktree' in snapshots.eval_worktree_root(p).name

    def test_accepts_str_or_path(self) -> None:
        from_str = snapshots.eval_worktree_root('/a/b/repo')
        from_path = snapshots.eval_worktree_root(Path('/a/b/repo'))
        assert from_str == from_path == Path('/a/b/repo-eval-worktrees')


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


class TestEvalWorktreeHermeticity:
    """create_eval_worktree must place worktrees OUTSIDE the repo (Defect B).

    A nested worktree lets a plain pytest inside it walk up to the live main
    repo's ancestor ``conftest.py``, which ``sys.path``-injects + pre-imports
    CURRENT-main packages into ``sys.modules`` and shadows the fixture's pinned
    ``pre_task_commit`` code. Relocating the worktree to a sibling of the repo
    removes that ancestor chain entirely (task 2881).
    """

    def test_worktree_created_outside_repo(
        self, tmp_repo: tuple[Path, str, str]
    ) -> None:
        """The created worktree is a sibling of the repo, not nested under it."""
        repo, first, _second = tmp_repo

        worktree_path, _ = asyncio.run(
            create_eval_worktree(repo, 'test_task', first)
        )
        try:
            assert not worktree_path.is_relative_to(repo), (
                f'worktree {worktree_path} must NOT be nested under {repo}'
            )
            assert worktree_path.is_relative_to(
                snapshots.eval_worktree_root(repo)
            ), f'worktree must live under {snapshots.eval_worktree_root(repo)}'
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', str(worktree_path)],
                cwd=str(repo),
                capture_output=True,
            )

    def test_pinned_code_not_shadowed_by_ancestor_conftest(
        self, tmp_path: Path
    ) -> None:
        """A plain subprocess pytest inside the worktree imports the PINNED code.

        Reproduces the RCA (Defect B) with a minimal two-commit repo:

        * commit A — 'fixture era': a root ``conftest.py`` that ``sys.path``-
          inserts ``pkg/src`` and pre-imports ``mymod`` (VALUE='fixture_era');
          ``pkg/tests/test_mymod.py`` asserts the fixture-era value; NO in-tree
          pytest anchor (so a NESTED worktree's rootdir escapes upward).
        * commit B / HEAD — 'current main': adds a root ``pytest.ini`` anchoring
          rootdir=repo and flips ``mymod`` to VALUE='current_main'. This is the
          live-repo ancestor tree that only shadows a nested worktree.

        A subprocess pytest (fresh ``sys.modules`` — essential, the shadow is a
        pre-import cache effect) inside the RELOCATED (sibling) worktree imports
        the pinned fixture-era value and passes with no flags. Pre-relocation
        (nested) the live-repo ancestor conftest pre-imports current_main and
        the suite fails.
        """
        repo = tmp_path / 'repo'
        (repo / 'pkg' / 'src').mkdir(parents=True)
        (repo / 'pkg' / 'tests').mkdir(parents=True)

        _git(['init', '-q', '-b', 'main'], cwd=repo)
        _git(['config', 'user.email', 'test@example.com'], cwd=repo)
        _git(['config', 'user.name', 'Test User'], cwd=repo)
        _git(['config', 'commit.gpgsign', 'false'], cwd=repo)

        # commit A — fixture era (checked out into the worktree)
        (repo / 'conftest.py').write_text(
            'import sys\n'
            'from pathlib import Path\n'
            "sys.path.insert(0, str(Path(__file__).parent / 'pkg' / 'src'))\n"
            'import mymod  # noqa: E402,F401\n'
        )
        (repo / 'pkg' / 'src' / 'mymod.py').write_text("VALUE = 'fixture_era'\n")
        (repo / 'pkg' / 'tests' / 'test_mymod.py').write_text(
            'import mymod\n'
            'def test_value():\n'
            "    assert mymod.VALUE == 'fixture_era'\n"
        )
        _git(['add', '-A'], cwd=repo)
        _git(['commit', '-q', '-m', 'A: fixture era'], cwd=repo)
        commit_a = _git(['rev-parse', 'HEAD'], cwd=repo)

        # commit B / HEAD — current main (the live-repo ancestor tree)
        (repo / 'pytest.ini').write_text('[pytest]\n')
        (repo / 'pkg' / 'src' / 'mymod.py').write_text("VALUE = 'current_main'\n")
        _git(['add', '-A'], cwd=repo)
        _git(['commit', '-q', '-m', 'B: current main'], cwd=repo)

        worktree_path, _ = asyncio.run(
            create_eval_worktree(repo, 'shadow_task', commit_a)
        )
        try:
            # Scrub PYTEST_ADDOPTS/PYTEST_CURRENT_TEST so the outer pytest run's
            # options/state do not leak into the child invocation.
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ('PYTEST_ADDOPTS', 'PYTEST_CURRENT_TEST')
            }
            proc = subprocess.run(
                [
                    sys.executable, '-m', 'pytest',
                    'pkg/tests/test_mymod.py',
                    '-p', 'no:cacheprovider', '-q',
                ],
                cwd=str(worktree_path),
                env=env,
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, (
                'pinned fixture-era code was shadowed by the live-repo '
                f'ancestor conftest (rc={proc.returncode}):\n'
                f'{proc.stdout}\n{proc.stderr}'
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
