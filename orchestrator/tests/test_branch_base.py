"""``own_work_base``: the base whose ``...HEAD`` range is the branch's own work.

Every test runs against real git repositories under ``tmp_path``: the
answer is git's, so a fake would only test the fake.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _merged_in_branch import MergedInBranch, branch_with_main_merged_in

from orchestrator.branch_base import own_work_base


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _files_since(checkout: Path, base: str) -> list[str]:
    return _git(checkout, 'diff', '--name-only', f'{base}...HEAD').splitlines()


def _branch_forked_in_parallel(merged: MergedInBranch, tmp_path: Path) -> Path:
    """A second task branch from the creation sha, with main advanced past it and never merged in."""
    worktree = tmp_path / 'parallel'
    _git(merged.repo, 'worktree', 'add', '-q', '-b', 'task/51', str(worktree), merged.creation_sha)
    (worktree / 'a.py').write_text('a = 3\n')
    _git(worktree, 'commit', '-q', '-am', 'parallel task work')
    return worktree


@pytest.mark.asyncio
class TestOwnWorkBase:
    async def test_main_merged_in_bases_on_the_merged_main_tip(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)

        base = await own_work_base(merged.worktree, 'main', merged.creation_sha)

        assert base == merged.main_tip
        fork_point = _git(merged.worktree, 'merge-base', 'HEAD', 'main')
        own_work = _git(merged.worktree, 'diff', '--name-only', fork_point, 'HEAD').splitlines()
        assert _files_since(merged.worktree, base) == own_work == ['a.py']
        assert 'm1.py' in _files_since(merged.worktree, merged.creation_sha)

    async def test_main_advanced_in_parallel_bases_on_the_creation_sha(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)
        worktree = _branch_forked_in_parallel(merged, tmp_path)

        base = await own_work_base(worktree, 'main', merged.creation_sha)

        assert base == merged.creation_sha
        assert _files_since(worktree, base) == ['a.py']

    async def test_main_already_containing_head_keeps_the_stored_base(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)
        _git(merged.repo, 'merge', '-q', '--no-ff', '--no-edit', 'task/50')
        assert _git(merged.worktree, 'merge-base', 'HEAD', 'main') == _git(
            merged.worktree, 'rev-parse', 'HEAD',
        )

        base = await own_work_base(merged.worktree, 'main', merged.creation_sha)

        assert base == merged.creation_sha
        assert _files_since(merged.worktree, base)

    async def test_an_unresolvable_main_falls_back_to_the_stored_base(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)

        base = await own_work_base(merged.worktree, 'no-such-branch', merged.creation_sha)

        assert base == merged.creation_sha

    async def test_an_unresolvable_main_without_a_stored_base_is_the_main_name(
        self, tmp_path: Path,
    ):
        merged = branch_with_main_merged_in(tmp_path)

        assert await own_work_base(merged.worktree, 'no-such-branch', None) == 'no-such-branch'
