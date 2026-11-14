"""``own_work_base``: the base whose ``...HEAD`` range is the branch's own work.

Every test runs against real git repositories under ``tmp_path``: the
answer is git's, so a fake would only test the fake.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _merged_in_branch import MergedInBranch, branch_with_main_merged_in, git

from orchestrator.branch_base import own_work_base


def _files_since(checkout: Path, base: str) -> list[str]:
    return git(checkout, 'diff', '--name-only', f'{base}...HEAD').splitlines()


def _branch_forked_in_parallel(merged: MergedInBranch, tmp_path: Path) -> Path:
    """A second task branch from the creation sha, with main advanced past it and never merged in."""
    worktree = tmp_path / 'parallel'
    git(merged.repo, 'worktree', 'add', '-q', '-b', 'task/51', str(worktree), merged.creation_sha)
    (worktree / 'a.py').write_text('a = 3\n')
    git(worktree, 'commit', '-q', '-am', 'parallel task work')
    return worktree


@pytest.mark.asyncio
class TestOwnWorkBase:
    async def test_main_merged_in_bases_on_the_merged_main_tip(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)

        base = await own_work_base(merged.worktree, 'main', merged.creation_sha)

        assert base == merged.main_tip
        fork_point = git(merged.worktree, 'merge-base', 'HEAD', 'main')
        own_work = git(merged.worktree, 'diff', '--name-only', fork_point, 'HEAD').splitlines()
        assert _files_since(merged.worktree, base) == own_work == ['a.py']
        assert 'm1.py' in _files_since(merged.worktree, merged.creation_sha)

    async def test_main_advanced_in_parallel_bases_on_the_creation_sha(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)
        worktree = _branch_forked_in_parallel(merged, tmp_path)

        base = await own_work_base(worktree, 'main', None)

        assert base == merged.creation_sha
        assert _files_since(worktree, base) == ['a.py']

    async def test_main_already_containing_head_keeps_the_stored_base(self, tmp_path: Path):
        merged = branch_with_main_merged_in(tmp_path)
        git(merged.repo, 'merge', '-q', '--no-ff', '--no-edit', 'task/50')
        assert git(merged.worktree, 'merge-base', 'HEAD', 'main') == git(
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

    async def test_an_unresolvable_main_is_logged_with_the_fallback_it_chose(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        merged = branch_with_main_merged_in(tmp_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.branch_base'):
            base = await own_work_base(merged.worktree, 'no-such-branch', merged.creation_sha)

        [warning] = [r for r in caplog.records if r.name == 'orchestrator.branch_base']
        assert warning.levelno == logging.WARNING
        for named in (str(merged.worktree), 'no-such-branch', base):
            assert named in warning.getMessage()

    async def test_main_already_containing_head_falls_back_silently(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        merged = branch_with_main_merged_in(tmp_path)
        git(merged.repo, 'merge', '-q', '--no-ff', '--no-edit', 'task/50')

        with caplog.at_level(logging.WARNING, logger='orchestrator.branch_base'):
            await own_work_base(merged.worktree, 'main', merged.creation_sha)

        assert not [r for r in caplog.records if r.name == 'orchestrator.branch_base']
