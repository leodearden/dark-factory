"""Tests for the post-merge pyright equivalence check.

Covers:
- ``PostMergePyrightResult`` dataclass and ``.broken`` property
- ``_check_post_merge_pyright`` function — real-git behavioural tests
  (using hermetic stand-in type_check_command), classification / fail-open
  edge cases, and call-site integration with MergeWorker / SpeculativeMergeWorker.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    PostMergePyrightResult,
    _check_post_merge_pyright,
)


# ---------------------------------------------------------------------------
# Fixtures — shared real-git setup (mirrors TestCheckPostMergeEquivalence)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# Stand-in type_check_command that exits 0 on a clean tree and non-zero on
# a tree carrying a `.BROKEN_UNION` marker file.  This exercises real
# worktree creation + verbatim unscoped command execution without needing
# a real pyright binary or a real Protocol break.
_TYPE_CMD_CONDITIONAL = (
    'python3 -c "'
    "import sys, pathlib; "
    "sys.exit(1 if pathlib.Path('.BROKEN_UNION').exists() else 0)"
    '"'
)


def _make_module_config(
    prefix: str = 'subpkg',
    type_check_command: str | None = _TYPE_CMD_CONDITIONAL,
) -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix,
        test_command=None,
        lint_command=None,
        type_check_command=type_check_command,
    )


# ---------------------------------------------------------------------------
# PostMergePyrightResult unit tests
# ---------------------------------------------------------------------------


class TestPostMergePyrightResult:
    def test_empty_result_is_not_broken(self):
        result = PostMergePyrightResult()
        assert result.broken is False
        assert result.failing_subprojects == []
        assert result.detail == ''

    def test_non_empty_failing_subprojects_is_broken(self):
        result = PostMergePyrightResult(failing_subprojects=['subpkg'])
        assert result.broken is True

    def test_with_detail(self):
        result = PostMergePyrightResult(
            failing_subprojects=['subpkg'],
            detail='error: src/foo.py:10: Missing method bar',
        )
        assert result.broken is True
        assert 'bar' in result.detail


# ---------------------------------------------------------------------------
# _check_post_merge_pyright — real-git behavioural tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPostMergePyrightBehavioral:
    """Real-git tests using a hermetic stand-in type_check_command."""

    async def test_clean_tree_returns_not_broken(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Clean tree + command that exits 0 → result.broken is False."""
        # Create a branch with a simple file
        wt = (await git_ops.create_worktree('pyright-clean')).path
        (wt / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Add mod.py')

        merge_result = await git_ops.merge_to_main(wt, 'pyright-clean')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-clean', max_attempts=1,
            )
            advanced_sha = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced_sha is not None

            mc = _make_module_config()
            result = await _check_post_merge_pyright(
                advanced_sha, git_ops, config, [mc], task_id='pyright-clean-test',
            )
            assert result.broken is False
            assert result.failing_subprojects == []
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_broken_tree_returns_broken_with_prefix_and_detail(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Broken tree + command that exits non-zero → broken=True, prefix listed, detail non-empty."""
        # Create a branch with a "broken union" marker file
        wt = (await git_ops.create_worktree('pyright-broken')).path
        (wt / 'mod.py').write_text('x = 1\n')
        # The marker file triggers the stand-in type_check_command to exit 1
        (wt / '.BROKEN_UNION').write_text('broken\n')
        await git_ops.commit(wt, 'Add mod.py + .BROKEN_UNION marker')

        merge_result = await git_ops.merge_to_main(wt, 'pyright-broken')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='pyright-broken', max_attempts=1,
            )
            advanced_sha = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced_sha is not None

            mc = _make_module_config(prefix='subpkg')
            result = await _check_post_merge_pyright(
                advanced_sha, git_ops, config, [mc], task_id='pyright-broken-test',
            )
            assert result.broken is True
            assert 'subpkg' in result.failing_subprojects
            # detail carries command output (exit code or stderr surrogate)
            assert result.detail != ''
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_empty_module_configs_returns_clean(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Empty module_configs → no-op, returns clean (not broken)."""
        # We don't even need a real merge for this; use any valid SHA
        advanced_sha = await git_ops.get_main_sha()

        result = await _check_post_merge_pyright(
            advanced_sha, git_ops, config, [], task_id='pyright-empty-mc',
        )
        assert result.broken is False
        assert result.failing_subprojects == []

    async def test_module_config_with_no_type_check_command_skipped(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A ModuleConfig whose type_check_command=None → skipped, returns clean."""
        advanced_sha = await git_ops.get_main_sha()
        mc = _make_module_config(type_check_command=None)

        result = await _check_post_merge_pyright(
            advanced_sha, git_ops, config, [mc], task_id='pyright-no-cmd',
        )
        assert result.broken is False
        assert result.failing_subprojects == []
