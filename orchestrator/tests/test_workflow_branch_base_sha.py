"""Tests that the workflow keeps metadata.branch_base_sha true for its branch.

Two dispatch paths are exercised:
  (a) create_worktree path — the normal task-start flow where git_ops.create_worktree
      is called and WorktreeInfo.base_commit is used.
  (b) external-worktree path — eval mode where self.worktree is already set and
      the base commit is obtained via `git rev-parse HEAD`.

A branch with commits of its own only SEEDS the base (append=True, OLD-wins).
A branch with no commits of its own has the base re-stamped to its tip
(append=False, an overwrite) at dispatch and after every rebase (task 7332), so
branch_is_degenerate keeps recognising it after the orchestrator moved it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import pydantic_spec

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, WorktreeInfo
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)
    # _setup_worktree_and_artifacts now reads live status at dispatch before
    # claiming 'in-progress' (commit 57e25f1260); None → non-terminal → proceed.
    scheduler.get_status = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.worktree_base = project_root / '.worktrees'
    git_ops.head_if_no_commits_beyond = AsyncMock(return_value=None)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf


@pytest.mark.asyncio
async def test_setup_worktree_writes_branch_base_sha(tmp_path: Path):
    """create_worktree path: branch_base_sha is written from WorktreeInfo.base_commit."""
    wf = _make_workflow(project_root=tmp_path)
    base_sha = 'a' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha)
    )
    wf._sync_worktree_venvs = AsyncMock()

    # Stub the .task/ contamination check in _setup_worktree_and_artifacts
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # branch_base_sha must be written with append=True so existing metadata
    # (prd, files, modules, etc.) is not clobbered.
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


@pytest.mark.asyncio
async def test_setup_worktree_writes_branch_base_sha_external_worktree(tmp_path: Path):
    """External-worktree (eval) path: branch_base_sha comes from git rev-parse HEAD."""
    wf = _make_workflow(project_root=tmp_path)
    # Pre-set worktree to trigger the external-worktree (eval) path
    wf.worktree = tmp_path
    base_sha = 'b' * 40

    # Mock the git rev-parse HEAD subprocess
    fake_proc = MagicMock()
    fake_proc.returncode = 0  # explicit 0 so the new returncode check doesn't misfire
    fake_proc.communicate = AsyncMock(return_value=(base_sha.encode() + b'\n', b''))

    with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # branch_base_sha must be written with append=True
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


@pytest.mark.asyncio
async def test_setup_worktree_external_rev_parse_failure_logs_warning_and_falls_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Rev-parse failure leaves _base_commit as None so consumers gate correctly.

    Unlike the update_task soft-fail (which keeps a valid SHA in memory because
    only the metadata write fails), this path has no SHA to record at all —
    `_base_commit` is left None.  Downstream consumers gate safely:
      - workflow.py `if self._merge_sha and self._base_commit:` (truthy, None is falsy)
      - workflow.py `and self._base_commit is not None` (identity check, correctly skips)
    The falsy `if base_commit:` gate at line 497 still skips the update_task write.
    """
    wf = _make_workflow(project_root=tmp_path)
    wf.worktree = tmp_path  # trigger external-worktree path

    fake_proc = MagicMock()
    fake_proc.returncode = 128
    fake_proc.communicate = AsyncMock(
        return_value=(b'', b'fatal: not a git repository\n')
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'), \
         patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # (a) must not raise — fall-through, not crash (no explicit assertion needed)
    # (b) base_commit is None — identity check distinguishes None from '' or other falsy values
    assert wf._base_commit is None, f'expected _base_commit is None, got {wf._base_commit!r}'
    # (c) update_task not called — `if base_commit:` gate skips it when falsy (None is falsy)
    cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
    # (d) exactly one WARNING about the failure, surfacing task_id, rev-parse, and stderr
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'not a git repository' in msg, f'stderr not in warning: {msg!r}'


@pytest.mark.asyncio
async def test_setup_worktree_update_task_soft_fail_logs_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Soft-fail coverage: update_task exception must not propagate; warning must be logged.

    The existing except-branch at workflow.py:491-496 swallows exceptions from
    scheduler.update_task and logs a warning.  This test pins that behavior so
    a future refactor cannot silently break the soft-fail contract.

    Assertions:
      (a) no exception propagates from _setup_worktree_and_artifacts
      (b) wf._base_commit is set from create_worktree before update_task is called
      (c) the WARNING log contains task_id, 'branch_base_sha', and 'citation-grep'
      (d) update_task was awaited exactly once (proving the except branch was entered,
          not short-circuited by `if base_commit:`)
    """
    wf = _make_workflow(project_root=tmp_path)
    # Leave wf.worktree = None so the create_worktree branch fires
    base_sha = 'c' * 40

    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha)
    )
    wf._sync_worktree_venvs = AsyncMock()
    wf.scheduler.update_task = AsyncMock(side_effect=Exception('mcp down'))

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'), \
         patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')

    # (a) no exception propagated (test would have failed above if it raised)
    # (b) _base_commit set from WorktreeInfo before update_task attempted
    assert wf._base_commit == base_sha, (
        f'expected _base_commit={base_sha!r}, got {wf._base_commit!r}'
    )
    # (c) one WARNING containing task_id, branch_base_sha keyword, and citation-grep
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'branch_base_sha' in msg, f"'branch_base_sha' not in warning: {msg!r}"
    # (d) update_task was called (proving the except branch fired, not `if base_commit:` skip)
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


# ── task 7332: re-stamp the base of a branch the orchestrator moved ─────────


@pytest.fixture
def real_git_ops(tmp_path: Path) -> GitOps:
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(config, seed_repo(tmp_path / 'repo'))


async def _advance_main_by_landing(git_ops: GitOps, tid: str) -> str:
    """Land a one-file ``task/<tid>`` on main and return main's new sha."""
    info = await git_ops.create_worktree(tid)
    (info.path / f'{tid}.py').write_text(f'{tid} = True\n')
    await git_ops.commit(info.path, f'Add {tid}')
    result = await git_ops.merge_to_main(info.path, tid)
    assert result.success and result.merge_commit and result.merge_worktree
    assert (await git_ops.advance_main(result.merge_commit)).result == 'advanced'
    await git_ops.cleanup_merge_worktree(result.merge_worktree)
    await git_ops.cleanup_worktree(info.path, tid)
    return await git_ops.get_main_sha()


def _workflow_on(git_ops: GitOps, worktree: Path, tmp_path: Path) -> TaskWorkflow:
    wf = _make_workflow(project_root=tmp_path)
    wf.worktree = worktree
    wf.git_ops.head_if_no_commits_beyond = git_ops.head_if_no_commits_beyond
    return wf


@pytest.mark.asyncio
class TestHeadIfNoCommitsBeyond:
    async def test_a_branch_without_commits_of_its_own_answers_its_tip(
        self, real_git_ops: GitOps,
    ):
        info = await real_git_ops.create_worktree('101')

        assert await real_git_ops.head_if_no_commits_beyond(info.path, 'main') == (
            await real_git_ops.get_main_sha()
        )

    async def test_a_branch_with_a_commit_of_its_own_answers_none(
        self, real_git_ops: GitOps,
    ):
        info = await real_git_ops.create_worktree('101')
        (info.path / 'work.py').write_text('work = 1\n')
        await real_git_ops.commit(info.path, 'impl(101): work')

        assert await real_git_ops.head_if_no_commits_beyond(info.path, 'main') is None

    async def test_an_unresolvable_base_answers_none(self, real_git_ops: GitOps):
        info = await real_git_ops.create_worktree('101')

        assert await real_git_ops.head_if_no_commits_beyond(
            info.path, 'no-such-ref',
        ) is None


@pytest.mark.asyncio
class TestRedispatchRestampsAMovedZeroCommitBranch:
    async def test_a_zero_commit_branch_moved_onto_a_later_main_is_restamped(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        await real_git_ops.create_worktree('101')
        later_main = await _advance_main_by_landing(real_git_ops, '900')
        reused = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, reused.path, tmp_path)
        wf.worktree = None
        wf.git_ops.create_worktree = AsyncMock(return_value=reused)
        wf._sync_worktree_venvs = AsyncMock()

        with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
            await wf._setup_worktree_and_artifacts('task/101')

        assert reused.base_commit == later_main
        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': later_main}, append=False,
        )

    async def test_a_branch_with_commits_of_its_own_only_seeds_its_base(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        first = await real_git_ops.create_worktree('101')
        (first.path / 'work.py').write_text('work = 1\n')
        await real_git_ops.commit(first.path, 'impl(101): work')
        await _advance_main_by_landing(real_git_ops, '900')
        reused = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, reused.path, tmp_path)
        wf.worktree = None
        wf.git_ops.create_worktree = AsyncMock(return_value=reused)
        wf._sync_worktree_venvs = AsyncMock()

        with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
            await wf._setup_worktree_and_artifacts('task/101')

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': reused.base_commit}, append=True,
        )


@pytest.mark.asyncio
class TestRestampAfterRebase:
    async def test_a_zero_commit_branch_rebased_onto_main_is_restamped(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        info = await real_git_ops.create_worktree('101')
        later_main = await _advance_main_by_landing(real_git_ops, '900')
        assert await real_git_ops.rebase_onto_main(info.path)
        wf = _workflow_on(real_git_ops, info.path, tmp_path)

        assert await wf._restamp_branch_base_if_no_own_commits('main') is True

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': later_main}, append=False,
        )

    async def test_a_branch_with_commits_of_its_own_keeps_its_base(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        info = await real_git_ops.create_worktree('101')
        (info.path / 'work.py').write_text('work = 1\n')
        await real_git_ops.commit(info.path, 'impl(101): work')
        await _advance_main_by_landing(real_git_ops, '900')
        assert await real_git_ops.rebase_onto_main(info.path)
        wf = _workflow_on(real_git_ops, info.path, tmp_path)

        assert await wf._restamp_branch_base_if_no_own_commits('main') is False

        cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()

    @pytest.mark.parametrize(
        'reply',
        [{'side_effect': RuntimeError('mcp down')}, {'return_value': False}],
        ids=['raises', 'refused'],
    )
    async def test_a_failed_restamp_write_is_logged_and_never_raised(
        self, real_git_ops: GitOps, tmp_path: Path,
        caplog: pytest.LogCaptureFixture, reply: dict,
    ):
        info = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, info.path, tmp_path)
        wf.scheduler.update_task = AsyncMock(**reply)

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await wf._restamp_branch_base_if_no_own_commits('main') is True

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and 'branch_base_sha' in warnings[0], warnings

    async def test_an_unreadable_branch_is_logged_and_never_restamped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        wf = _make_workflow(project_root=tmp_path)
        wf.worktree = tmp_path
        wf.git_ops.head_if_no_commits_beyond = AsyncMock(side_effect=OSError('gone'))

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await wf._restamp_branch_base_if_no_own_commits('main') is False

        cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
        assert any(r.levelno == logging.WARNING for r in caplog.records)
