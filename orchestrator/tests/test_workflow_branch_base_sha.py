"""Tests that the workflow keeps metadata.branch_base_sha true for its branch.

Two dispatch paths are exercised:
  (a) create_worktree path — the normal task-start flow where git_ops.create_worktree
      is called and WorktreeInfo.base_commit is used.
  (b) external-worktree path — eval mode where self.worktree is already set and
      the base commit is obtained via `git rev-parse HEAD`.

A branch with commits of its own, and an eval worktree, only SEED the base
(append=True, OLD-wins). A branch with no commits of its own (its ref's tip is
reachable from main) has the base re-stamped to its tip (append=False, an
overwrite) at dispatch and after every successful rebase (task 7332), so
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
from shared.git_async import run_git

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, WorktreeInfo
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    recorded_base: str | None = None,
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked.

    *recorded_base* is the task's ``metadata.branch_base_sha`` as dispatched.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    if recorded_base is not None:
        assignment.task['metadata'] = {'branch_base_sha': recorded_base}
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

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    wf._sync_worktree_venvs = AsyncMock()
    return wf


async def _setup(wf: TaskWorkflow) -> None:
    """Dispatch-time setup, with the .task/ contamination scan stubbed."""
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('101')


async def _restamp(wf: TaskWorkflow) -> bool:
    return await wf._restamp_branch_base_if_no_own_commits()


@pytest.mark.asyncio
async def test_setup_worktree_restamps_a_branch_without_own_commits(tmp_path: Path):
    """create_worktree path: a branch with no own commits gets its tip as its base, overwritten."""
    wf = _make_workflow(project_root=tmp_path)
    base_sha = 'a' * 40
    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha)
    )
    wf.git_ops.branch_tip_if_no_commits_beyond = AsyncMock(return_value=base_sha)

    await _setup(wf)

    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=False
    )


@pytest.mark.asyncio
async def test_setup_worktree_external_worktree_only_seeds_its_head(tmp_path: Path):
    """External-worktree (eval) path: the HEAD from git rev-parse is seeded, never re-stamped."""
    wf = _make_workflow(project_root=tmp_path)
    wf.worktree = tmp_path
    base_sha = 'b' * 40
    wf.git_ops.branch_tip_if_no_commits_beyond = AsyncMock(return_value=base_sha)

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(base_sha.encode() + b'\n', b''))

    with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)):
        await _setup(wf)

    cast(AsyncMock, wf.git_ops.branch_tip_if_no_commits_beyond).assert_not_awaited()
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=True
    )


@pytest.mark.asyncio
async def test_setup_worktree_external_rev_parse_failure_logs_warning_and_falls_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Rev-parse failure leaves _base_commit as None so consumers gate correctly.

    Unlike a failed branch_base_sha write (which keeps a valid SHA in memory
    because only the metadata write fails), this path has no SHA to record at
    all — `_base_commit` is left None, and the `if base_commit` gate skips the
    write.
    """
    wf = _make_workflow(project_root=tmp_path)
    wf.worktree = tmp_path  # trigger external-worktree path

    fake_proc = MagicMock()
    fake_proc.returncode = 128
    fake_proc.communicate = AsyncMock(
        return_value=(b'', b'fatal: not a git repository\n')
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'), \
         patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=fake_proc)):
        await _setup(wf)

    assert wf._base_commit is None
    cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'not a git repository' in msg, f'stderr not in warning: {msg!r}'


@pytest.mark.asyncio
async def test_setup_worktree_update_task_soft_fail_logs_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A raising re-stamp write is logged once, never propagated, and leaves _base_commit set."""
    wf = _make_workflow(project_root=tmp_path)
    base_sha = 'c' * 40
    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=tmp_path, base_commit=base_sha)
    )
    wf.git_ops.branch_tip_if_no_commits_beyond = AsyncMock(return_value=base_sha)
    wf.scheduler.update_task = AsyncMock(side_effect=Exception('mcp down'))

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
        await _setup(wf)

    assert wf._base_commit == base_sha
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f'expected 1 warning, got {len(warnings)}: {[r.getMessage() for r in warnings]}'
    msg = warnings[0].getMessage()
    assert '101' in msg, f'task_id not in warning: {msg!r}'
    assert 'branch_base_sha' in msg, f"'branch_base_sha' not in warning: {msg!r}"
    cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
        '101', {'branch_base_sha': base_sha}, append=False
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


async def _commit_own_work(git_ops: GitOps, worktree: Path) -> str | None:
    (worktree / 'work.py').write_text('work = 1\n')
    return await git_ops.commit(worktree, 'impl(101): work')


def _workflow_on(
    git_ops: GitOps, tmp_path: Path, *, recorded_base: str | None = None,
) -> TaskWorkflow:
    wf = _make_workflow(project_root=tmp_path, recorded_base=recorded_base)
    wf.git_ops.config = git_ops.config
    wf.git_ops.branch_tip_if_no_commits_beyond = git_ops.branch_tip_if_no_commits_beyond
    return wf


async def _dispatch(wf: TaskWorkflow, info: WorktreeInfo) -> None:
    wf.git_ops.create_worktree = AsyncMock(return_value=info)
    await _setup(wf)


@pytest.mark.asyncio
class TestBranchTipIfNoCommitsBeyond:
    async def test_a_branch_without_commits_of_its_own_answers_its_tip(
        self, real_git_ops: GitOps,
    ):
        await real_git_ops.create_worktree('101')

        assert await real_git_ops.branch_tip_if_no_commits_beyond('task/101', 'main') == (
            await real_git_ops.get_main_sha()
        )

    async def test_a_branch_with_a_commit_of_its_own_answers_none(
        self, real_git_ops: GitOps,
    ):
        info = await real_git_ops.create_worktree('101')
        await _commit_own_work(real_git_ops, info.path)

        assert await real_git_ops.branch_tip_if_no_commits_beyond('task/101', 'main') is None

    async def test_the_ref_is_measured_not_a_head_detached_onto_main(
        self, real_git_ops: GitOps,
    ):
        info = await real_git_ops.create_worktree('101')
        await _commit_own_work(real_git_ops, info.path)
        assert (await run_git(['git', 'checkout', '--detach', 'main'], info.path)).returncode == 0

        assert await real_git_ops.branch_tip_if_no_commits_beyond('task/101', 'main') is None

    async def test_a_missing_branch_or_base_answers_none(self, real_git_ops: GitOps):
        await real_git_ops.create_worktree('101')

        assert await real_git_ops.branch_tip_if_no_commits_beyond('task/999', 'main') is None
        assert await real_git_ops.branch_tip_if_no_commits_beyond(
            'task/101', 'no-such-ref',
        ) is None


@pytest.mark.asyncio
class TestDispatchRestamp:
    async def test_a_zero_commit_branch_moved_onto_a_later_main_is_restamped(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        first = await real_git_ops.create_worktree('101')
        later_main = await _advance_main_by_landing(real_git_ops, '900')
        reused = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=first.base_commit)

        await _dispatch(wf, reused)

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': later_main}, append=False,
        )

    async def test_a_base_already_at_the_tip_is_not_rewritten(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        info = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=info.base_commit)

        await _dispatch(wf, info)

        cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()

    async def test_a_branch_with_commits_of_its_own_only_seeds_its_base(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        first = await real_git_ops.create_worktree('101')
        await _commit_own_work(real_git_ops, first.path)
        await _advance_main_by_landing(real_git_ops, '900')
        reused = await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=first.base_commit)

        await _dispatch(wf, reused)

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': reused.base_commit}, append=True,
        )

    async def test_a_fallback_base_at_head_never_overwrites_a_branch_with_work(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        """create_worktree falls back to HEAD when merge-base fails; that is no proof of emptiness."""
        info = await real_git_ops.create_worktree('101')
        head = await _commit_own_work(real_git_ops, info.path)
        assert head is not None
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=info.base_commit)

        await _dispatch(wf, WorktreeInfo(path=info.path, base_commit=head))

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': head}, append=True,
        )


@pytest.mark.asyncio
class TestRestampAfterRebase:
    async def test_a_zero_commit_branch_rebased_onto_main_is_restamped(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        info = await real_git_ops.create_worktree('101')
        later_main = await _advance_main_by_landing(real_git_ops, '900')
        assert await real_git_ops.rebase_onto_main(info.path)
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=info.base_commit)

        assert await _restamp(wf) is True
        assert await _restamp(wf) is True

        cast(AsyncMock, wf.scheduler.update_task).assert_awaited_once_with(
            '101', {'branch_base_sha': later_main}, append=False,
        )

    async def test_a_branch_with_commits_of_its_own_keeps_its_base(
        self, real_git_ops: GitOps, tmp_path: Path,
    ):
        info = await real_git_ops.create_worktree('101')
        await _commit_own_work(real_git_ops, info.path)
        await _advance_main_by_landing(real_git_ops, '900')
        assert await real_git_ops.rebase_onto_main(info.path)
        wf = _workflow_on(real_git_ops, tmp_path, recorded_base=info.base_commit)

        assert await _restamp(wf) is False

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
        await real_git_ops.create_worktree('101')
        wf = _workflow_on(real_git_ops, tmp_path)
        wf.scheduler.update_task = AsyncMock(**reply)

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await _restamp(wf) is True

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and 'branch_base_sha' in warnings[0], warnings

    async def test_an_unreadable_branch_is_logged_and_never_restamped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        wf = _make_workflow(project_root=tmp_path)
        wf.git_ops.branch_tip_if_no_commits_beyond = AsyncMock(side_effect=OSError('gone'))

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await _restamp(wf) is False

        cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
        assert any(r.levelno == logging.WARNING for r in caplog.records)
