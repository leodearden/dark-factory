"""Merge-lane regressions for a member stacked on an unlanded base (task 5618).

A train member stacked onto a predecessor that later left without landing
carries the predecessor's commits.  The merge lane must merge, re-drive and
attribute such a member on its OWN delta.  These tests drive the lane through
its public seams against real temp git repositories:

* ``TestAdmissionUnstacks`` calls the module function ``classify_and_merge``
  (the single-request merge core) directly.  A train tip handed to it keeps
  its co-members' commits.
* ``TestDerailRedriveUnstacks`` and ``TestTrainLanding`` push a coalesce
  ``GroupMergeRequest`` through the real merger loop (``queue.put`` +
  ``worker.run()``), with the real ``_do_train_merge``: nothing in
  ``orchestrator.merge_queue`` is patched.

Base fixture: main has shared.txt and other.txt.  Stacking is done with the
production ``GitOps.stack_train_branches``.  After stacking, main advances
with a commit that edits shared.txt, so the base no longer merges cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.branch_stack import (
    STACKED_ON_UNLANDED_BASE_REASON_PREFIX,
    StackBaseLedger,
    UnstackResult,
)
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    Decided,
    GroupMergeRequest,
    MergedOk,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
    classify_and_merge,
)
from orchestrator.merge_types import QueuedBranch

# ---------------------------------------------------------------------------
# Fixtures (copied from test_merge_guard_pipeline.py, per-file convention)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _git(repo, 'init', '-b', 'main')
    await _git(repo, 'config', 'user.email', 'test@test.com')
    await _git(repo, 'config', 'user.name', 'Test')
    await _commit(repo, {'shared.txt': 'v0\n', 'other.txt': 'v0\n'}, 'initial')


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


def _make_request(
    task_id: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


async def _commit(cwd: Path, writes: dict[str, str], msg: str) -> str:
    """Write and commit exactly *writes*; a bare ``git add -A`` in the main
    checkout would stage the nested ``.worktrees/*`` checkouts as gitlinks."""
    for name, content in writes.items():
        (cwd / name).write_text(content)
    await _git(cwd, 'add', '--', *writes)
    await _git(cwd, 'commit', '-m', msg)
    return (await _git(cwd, 'rev-parse', 'HEAD')).strip()


async def _member(git_ops: GitOps, name: str, writes: dict[str, str]) -> Path:
    """task/<name> from main at git_ops.worktree_base / <name>, one commit."""
    wt = (await git_ops.create_worktree(name)).path
    await _commit(wt, writes, f'{name}: write files')
    return wt


async def _stack(git_ops: GitOps, anchor: str, member: str) -> None:
    result = await git_ops.stack_train_branches([anchor, member])
    assert result.survivors == [anchor, member]


async def _advance_main(git_ops: GitOps, writes: dict[str, str]) -> str:
    return await _commit(git_ops.project_root, writes, 'main advances')


async def _sha(git_ops: GitOps, ref: str) -> str:
    return (await _git(git_ops.project_root, 'rev-parse', '--verify', ref)).strip()


async def _diff_vs_main(git_ops: GitOps, branch: str) -> set[str]:
    out = await _git(git_ops.project_root, 'diff', '--name-only', f'main...{branch}')
    return {line for line in out.splitlines() if line}


def _ledger(git_ops: GitOps) -> StackBaseLedger:
    return StackBaseLedger(git_ops.project_root, _run)


async def _stack_m_on_unlanded_p(
    git_ops: GitOps, *, m_writes: dict[str, str], main_writes: dict[str, str],
) -> Path:
    """task/P edits shared.txt; task/M is stacked on it; main then advances."""
    await _member(git_ops, 'P', {'shared.txt': 'p\n', 'p_only.txt': 'p\n'})
    m_wt = await _member(git_ops, 'M', m_writes)
    await _stack(git_ops, 'P', 'M')
    await _advance_main(git_ops, main_writes)
    return m_wt


# ---------------------------------------------------------------------------
# Admission: classify_and_merge
# ---------------------------------------------------------------------------


async def _classify(git_ops: GitOps, request: MergeRequest) -> MergedOk | Decided:
    worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
    return await classify_and_merge(
        worker,
        request,
        await git_ops.get_main_sha(),
        speculative=False,
        started_monotonic=None,
    )


async def _admit(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, worktree: Path,
) -> MergedOk | Decided:
    return await _classify(git_ops, _make_request(task_id, worktree, config))


async def _merged_files(git_ops: GitOps, base: str, result: MergedOk) -> set[str]:
    merge_commit = result.merge_result.merge_commit
    assert merge_commit is not None
    changed = await _git(git_ops.project_root, 'diff', '--name-only', base, merge_commit)
    return set(changed.split())


async def _cleanup(git_ops: GitOps, result: MergedOk | Decided) -> None:
    merge_wt = (
        result.merge_wt if isinstance(result, MergedOk)
        else result.merge_result.merge_worktree if result.merge_result is not None
        else None
    )
    if merge_wt is not None:
        await git_ops.cleanup_merge_worktree(merge_wt)


@pytest.mark.asyncio
class TestAdmissionUnstacks:
    async def test_member_stacked_on_unlanded_predecessor_merges_its_own_delta(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        m_wt = await _stack_m_on_unlanded_p(
            git_ops, m_writes={'m.txt': 'm\n'}, main_writes={'shared.txt': 'main\n'},
        )
        main_sha = await git_ops.get_main_sha()

        result = await _admit(git_ops, config, 'M', m_wt)

        try:
            assert isinstance(result, MergedOk), result
            assert await _merged_files(git_ops, main_sha, result) == {'m.txt'}
            assert await _ledger(git_ops).base_of('task/M') is None
        finally:
            await _cleanup(git_ops, result)

    async def test_unstack_conflict_reports_only_the_members_own_conflicts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        m_wt = await _stack_m_on_unlanded_p(
            git_ops,
            m_writes={'m.txt': 'm\n', 'other.txt': 'm\n'},
            main_writes={'shared.txt': 'main\n', 'other.txt': 'main\n'},
        )
        tip_before = await _sha(git_ops, 'task/M')

        result = await _admit(git_ops, config, 'M', m_wt)

        try:
            assert isinstance(result, Decided), result
            assert result.outcome.status == 'blocked'
            reason = result.outcome.reason or ''
            assert reason.startswith(STACKED_ON_UNLANDED_BASE_REASON_PREFIX)
            assert 'other.txt' in reason
            assert 'task/P' in reason
            assert 'shared.txt' not in reason
            assert await _sha(git_ops, 'task/M') == tip_before
        finally:
            await _cleanup(git_ops, result)

    async def test_unstacked_branch_admission_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        plain_wt = await _member(git_ops, 'plain', {'plain.txt': 'plain\n'})

        result = await _admit(git_ops, config, 'plain', plain_wt)

        try:
            assert isinstance(result, MergedOk), result
        finally:
            await _cleanup(git_ops, result)

    async def test_train_tip_keeps_its_co_members_commits(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        await _member(git_ops, 'A', {'a.txt': 'a\n'})
        await _member(git_ops, 'T', {'t.txt': 't\n'})
        await _stack(git_ops, 'A', 'T')
        main_sha = await git_ops.get_main_sha()
        train = _coalesce_train(git_ops, config, members=['A', 'T'], statuses={})

        result = await _classify(git_ops, train.request)

        try:
            assert isinstance(result, MergedOk), result
            assert await _merged_files(git_ops, main_sha, result) == {'a.txt', 't.txt'}
            assert await _ledger(git_ops).base_of('task/T') == await _sha(git_ops, 'task/A')
        finally:
            await _cleanup(git_ops, result)


# ---------------------------------------------------------------------------
# Coalesce trains through the real merger loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Train:
    request: GroupMergeRequest
    mark_member_done: AsyncMock
    redrive_member: AsyncMock


def _coalesce_train(
    git_ops: GitOps,
    config: OrchestratorConfig,
    *,
    members: list[str],
    statuses: dict[str, str],
) -> _Train:
    tip = members[-1]
    mark_member_done = AsyncMock()
    redrive_member = AsyncMock()
    request = GroupMergeRequest(
        task_id=tip,
        branch=QueuedBranch.parse(tip, config.git.branch_prefix),
        worktree=git_ops.worktree_base / tip,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        train_id=f'coalesce-{tip}-test',
        member_task_ids=members,
        tip_branch=QueuedBranch.parse(tip, config.git.branch_prefix),
        tip_task_id=tip,
        status_check=AsyncMock(return_value=statuses),
        mark_member_done=mark_member_done,
        redrive_member=redrive_member,
    )
    return _Train(request, mark_member_done, redrive_member)


async def _run_through_merger(
    git_ops: GitOps, train: _Train, tmp_path: Path,
) -> MergeOutcome:
    queue: asyncio.Queue = asyncio.Queue()
    worker = SpeculativeMergeWorker(
        git_ops, queue,
        event_store=EventStore(db_path=tmp_path / 'events.db', run_id='unstack-train'),
    )
    await queue.put(train.request)
    worker_task = asyncio.create_task(worker.run())
    try:
        return await asyncio.wait_for(train.request.result, timeout=30)
    finally:
        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


async def _derail_coalesce_train(
    git_ops: GitOps,
    config: OrchestratorConfig,
    tmp_path: Path,
    *,
    members: list[str],
    statuses: dict[str, str],
) -> AsyncMock:
    """Run one coalesce train through the real merger loop; return redrive_member."""
    train = _coalesce_train(git_ops, config, members=members, statuses=statuses)
    outcome = await _run_through_merger(git_ops, train, tmp_path)
    assert outcome.status == 'blocked', outcome
    return train.redrive_member


def _redriven(redrive_member: AsyncMock) -> list[tuple]:
    return [call.args for call in redrive_member.await_args_list]


class _UnstackRaises(GitOps):
    """A GitOps whose un-stack fails with an exception it does not type."""

    async def unstack_from_unlanded_base(self, full_branch: str) -> UnstackResult:
        raise OSError(f'git could not be spawned to un-stack {full_branch}')


@pytest.mark.asyncio
class TestDerailRedriveUnstacks:
    async def test_predecessor_left_queue_member_unstacked_before_redrive(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        await _stack_m_on_unlanded_p(
            git_ops, m_writes={'m.txt': 'm\n'}, main_writes={'shared.txt': 'main\n'},
        )
        p_tip = await _sha(git_ops, 'task/P')

        redrive_member = await _derail_coalesce_train(
            git_ops, config, tmp_path,
            members=['P', 'M'],
            statuses={'P': 'pending', 'M': 'merge-deferred'},
        )

        assert _redriven(redrive_member) == [('M', False, None)]
        assert await _diff_vs_main(git_ops, 'task/M') == {'m.txt'}
        assert await _sha(git_ops, 'task/P') == p_tip

    async def test_tip_rebase_conflict_derail_unstacks_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        await _member(git_ops, 'A', {'shared.txt': 'a\n'})
        await _member(git_ops, 'T', {'t.txt': 't\n'})
        await _stack(git_ops, 'A', 'T')
        await _advance_main(git_ops, {'shared.txt': 'main\n'})
        a_tip = await _sha(git_ops, 'task/A')

        redrive_member = await _derail_coalesce_train(
            git_ops, config, tmp_path,
            members=['A', 'T'],
            statuses={'A': 'merge-deferred', 'T': 'merge-deferred'},
        )

        assert sorted(_redriven(redrive_member)) == [
            ('A', False, None), ('T', False, None),
        ]
        assert await _diff_vs_main(git_ops, 'task/T') == {'t.txt'}
        assert await _sha(git_ops, 'task/A') == a_tip

    async def test_unstack_conflict_does_not_block_redrive(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        await _member(git_ops, 'A', {'shared.txt': 'a\n'})
        await _member(git_ops, 'T', {'t.txt': 't\n', 'other.txt': 't\n'})
        await _stack(git_ops, 'A', 'T')
        await _advance_main(git_ops, {'shared.txt': 'main\n', 'other.txt': 'main\n'})
        t_tip = await _sha(git_ops, 'task/T')
        a_tip = await _sha(git_ops, 'task/A')

        redrive_member = await _derail_coalesce_train(
            git_ops, config, tmp_path,
            members=['A', 'T'],
            statuses={'A': 'merge-deferred', 'T': 'merge-deferred'},
        )

        assert ('T', False, None) in _redriven(redrive_member)
        assert await _sha(git_ops, 'task/T') == t_tip
        assert await _ledger(git_ops).base_of('task/T') == a_tip

    async def test_unstack_raising_does_not_block_redrive(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        await _member(git_ops, 'A', {'shared.txt': 'a\n'})
        await _member(git_ops, 'T', {'t.txt': 't\n'})
        await _stack(git_ops, 'A', 'T')
        await _advance_main(git_ops, {'shared.txt': 'main\n'})

        redrive_member = await _derail_coalesce_train(
            _UnstackRaises(git_ops.config, git_ops.project_root), config, tmp_path,
            members=['A', 'T'],
            statuses={'A': 'merge-deferred', 'T': 'merge-deferred'},
        )

        assert sorted(_redriven(redrive_member)) == [
            ('A', False, None), ('T', False, None),
        ]


# ---------------------------------------------------------------------------
# A stacked train that lands
# ---------------------------------------------------------------------------


@pytest.fixture
def no_op_verify_config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        test_command='true',
        lint_command='true',
        type_check_command='true',
    )


@pytest.mark.asyncio
class TestTrainLanding:
    async def test_stacked_train_lands_every_members_files(
        self, git_ops: GitOps, no_op_verify_config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        await _member(git_ops, 'A', {'a.txt': 'a\n'})
        await _member(git_ops, 'T', {'t.txt': 't\n'})
        await _stack(git_ops, 'A', 'T')
        main_before = await _advance_main(git_ops, {'other.txt': 'main\n'})
        train = _coalesce_train(
            git_ops, no_op_verify_config,
            members=['A', 'T'],
            statuses={'A': 'merge-deferred', 'T': 'merge-deferred'},
        )

        outcome = await _run_through_merger(git_ops, train, tmp_path)

        assert outcome.status == 'done', outcome
        landed = await _git(git_ops.project_root, 'diff', '--name-only', main_before, 'main')
        assert set(landed.split()) == {'a.txt', 't.txt'}
        assert sorted(
            call.args[0] for call in train.mark_member_done.await_args_list
        ) == ['A', 'T']
