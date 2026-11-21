"""GitOps.create_worktree stacks a task branch on a predecessor's commit (task 10473).

``create_worktree(stack_on=StackTarget(...))`` starts the branch at that
commit, from the warm pool where there is one, and returns only once the
stack-base ledger answers the commit.  A declared-train successor is one user
of the same input: it stacks on its predecessor's tip, records it (task 7660)
and is still made cold.

Real temp git repositories throughout (``_stacked_repo``); failures are forced
only through a PATH git shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _stacked_repo import (
    add_worktree,
    commit_files,
    fail_git_calls_starting,
    ledger_of,
    make_repo,
    make_warm_repo,
    pool_lane,
    rev_parse,
)

from orchestrator.branch_stack import StackBaseRead, StackRecordState, StackTarget
from orchestrator.git_ops import GitOps, StackRecordUnconfirmed, _run

_TRAIN_SUCCESSOR = {'id': 'T', 'order': 1, 'members': ['P', 'M']}


@dataclass(frozen=True)
class Predecessor:
    git_ops: GitOps
    tip: str

    @property
    def root(self) -> Path:
        return self.git_ops.project_root

    @property
    def target(self) -> StackTarget:
        return StackTarget('task/P', self.tip)


async def _with_predecessor(git_ops: GitOps) -> Predecessor:
    """task/P carries p.txt above main; main then moves on, so P's tip is
    not where main is."""
    p_wt = await add_worktree(git_ops, 'P', 'main')
    tip = await commit_files(p_wt, {'p.txt': 'p\n'}, 'p1: write p.txt')
    await commit_files(git_ops.project_root, {'main.txt': 'main\n'}, 'main advances')
    return Predecessor(git_ops=git_ops, tip=tip)


@pytest_asyncio.fixture
async def cold(tmp_path: Path) -> Predecessor:
    return await _with_predecessor(await make_repo(tmp_path / 'repo'))


@pytest_asyncio.fixture
async def warm(tmp_path: Path) -> Predecessor:
    return await _with_predecessor(await make_warm_repo(tmp_path / 'repo'))


async def _branch_exists(root: Path, full_branch: str) -> bool:
    rc, _, _ = await _run(['git', 'rev-parse', '--verify', '--quiet', full_branch], cwd=root)
    return rc == 0


@pytest.mark.asyncio
class TestCreateWorktreeStacksOnACommit:
    async def test_a_cold_branch_starts_at_the_commit_and_the_ledger_answers_it(
        self, cold: Predecessor,
    ) -> None:
        info = await cold.git_ops.create_worktree('D', stack_on=cold.target)

        assert info.path == cold.git_ops.worktree_base / 'D'
        assert await rev_parse(cold.root, 'task/D') == cold.tip
        assert info.base_commit == cold.tip
        assert await cold.git_ops.stack_base_read('task/D') == StackBaseRead(
            StackRecordState.RECORDED, cold.tip,
        )
        assert await ledger_of(cold.git_ops).base_of('task/D') == cold.tip

    async def test_a_warm_pool_hands_out_a_lane_started_at_the_commit(
        self, warm: Predecessor,
    ) -> None:
        info = await warm.git_ops.create_worktree('D', stack_on=warm.target)

        assert info.path == pool_lane(warm.git_ops)
        assert await rev_parse(warm.root, 'task/D') == warm.tip
        assert info.base_commit == warm.tip
        assert await ledger_of(warm.git_ops).base_of('task/D') == warm.tip

    async def test_a_fresh_train_successor_records_its_predecessor_tip(
        self, cold: Predecessor,
    ) -> None:
        info = await cold.git_ops.create_worktree('M', train=_TRAIN_SUCCESSOR)

        assert info.path == cold.git_ops.worktree_base / 'M'
        assert await rev_parse(cold.root, 'task/M') == cold.tip
        assert await ledger_of(cold.git_ops).base_of('task/M') == cold.tip

    async def test_a_train_successor_is_still_made_cold_beside_a_warm_pool(
        self, warm: Predecessor,
    ) -> None:
        info = await warm.git_ops.create_worktree('M', train=_TRAIN_SUCCESSOR)

        assert info.path == warm.git_ops.worktree_base / 'M'
        assert info.path != pool_lane(warm.git_ops)
        assert await ledger_of(warm.git_ops).base_of('task/M') == warm.tip

    async def test_a_stack_target_for_a_train_successor_is_refused(
        self, cold: Predecessor,
    ) -> None:
        with pytest.raises(ValueError):
            await cold.git_ops.create_worktree(
                'M', train=_TRAIN_SUCCESSOR, stack_on=cold.target,
            )

        assert not await _branch_exists(cold.root, 'task/M')

    async def test_a_record_that_does_not_read_back_raises(
        self, cold: Predecessor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fail_git_calls_starting(
            monkeypatch, tmp_path / 'shim', 'update-ref refs/dark-factory/stack-base/',
        )

        with pytest.raises(StackRecordUnconfirmed) as raised:
            await cold.git_ops.create_worktree('D', stack_on=cold.target)

        assert 'task/D' in str(raised.value)
        assert cold.tip in str(raised.value)

    async def test_a_stack_commit_that_does_not_resolve_never_falls_back_to_main(
        self, cold: Predecessor,
    ) -> None:
        with pytest.raises(RuntimeError):
            await cold.git_ops.create_worktree(
                'D', stack_on=StackTarget('task/P', 'deadbeef' * 5),
            )

        assert not await _branch_exists(cold.root, 'task/D')
