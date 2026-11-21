"""GitOps.create_worktree stacks a task branch on a predecessor's commit (task 10473).

``create_worktree(stack_on=StackTarget(...))`` starts the branch at that
commit, from the warm pool where there is one, and returns only once the
stack-base ledger answers the commit.  A declared-train successor is one user
of the same input: it stacks on its predecessor's tip, records it (task 7660)
and is still made cold.

On an existing branch, every reuse route (the registered cold worktree, the
cold re-attach, a warm-pool lane) re-stacks the branch's own delta onto the
target instead of rebasing it onto main (PRD rows 42, 43 and 47).

Real temp git repositories throughout (``_stacked_repo``); failures are forced
only through a PATH git shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _stacked_repo import (
    D_FILE,
    RestackScene,
    add_worktree,
    build_restack_scene,
    commit_files,
    fail_git_calls_starting,
    git,
    ledger_of,
    make_repo,
    make_warm_repo,
    pool_lane,
    release_into_lane,
    rev_parse,
)

from orchestrator.branch_stack import (
    RestackConflict,
    RestackTrialOutcome,
    StackBaseRead,
    StackRecordState,
    StackTarget,
)
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


# ---------------------------------------------------------------------------
# The re-stack through each reuse route (PRD rows 42, 43 and 47)
# ---------------------------------------------------------------------------

_UNCOMMITTED = 'd edited, not yet committed\n'


async def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    rc, _, _ = await _run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=root)
    return rc == 0


def _new_tip(scene: RestackScene) -> StackTarget:
    return StackTarget('task/P', scene.p_new)


async def _registered(tmp_path: Path, *, conflicting: bool = False) -> RestackScene:
    scene = await build_restack_scene(await make_repo(tmp_path / 'repo'), conflicting=conflicting)
    if not conflicting:
        (scene.d_worktree / D_FILE).write_text(_UNCOMMITTED)
    return scene


async def _removed(tmp_path: Path) -> RestackScene:
    scene = await build_restack_scene(await make_repo(tmp_path / 'repo'))
    await git(scene.root, 'worktree', 'remove', '--force', str(scene.d_worktree))
    return scene


async def _in_a_lane(tmp_path: Path, *, conflicting: bool = False) -> RestackScene:
    scene = await build_restack_scene(
        await make_warm_repo(tmp_path / 'repo'), conflicting=conflicting,
    )
    if not conflicting:
        (scene.d_worktree / D_FILE).write_text(_UNCOMMITTED)
    await release_into_lane(scene.git_ops, 'D')
    return scene


async def _assert_restacked_on_the_new_tip(scene: RestackScene, info_base: str) -> None:
    root = scene.root
    diff = await git(root, 'diff', '--name-only', scene.p_new, 'task/D')
    assert set(diff.split()) == {D_FILE}
    assert await _is_ancestor(root, scene.p_new, 'task/D')
    assert not await _is_ancestor(root, scene.main_tip, 'task/D')
    assert not await _is_ancestor(root, scene.p_old, 'task/D')
    assert await scene.git_ops.stack_base_read('task/D') == StackBaseRead(
        StackRecordState.RECORDED, scene.p_new,
    )
    assert info_base == scene.p_new


async def _assert_left_on_the_old_tip(scene: RestackScene, wt: Path) -> None:
    assert await rev_parse(scene.root, 'task/D') == scene.d_tip
    assert await ledger_of(scene.git_ops).base_of('task/D') == scene.p_old
    assert (await git(wt, 'status', '--porcelain')).strip() == ''
    rebase_dir = Path((await git(wt, 'rev-parse', '--git-path', 'rebase-merge')).strip())
    assert not (rebase_dir if rebase_dir.is_absolute() else wt / rebase_dir).exists()


@pytest.mark.asyncio
class TestRestackThroughEachReuseRoute:
    async def test_a_registered_cold_worktree_is_restacked(self, tmp_path: Path) -> None:
        scene = await _registered(tmp_path)

        info = await scene.git_ops.create_worktree('D', stack_on=_new_tip(scene))

        assert info.path == scene.d_worktree
        await _assert_restacked_on_the_new_tip(scene, info.base_commit)
        assert await git(scene.root, 'show', f'task/D:{D_FILE}') == _UNCOMMITTED

    async def test_a_removed_worktree_is_reattached_and_restacked(
        self, tmp_path: Path,
    ) -> None:
        scene = await _removed(tmp_path)

        info = await scene.git_ops.create_worktree('D', stack_on=_new_tip(scene))

        assert info.path == scene.d_worktree
        await _assert_restacked_on_the_new_tip(scene, info.base_commit)

    async def test_a_warm_lane_is_restacked(self, tmp_path: Path) -> None:
        scene = await _in_a_lane(tmp_path)

        info = await scene.git_ops.create_worktree('D', stack_on=_new_tip(scene))

        assert info.path == pool_lane(scene.git_ops)
        await _assert_restacked_on_the_new_tip(scene, info.base_commit)
        assert await git(scene.root, 'show', f'task/D:{D_FILE}') == _UNCOMMITTED

    async def test_a_conflicting_registered_worktree_raises_and_moves_nothing(
        self, tmp_path: Path,
    ) -> None:
        scene = await _registered(tmp_path, conflicting=True)

        with pytest.raises(RestackConflict):
            await scene.git_ops.create_worktree('D', stack_on=_new_tip(scene))

        await _assert_left_on_the_old_tip(scene, scene.d_worktree)

    async def test_a_conflicting_warm_lane_raises_and_keeps_the_lane(
        self, tmp_path: Path,
    ) -> None:
        scene = await _in_a_lane(tmp_path, conflicting=True)
        lane = pool_lane(scene.git_ops)

        with pytest.raises(RestackConflict):
            await scene.git_ops.create_worktree('D', stack_on=_new_tip(scene))

        await _assert_left_on_the_old_tip(scene, lane)
        assert (await git(lane, 'symbolic-ref', 'HEAD')).strip() == 'refs/heads/task/D'


@pytest.mark.asyncio
class TestRestackTrialThroughGitOps:
    async def test_a_clean_scene_is_clean(self, tmp_path: Path) -> None:
        scene = await build_restack_scene(await make_repo(tmp_path / 'repo'))

        trial = await scene.git_ops.restack_trial('task/D', _new_tip(scene))

        assert trial.outcome is RestackTrialOutcome.CLEAN

    async def test_a_conflicting_scene_conflicts_and_moves_nothing(
        self, tmp_path: Path,
    ) -> None:
        scene = await build_restack_scene(await make_repo(tmp_path / 'repo'), conflicting=True)

        trial = await scene.git_ops.restack_trial('task/D', _new_tip(scene))

        assert trial.outcome is RestackTrialOutcome.CONFLICT
        await _assert_left_on_the_old_tip(scene, scene.d_worktree)

    async def test_a_branch_with_no_record_is_unknown(self, tmp_path: Path) -> None:
        scene = await build_restack_scene(await make_repo(tmp_path / 'repo'))
        await scene.ledger.forget('task/D')

        trial = await scene.git_ops.restack_trial('task/D', _new_tip(scene))

        assert trial.outcome is RestackTrialOutcome.UNKNOWN
