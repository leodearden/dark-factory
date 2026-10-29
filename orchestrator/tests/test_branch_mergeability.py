"""Real-git tests for ``orchestrator.branch_mergeability`` (task 7516).

The pre-enqueue rebind (``workflow.py::TaskWorkflow._submit_to_merge_queue``,
task 1923) makes the queue land the worktree HEAD, so the verdict is read from
HEAD first. These cases build the two shapes that produced sha-less merge
successes (dark_factory 3493 and 3580, reify 8363) in a real repository and
check the verdict the reader gives them, plus the stale-ref and absent-ref
shapes that must stay mergeable.

The repository comes from ``_git_fixtures.py::seed_repo``; every git call this
file makes runs through ``_Checkout``, which applies both esc-3072-3 isolation
guards (``assert_isolated_git_repo`` before any subprocess, and
``git_env_with_ceiling``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.branch_mergeability import BranchState, Mergeability, read_branch_state
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import QueuedBranch

BRANCH = QueuedBranch.parse('42', 'task/')


class _Checkout:
    """git commands in one guarded checkout: a repo root or a linked worktree."""

    def __init__(self, root: Path) -> None:
        assert_isolated_git_repo(root)
        self.root = root
        self._env = git_env_with_ceiling(root)

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self._env, text=True, check=False,
        )
        assert proc.returncode == 0, (
            f'git {" ".join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}'
        )
        return proc.stdout.strip()

    def write(self, rel: str, content: str) -> None:
        (self.root / rel).write_text(content)

    def commit(self, message: str, files: dict[str, str]) -> str:
        for rel, content in files.items():
            self.write(rel, content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)


class _Scene:
    """A seeded repo with ``task/42`` checked out in a linked worktree, one commit ahead."""

    def __init__(self, tmp_path: Path) -> None:
        self.repo = _Checkout(seed_repo(tmp_path / 'repo'))
        worktree = tmp_path / 'wt'
        self.repo.git('worktree', 'add', str(worktree), '-b', BRANCH.full_name, 'main')
        self.wt = _Checkout(worktree)
        self.main_sha = self.repo.sha('main')
        self.task_commit = self.wt.commit('feat: task 42 work', {'feature.py': 'x = 1\n'})
        self.git_ops = GitOps(
            GitConfig(
                main_branch='main', branch_prefix='task/', remote='origin',
                worktree_dir='.worktrees', push_after_advance=False,
            ),
            self.repo.root,
        )

    async def read(self) -> BranchState:
        return await read_branch_state(
            self.git_ops, self.wt.root, BRANCH,
            worktree_head=self.wt.sha('HEAD'), main_sha=self.main_sha,
        )


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    return _Scene(tmp_path)


@pytest.mark.asyncio
async def test_a_clean_branch_with_work_beyond_main_is_mergeable(scene: _Scene) -> None:
    state = await scene.read()

    assert state.mergeability is Mergeability.MERGEABLE
    assert state.branch_tip == scene.task_commit
    assert state.worktree_dirty is False


@pytest.mark.asyncio
async def test_a_branch_reset_to_main_with_the_work_staged_is_empty(scene: _Scene) -> None:
    scene.wt.git('reset', '--soft', 'main')

    state = await scene.read()

    assert state.mergeability is Mergeability.BRANCH_EMPTY
    assert state.worktree_dirty is True
    assert state.branch_tip == scene.main_sha
    assert state.worktree_head == scene.main_sha


@pytest.mark.asyncio
async def test_a_worktree_moved_onto_main_while_the_branch_is_ahead_is_off_branch(
    scene: _Scene,
) -> None:
    scene.wt.git('checkout', '-b', f'{BRANCH.full_name}-squash', 'main')
    scene.wt.write('feature.py', 'x = 1\n')
    scene.wt.git('add', 'feature.py')

    state = await scene.read()

    assert state.mergeability is Mergeability.WORKTREE_OFF_BRANCH
    assert state.branch_tip == scene.task_commit
    assert state.worktree_head == scene.main_sha


@pytest.mark.asyncio
async def test_a_detached_head_carrying_newer_work_than_a_stale_ref_is_mergeable(
    scene: _Scene,
) -> None:
    scene.wt.git('checkout', '--detach')
    newer = scene.wt.commit('feat: task 42 follow-up', {'more.py': 'y = 2\n'})

    state = await scene.read()

    assert state.mergeability is Mergeability.MERGEABLE
    assert state.worktree_head == newer
    assert state.branch_tip == scene.task_commit


@pytest.mark.asyncio
async def test_a_head_ahead_of_a_ref_left_on_main_is_mergeable(scene: _Scene) -> None:
    scene.wt.git('checkout', '--detach')
    newer = scene.wt.commit('feat: task 42 follow-up', {'more.py': 'y = 2\n'})
    scene.repo.git('branch', '-f', BRANCH.full_name, 'main')

    state = await scene.read()

    assert state.mergeability is Mergeability.MERGEABLE
    assert state.worktree_head == newer
    assert state.branch_tip == scene.main_sha


@pytest.mark.asyncio
async def test_an_absent_ref_falls_back_to_the_worktree_head(scene: _Scene) -> None:
    scene.wt.git('checkout', '--detach')
    scene.repo.git('branch', '-D', BRANCH.full_name)

    ahead = await scene.read()
    assert ahead.branch_tip is None
    assert ahead.mergeability is Mergeability.MERGEABLE
    assert ahead.worktree_head == scene.task_commit

    scene.wt.git('reset', '--hard', 'main')
    on_main = await scene.read()
    assert on_main.mergeability is Mergeability.BRANCH_EMPTY


def test_an_unresolved_worktree_head_is_unreadable() -> None:
    state = BranchState(
        branch=BRANCH, branch_tip=None, worktree_head='', main_sha='a' * 40,
        head_beyond_main=False, tip_beyond_main=False, worktree_dirty=False,
    )

    assert state.mergeability is Mergeability.UNREADABLE


_MAIN = 'm' * 40
_HEAD = 'h' * 40
_TIP = 't' * 40


@pytest.mark.parametrize(
    ('head_beyond_main', 'branch_tip', 'tip_beyond_main', 'expected'),
    [
        (True, _MAIN, False, Mergeability.MERGEABLE),
        (True, _TIP, True, Mergeability.MERGEABLE),
        (True, None, False, Mergeability.MERGEABLE),
        (False, _TIP, True, Mergeability.WORKTREE_OFF_BRANCH),
        (False, _MAIN, False, Mergeability.BRANCH_EMPTY),
        (False, None, False, Mergeability.BRANCH_EMPTY),
    ],
    ids=[
        'head-beyond-tip-on-main',
        'head-beyond-tip-beyond-elsewhere',
        'head-beyond-ref-absent',
        'head-on-main-tip-beyond',
        'head-on-main-tip-on-main',
        'head-on-main-ref-absent',
    ],
)
def test_the_verdict_follows_the_worktree_head_first(
    head_beyond_main: bool,
    branch_tip: str | None,
    tip_beyond_main: bool,
    expected: Mergeability,
) -> None:
    worktree_head = _HEAD if head_beyond_main else _MAIN
    state = BranchState(
        branch=BRANCH, branch_tip=branch_tip, worktree_head=worktree_head,
        main_sha=_MAIN, head_beyond_main=head_beyond_main,
        tip_beyond_main=tip_beyond_main, worktree_dirty=False,
    )

    assert state.mergeability is expected


def test_a_refusal_names_the_branch_its_shas_and_the_recovery_hint() -> None:
    tip, main_sha = '1c3a2b3ae3' + 'a' * 30, '8bee6520df' + 'b' * 30
    state = BranchState(
        branch=BRANCH, branch_tip=tip, worktree_head=main_sha, main_sha=main_sha,
        head_beyond_main=False, tip_beyond_main=True, worktree_dirty=True,
    )

    summary = state.refusal_summary()
    detail = state.refusal_detail(
        task_id='42', worktree=Path('/fake/wt'), landing_verdict=None,
    )

    assert 'task/42' in summary
    assert tip[:7] in summary
    assert main_sha[:7] in summary
    assert 'git reflog show task/42' in detail
