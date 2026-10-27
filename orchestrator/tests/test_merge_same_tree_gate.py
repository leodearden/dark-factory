"""A rebased tree byte-identical to the verified tree clears the gate with no re-verify.

Task 6647.  Main moves under a merge verify by landing a commit whose patch
the request already carries (a follower stacked on its head's pre-rebase
commit).  The rebase drops that commit as already upstream, so the rebased
tree IS the tree the verify passed, and even a whole-tree gate has nothing
new to see.  Every scenario drives the production lane over a real git repo
with a whole-tree merge gate, so the disjoint fast path can never be what
clears it, and with ``merge_gate_history_pure`` declared unless the scenario
is about the project that does not declare it.

The single-request gate re-verifies through the production verifier, not the
injected one, so those scenarios read the gate's decision from its
``gate_retry`` row; the train's re-verify goes through the injected verifier,
so the train scenario also counts verify calls.
"""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, merge_through_lane
from _merge_lane_verifier_doubles import ScriptedVerifier
from _orch_helpers import git_env_with_ceiling
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import GroupMergeRequest, MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')

Mover = Callable[[Path], str]


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _commit_file(worktree: Path, name: str) -> str:
    (worktree / name).write_text(f'{name}\n')
    _git(worktree, 'add', '--', name)
    _git(worktree, 'commit', '-m', f'add {name}')
    return _git(worktree, 'rev-parse', 'HEAD')


def _tree(repo: Path, rev: str) -> str:
    return _git(repo, 'rev-parse', f'{rev}^{{tree}}')


def _lands_patch_of(commit: str) -> Mover:
    """Move main by landing *commit*'s patch under a new sha, as a sibling's landing would."""
    def mover(repo: Path) -> str:
        _git(repo, 'cherry-pick', '-x', commit)
        landed = _git(repo, 'rev-parse', 'HEAD')
        assert landed != commit, 'the patch must land under a new sha'
        return landed
    return mover


def _unrelated_drift(repo: Path) -> str:
    return _commit_file(repo, f'unrelated-{uuid.uuid4().hex[:8]}.txt')


@dataclasses.dataclass(frozen=True)
class _VerifyCall:
    head: str
    tree: str


class _MovingVerify:
    """A merge verify that records what it saw and moves main on its first call."""

    def __init__(self, repo: Path, mover: Mover) -> None:
        self._repo = repo
        self._mover = mover
        self.calls: list[_VerifyCall] = []

    async def __call__(self, worktree: Path, *args: Any, **kwargs: Any) -> VerifyResult:
        head = _git(worktree, 'rev-parse', 'HEAD')
        self.calls.append(_VerifyCall(head=head, tree=_tree(worktree, head)))
        if len(self.calls) == 1:
            self._mover(self._repo)
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig


def _scene(
    tmp_path: Path, *, history_pure: bool = True, history_leg: str | None = None,
) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    git = GitConfig(
        main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return _Scene(
        repo=repo, git_ops=GitOps(git, repo),
        config=lane_scene_config(
            repo, git, merge_verify_breadth='full', merge_gate_history_pure=history_pure,
            merge_gate_history_test_command=history_leg,
        ),
    )


def _leg_writing_head_to(marker: Path) -> str:
    return f'git rev-parse HEAD > {marker}'


async def _stacked_follower(scene: _Scene) -> tuple[Path, str]:
    """Branch ``task/follower``: the head's commit, then its own. Returns (worktree, head commit)."""
    head_wt = (await scene.git_ops.create_worktree('head')).path
    head_commit = _commit_file(head_wt, 'head.txt')
    follower_wt = (await scene.git_ops.create_worktree('follower')).path
    _git(follower_wt, 'merge', '--ff-only', head_commit)
    _commit_file(follower_wt, 'follower.txt')
    return follower_wt, head_commit


def _request(scene: _Scene, task_id: str, worktree: Path) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, scene.config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
    )


async def _run(
    scene: _Scene, request: Any, verify: _MovingVerify, store: _RecordingEventStore,
) -> MergeOutcome:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=ScriptedVerifier(verify), event_store=store,
    )
    return await merge_through_lane(lane, queue, request)


def _gate_retries(store: _RecordingEventStore) -> list[dict[str, Any]]:
    return [
        record['data'] for name, record in store.events
        if name == 'merge_attempt' and record['data']['outcome'] == 'gate_retry'
    ]


async def test_rebase_that_drops_an_upstream_patch_lands_without_a_reverify(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    follower_wt, head_commit = await _stacked_follower(scene)
    verify = _MovingVerify(scene.repo, _lands_patch_of(head_commit))
    store = _RecordingEventStore()

    outcome = await _run(scene, _request(scene, 'follower', follower_wt), verify, store)

    assert outcome.status == 'done', outcome
    assert _tree(scene.repo, 'main') == verify.calls[0].tree
    assert [row['gate_clearance'] for row in _gate_retries(store)] == ['same_tree']


async def test_same_tree_reverifies_when_the_gate_reads_history(tmp_path: Path) -> None:
    scene = _scene(tmp_path, history_pure=False)
    follower_wt, head_commit = await _stacked_follower(scene)
    verify = _MovingVerify(scene.repo, _lands_patch_of(head_commit))
    store = _RecordingEventStore()

    outcome = await _run(scene, _request(scene, 'follower', follower_wt), verify, store)

    assert outcome.status == 'done', outcome
    assert _tree(scene.repo, 'main') == verify.calls[0].tree
    assert [row['gate_clearance'] for row in _gate_retries(store)] == ['reverified']


async def test_rebase_onto_an_unrelated_drift_still_reverifies(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    follower_wt, _ = await _stacked_follower(scene)
    verify = _MovingVerify(scene.repo, _unrelated_drift)
    store = _RecordingEventStore()

    outcome = await _run(scene, _request(scene, 'follower', follower_wt), verify, store)

    assert outcome.status == 'done', outcome
    assert _tree(scene.repo, 'main') != verify.calls[0].tree
    assert [row['gate_clearance'] for row in _gate_retries(store)] == ['reverified']


async def test_train_whose_member_patch_lands_upstream_skips_the_reverify(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    outcome, verify, store = await _run_train_whose_member_patch_lands_upstream(scene)

    assert outcome.status == 'done', outcome
    assert len(verify.calls) == 1
    assert _tree(scene.repo, 'main') == verify.calls[0].tree
    [row] = _gate_retries(store)
    assert row['gate_clearance'] == 'same_tree'
    assert row['train_id'] == 'train-6647'


async def test_rebase_that_drops_an_upstream_patch_runs_only_the_history_leg(
    tmp_path: Path,
) -> None:
    marker = tmp_path / 'leg-head'
    scene = _scene(tmp_path, history_leg=_leg_writing_head_to(marker))
    follower_wt, head_commit = await _stacked_follower(scene)
    verify = _MovingVerify(scene.repo, _lands_patch_of(head_commit))
    store = _RecordingEventStore()

    outcome = await _run(scene, _request(scene, 'follower', follower_wt), verify, store)

    assert outcome.status == 'done', outcome
    assert len(verify.calls) == 1
    assert [row['gate_clearance'] for row in _gate_retries(store)] == ['history_leg']
    assert marker.read_text().strip() == _git(scene.repo, 'rev-parse', 'main')


async def test_train_whose_member_patch_lands_upstream_runs_only_the_history_leg(
    tmp_path: Path,
) -> None:
    marker = tmp_path / 'leg-head'
    scene = _scene(tmp_path, history_leg=_leg_writing_head_to(marker))
    outcome, verify, store = await _run_train_whose_member_patch_lands_upstream(scene)

    assert outcome.status == 'done', outcome
    assert len(verify.calls) == 1
    [row] = _gate_retries(store)
    assert row['gate_clearance'] == 'history_leg'
    assert row['train_id'] == 'train-6647'
    assert marker.read_text().strip() == _git(scene.repo, 'rev-parse', 'main')


async def test_rebase_onto_an_unrelated_drift_does_not_run_the_history_leg(
    tmp_path: Path,
) -> None:
    marker = tmp_path / 'leg-head'
    scene = _scene(tmp_path, history_leg=_leg_writing_head_to(marker))
    follower_wt, _ = await _stacked_follower(scene)
    verify = _MovingVerify(scene.repo, _unrelated_drift)
    store = _RecordingEventStore()

    outcome = await _run(scene, _request(scene, 'follower', follower_wt), verify, store)

    assert outcome.status == 'done', outcome
    assert [row['gate_clearance'] for row in _gate_retries(store)] == ['reverified']
    assert not marker.exists()


async def _run_train_whose_member_patch_lands_upstream(
    scene: _Scene,
) -> tuple[MergeOutcome, _MovingVerify, _RecordingEventStore]:
    """A three-member train whose first member's patch lands on main under its verify."""
    members = ('tm-a', 'tm-b', 'tm-c')
    worktrees: dict[str, Path] = {}
    for member in members:
        worktrees[member] = (await scene.git_ops.create_worktree(member)).path
        _commit_file(worktrees[member], f'{member}.txt')
    stacked = await scene.git_ops.stack_train_branches(list(members))
    assert stacked.survivors == list(members), stacked
    first_member_commit = _git(scene.repo, 'rev-parse', f'task/{members[0]}')
    tip = QueuedBranch.parse(members[-1], scene.config.git.branch_prefix)
    request = GroupMergeRequest(
        task_id=members[-1],
        branch=tip,
        worktree=worktrees[members[-1]],
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        train_id='train-6647',
        member_task_ids=list(members),
        tip_branch=tip,
        tip_task_id=members[-1],
        status_check=AsyncMock(return_value=dict.fromkeys(members, 'merge-deferred')),
        mark_member_done=AsyncMock(),
        redrive_member=AsyncMock(),
    )
    verify = _MovingVerify(scene.repo, _lands_patch_of(first_member_commit))
    store = _RecordingEventStore()
    return await _run(scene, request, verify, store), verify, store
