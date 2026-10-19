"""A head whose verified merge commit a later landing carried onto main lands without a gate.

Task 7196 (the "already on main" slice of 6155).  While the head verifies,
main moves to a child of the head's own merge commit.  The CAS advance then
rebases the parked worktree onto main itself, and re-verifying that would
gate main's own tree.  The head instead finalizes ``done`` by its own merge
commit.  Every scenario drives the production lane over a real git repo with
a whole-tree merge gate.
"""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, merge_through_lane, running_lane
from _merge_lane_verifier_doubles import ScriptedVerifier
from _orch_helpers import git_env_with_ceiling, wait_responsive
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env
from test_merge_queue_concurrent_verify import _gated_runner, _inject_two_host_allocator

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane, MergeRequest, QueuedBranch
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')

Mover = Callable[[Path, str], str]


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


def _successor_lands(repo: Path, verified_head: str) -> str:
    """Move main to a child of *verified_head*, as a later landing that carried it would."""
    _git(repo, 'merge', '--ff-only', verified_head)
    return _commit_file(repo, 'successor.txt')


def _unrelated_drift(repo: Path, verified_head: str) -> str:
    return _commit_file(repo, 'unrelated.txt')


class _MovingVerify:
    """A merge verify that records the HEAD it was handed and moves main on its first call."""

    def __init__(self, repo: Path, mover: Mover) -> None:
        self._repo = repo
        self._mover = mover
        self.heads: list[str] = []
        self.drift: str | None = None

    async def __call__(self, worktree: Path, *args: Any, **kwargs: Any) -> VerifyResult:
        self.heads.append(_git(worktree, 'rev-parse', 'HEAD'))
        if len(self.heads) == 1:
            self.drift = self._mover(self._repo, self.heads[0])
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig


def _scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    git = GitConfig(
        main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return _Scene(
        repo=repo, git_ops=GitOps(git, repo),
        config=lane_scene_config(repo, git, merge_verify_breadth='full'),
    )


async def _request(scene: _Scene, task_id: str) -> MergeRequest:
    worktree = (await scene.git_ops.create_worktree(task_id)).path
    _commit_file(worktree, f'{task_id}.txt')
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


def _lane(
    scene: _Scene, verify: _MovingVerify, store: _RecordingEventStore,
) -> tuple[MergeLane, asyncio.Queue[Any]]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=ScriptedVerifier(verify), event_store=store,
    )
    return lane, queue


def _attempt_outcomes(store: _RecordingEventStore, task_id: str) -> list[str]:
    return [
        record['data']['outcome'] for name, record in store.events
        if name == 'merge_attempt' and record['task_id'] == task_id
    ]


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    env = git_env_with_ceiling(repo)
    for key in git_redirect_env(env):
        del env[key]
    return subprocess.run(
        ['git', 'merge-base', '--is-ancestor', ancestor, descendant],
        cwd=repo, env=env, check=False,
    ).returncode == 0


async def test_head_carried_onto_main_by_a_successor_finalizes_done_without_a_gate(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    request = await _request(scene, 'carried')
    verify = _MovingVerify(scene.repo, _successor_lands)
    store = _RecordingEventStore()
    lane, queue = _lane(scene, verify, store)

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    [merge_commit] = verify.heads
    assert outcome.merge_sha == merge_commit
    assert _git(scene.repo, 'rev-parse', 'main') == verify.drift
    assert _is_ancestor(scene.repo, merge_commit, 'main')
    assert 'gate_retry' not in _attempt_outcomes(store, 'carried')


async def test_head_whose_main_moved_by_an_unrelated_commit_still_goes_through_the_gate(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    request = await _request(scene, 'drifted')
    verify = _MovingVerify(scene.repo, _unrelated_drift)
    store = _RecordingEventStore()
    lane, queue = _lane(scene, verify, store)

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    [merge_commit] = verify.heads
    assert outcome.merge_sha != merge_commit
    assert outcome.merge_sha == _git(scene.repo, 'rev-parse', 'main')
    assert 'gate_retry' in _attempt_outcomes(store, 'drifted')


class _AncestryProbeFails(GitOps):
    """``GitOps`` whose ancestry probe answers "unknown" (a git error) for one ancestor."""

    failing_ancestor: str | None = None

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None:
        if ancestor == self.failing_ancestor:
            return None
        return await super().is_ancestor_checked(ancestor, descendant)


async def test_ancestry_probe_error_falls_through_to_the_gate(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    git_ops = _AncestryProbeFails(scene.config.git, scene.repo)
    scene = dataclasses.replace(scene, git_ops=git_ops)
    request = await _request(scene, 'probe-error')

    def successor_lands_and_probe_fails(repo: Path, verified_head: str) -> str:
        git_ops.failing_ancestor = verified_head
        return _successor_lands(repo, verified_head)

    verify = _MovingVerify(scene.repo, successor_lands_and_probe_fails)
    store = _RecordingEventStore()
    lane, queue = _lane(scene, verify, store)

    outcome = await merge_through_lane(lane, queue, request)

    assert outcome.status == 'done', outcome
    [merge_commit] = verify.heads
    assert outcome.merge_sha != merge_commit
    assert 'gate_retry' in _attempt_outcomes(store, 'probe-error')


async def test_in_flight_follower_of_a_carried_head_is_not_cascaded(tmp_path: Path) -> None:
    """The head finalizes as landed, so the follower verifying behind it is never torn down."""
    scene = _scene(tmp_path)
    head = await _request(scene, 'carried-head')
    follower = await _request(scene, 'follower')
    head_entered, head_release = asyncio.Event(), asyncio.Event()
    follower_entered, follower_release = asyncio.Event(), asyncio.Event()
    remote = _gated_runner(follower_release, follower_entered, name='remote-follower')
    head_verified: list[str] = []

    async def head_parks(worktree: Path, *args: Any, **kwargs: Any) -> VerifyResult:
        head_verified.append(_git(worktree, 'rev-parse', 'HEAD'))
        head_entered.set()
        await head_release.wait()
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )

    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=ScriptedVerifier(head_parks),
        event_store=_RecordingEventStore(),
    )
    async with running_lane(lane) as run:
        _inject_two_host_allocator(lane, remote)
        await queue.put(head)
        await queue.put(follower)
        await wait_responsive(head_entered.wait(), label="the head's verify to enter")
        await wait_responsive(follower_entered.wait(), label="the follower's remote verify to enter")
        [head_merge_commit] = head_verified
        successor = _successor_lands(scene.repo, head_merge_commit)
        head_release.set()
        follower_release.set()
        head_outcome = await run.outcome(head)
        follower_outcome = await run.outcome(follower)

    assert head_outcome.status == 'done', head_outcome
    assert head_outcome.merge_sha == head_merge_commit
    assert follower_outcome.status == 'done', follower_outcome
    remote.cancel_verify.assert_not_awaited()
    assert remote.run_merge_verify.await_count == 1
    assert _is_ancestor(scene.repo, successor, 'main')
    assert {'follower.txt', 'carried-head.txt', 'successor.txt'} <= set(
        _git(scene.repo, 'ls-tree', '-r', '--name-only', 'main').splitlines()
    )
