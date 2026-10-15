"""A train's verify reaches the lane's injected ``VerifyPort`` and runs on a
leased verify host (task 6160).

Every scene drives a stacked train through the lane's own queue over a real
tmp git repo, with the verify scripted by ``FakeVerifier``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, call

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    fails,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    merge_through_lane,
    running_lane,
)
from _orch_helpers import wait_responsive
from _recording_event_store import _RecordingEventStore

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    TRAIN_VERIFY_FAILED_REASON_PREFIX,
    GroupMergeRequest,
    MergeLane,
    MergeRequest,
    QueuedBranch,
)
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import RunnerUnavailable

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(repo=repo, git_ops=GitOps(_GIT, repo), config=lane_scene_config(repo, _GIT))


@pytest.fixture
def remote_scene(tmp_path: Path) -> _Scene:
    """A scene whose lane prefers a free remote host and, as DF's own config
    does, adopts a remote green without a local cross-check."""
    repo = seed_repo(tmp_path / 'repo', _SEED)
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=False,
    )
    return _Scene(repo=repo, git_ops=GitOps(_GIT, repo), config=config)


async def _branch(scene: _Scene, name: str) -> Path:
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return worktree


@dataclasses.dataclass(frozen=True)
class _Train:
    """A stacked train's request and the member callbacks it was built with."""

    request: GroupMergeRequest
    mark_member_done: AsyncMock
    redrive_member: AsyncMock

    @property
    def tip(self) -> str:
        return self.request.tip_task_id


async def _stacked_train(
    scene: _Scene, members: int, *, train_id: str = 'train-6160',
) -> _Train:
    names = [f'tr-{index}' for index in range(members)]
    worktrees = [await _branch(scene, name) for name in names]
    stacked = await scene.git_ops.stack_train_branches(names)
    assert stacked.survivors == names, stacked
    tip = QueuedBranch.parse(names[-1], _GIT.branch_prefix)
    mark_member_done = AsyncMock()
    redrive_member = AsyncMock()
    request = GroupMergeRequest(
        task_id=names[-1],
        branch=tip,
        worktree=worktrees[-1],
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        train_id=train_id,
        member_task_ids=names,
        tip_branch=tip,
        tip_task_id=names[-1],
        status_check=AsyncMock(return_value=dict.fromkeys(names, 'merge-deferred')),
        mark_member_done=mark_member_done,
        redrive_member=redrive_member,
    )
    return _Train(request, mark_member_done, redrive_member)


async def _single(scene: _Scene, name: str) -> MergeRequest:
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=await _branch(scene, name),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
    )


class _FakeRemote:
    """A remote verify host, duck-typed to ``verify_runner.py::VerifyRunner``.

    Not a ``RemoteRunner``, so ``VerifyRunnerPool.dispatch`` calls it with
    ``(merge_sha, spec)`` alone and skips the INV-2 code sync. ``calls`` keeps
    every merge SHA it was asked to verify. A verify waits for *release* when
    one is given, setting *entered* first. An *unreachable* host raises
    ``RunnerUnavailable`` and never probes healthy or clean, so the reprobe
    sweep cannot re-admit it during a test. ``cancel_verify`` answers
    *cancel_rc*. ``live`` is set while a verify runs, as ``RemoteRunner``'s
    in-flight id is; ``cancels`` records whether each remote cancel found a
    verify still live to abort, and ``clean_probes`` counts clean-probes.
    """

    is_local: ClassVar[bool] = False

    def __init__(
        self,
        name: str = 'laptop',
        *,
        release: asyncio.Event | None = None,
        entered: asyncio.Event | None = None,
        unreachable: bool = False,
        cancel_rc: int = 0,
    ) -> None:
        self.name = name
        self.release = release
        self.entered = entered
        self.unreachable = unreachable
        self.cancel_rc = cancel_rc
        self.calls: list[str] = []
        self.live = False
        self.cancels: list[bool] = []
        self.clean_probes = 0

    async def health(self) -> bool:
        return not self.unreachable

    async def probe_clean(self) -> bool:
        self.clean_probes += 1
        return not self.unreachable

    async def cancel_verify(self) -> int:
        self.cancels.append(self.live)
        return self.cancel_rc

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        self.calls.append(merge_sha)
        if self.unreachable:
            raise RunnerUnavailable(f'{self.name} is unreachable')
        self.live = True
        try:
            if self.entered is not None:
                self.entered.set()
            if self.release is not None:
                await self.release.wait()
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='',
                summary=f'green on {self.name}',
            )
        finally:
            self.live = False


def _slot_states(lane: MergeLane) -> dict[str, str]:
    """Each verify host's slot state, as the lane's public census reports it."""
    return {host['name']: host['slot_state'] for host in lane.snapshot()['hosts']}


def _in_flight(lane: MergeLane, task_id: str) -> list[dict[str, Any]]:
    """The restart drain's in-flight verify records for *task_id*."""
    records = lane.snapshot()['restart_drain']['verifies_in_flight']
    return [record for record in records if record['task_id'] == task_id]


# ─── The train verify obeys the lane's injected verifier ────────────────────


async def test_a_red_verify_on_the_lanes_verifier_derails_the_train(scene: _Scene) -> None:
    train = await _stacked_train(scene, 2)
    verifier = FakeVerifier(default=fails(category='pytest', summary='red train verify'))
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier)
    main_before = await scene.git_ops.get_main_sha()

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), outcome
    assert outcome.failure_category == 'pytest'
    assert await scene.git_ops.get_main_sha() == main_before
    assert verifier.verified == [train.tip]
    train.mark_member_done.assert_not_awaited()


async def test_the_lanes_disk_guard_skips_the_train_verify(scene: _Scene) -> None:
    train = await _stacked_train(scene, 2)
    verifier = FakeVerifier(disk_reason='disk guard: test reason')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier)
    main_before = await scene.git_ops.get_main_sha()

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason == 'disk guard: test reason'
    assert outcome.verify_skipped is True
    assert verifier.verified == []
    assert await scene.git_ops.get_main_sha() == main_before


# ─── Remote verify hosts, given to the lane directly ────────────────────────


async def test_a_lane_given_remote_runners_verifies_a_single_on_a_free_remote_under_prefer_remote(
    remote_scene: _Scene,
) -> None:
    single = await _single(remote_scene, 'solo')
    remote = _FakeRemote()
    verifier = FakeVerifier()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), remote_runners=[remote],
    )

    outcome = await merge_through_lane(lane, queue, single)

    assert outcome.status == 'done', outcome
    assert len(remote.calls) == 1
    assert verifier.verified == []


# ─── A train verifies on a host it leases from the lane ─────────────────────


async def test_a_train_verify_holds_the_local_host_slot_until_it_lands(scene: _Scene) -> None:
    train = await _stacked_train(scene, 2)
    entered = asyncio.Event()
    release = asyncio.Event()
    verifier = FakeVerifier(scripts={train.tip: hangs_until(release, entered=entered)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier, clock=healthy_verify_clock())

    async with running_lane(lane) as run:
        await queue.put(train.request)
        await wait_responsive(entered.wait(), label='train verify under way')
        while_verifying = _slot_states(lane).get('local')
        release.set()
        outcome = await run.outcome(train.request)
        after_landing = _slot_states(lane).get('local')

    assert while_verifying == 'busy'
    assert outcome.status == 'done', outcome
    assert after_landing == 'free'


async def test_a_two_member_train_verifies_on_a_free_remote_under_prefer_remote(
    remote_scene: _Scene,
) -> None:
    train = await _stacked_train(remote_scene, 2)
    entered = asyncio.Event()
    release = asyncio.Event()
    remote = _FakeRemote(release=release, entered=entered)
    verifier = FakeVerifier()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), remote_runners=[remote],
    )

    async with running_lane(lane) as run:
        await queue.put(train.request)
        await wait_responsive(entered.wait(), label='train verify under way on the remote')
        while_verifying = _slot_states(lane).get('laptop')
        records = _in_flight(lane, train.tip)
        release.set()
        outcome = await run.outcome(train.request)
        after_landing = _slot_states(lane).get('laptop')

    assert while_verifying == 'busy'
    assert [(r['host'], r['kind']) for r in records] == [('laptop', 'train')]
    assert outcome.status == 'done', outcome
    assert after_landing == 'free'
    assert len(remote.calls) == 1
    assert verifier.verified == []


@dataclasses.dataclass(frozen=True)
class _Unwound:
    """A remote's host census row, and whether its verify was still live,
    read the moment a cancelled lane's ``run()`` finished."""

    host: dict[str, Any]
    verify_live: bool


async def _abort_a_train_verifying_on(scene: _Scene, remote: _FakeRemote) -> _Unwound:
    """Cancel the lane's ``run()`` while a two-member train verifies on
    *remote*, as a lane crash or a hard shutdown does."""
    assert remote.entered is not None and remote.release is not None
    train = await _stacked_train(scene, 2)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=FakeVerifier(), clock=healthy_verify_clock(), remote_runners=[remote],
    )

    with pytest.raises(asyncio.CancelledError):
        async with running_lane(lane) as run:
            await queue.put(train.request)
            await wait_responsive(remote.entered.wait(), label='train verify under way on the remote')
            run.task.cancel()
            await wait_responsive(asyncio.wait({run.task}), label='lane run() unwound')
            (row,) = [host for host in lane.snapshot()['hosts'] if host['name'] == remote.name]
            unwound = _Unwound(host=row, verify_live=remote.live)

    assert not remote.release.is_set()
    return unwound


async def test_cancelling_a_train_mid_verify_aborts_its_remote_verify_while_it_is_live(
    remote_scene: _Scene,
) -> None:
    """A train's verify runs inline in the merger, so cancelling the lane must
    abort the remote verify through the worker's teardown chokepoint before
    the local verify unwinds, and then free the slot."""
    remote = _FakeRemote(release=asyncio.Event(), entered=asyncio.Event())

    unwound = await _abort_a_train_verifying_on(remote_scene, remote)

    assert remote.cancels[:1] == [True], (
        'the abort must precede the local cancel, or the remote verify-merge '
        f'process is orphaned (task 1757); cancels found live: {remote.cancels}'
    )
    assert unwound.verify_live is False
    assert (unwound.host['slot_state'], unwound.host['quarantined']) == ('free', False)


async def test_a_refused_cancel_parks_the_aborted_trains_remote_until_it_probes_clean(
    remote_scene: _Scene,
) -> None:
    """A remote that refuses the cancel may still be running the verify, so
    its slot is parked and freed only once the host probes clean."""
    remote = _FakeRemote(release=asyncio.Event(), entered=asyncio.Event(), cancel_rc=1)

    unwound = await _abort_a_train_verifying_on(remote_scene, remote)

    assert remote.clean_probes == 1
    assert (unwound.host['slot_state'], unwound.host['quarantined']) == ('free', False)


async def test_a_train_of_more_than_two_members_verifies_on_local_until_task_6106(
    remote_scene: _Scene,
) -> None:
    """Pins the member cap: until task 6106 lands, a remote MemoryMax kill
    reads as a red verdict, so a train of three stays off a free remote."""
    train = await _stacked_train(remote_scene, 3)
    remote = _FakeRemote()
    verifier = FakeVerifier()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), remote_runners=[remote],
    )

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'done', outcome
    assert remote.calls == []
    assert verifier.verified == [train.tip]


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)


async def test_a_train_whose_hosts_are_all_taken_verifies_unleased_instead_of_waiting(
    scene: _Scene,
) -> None:
    """A predecessor's held verify keeps the one host. On the fake clock the
    train's park behind that predecessor runs out at once, and the train
    must then verify unleased on local rather than wait for the slot.

    The train lands, flipping its members, while the predecessor is still
    held; its outcome resolves only after the predecessor's, because the
    lane resolves outcomes in merge order.
    """
    pred = await _single(scene, 'pred')
    train = await _stacked_train(scene, 2)
    release_pred = asyncio.Event()
    verifier = FakeVerifier(scripts={'pred': hangs_until(release_pred)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier, clock=healthy_verify_clock())

    async with running_lane(lane) as run:
        await queue.put(pred)
        await wait_responsive(verifier.await_entry(1), label='predecessor verify under way')
        await queue.put(train.request)
        await wait_responsive(
            _until(lambda: train.mark_member_done.await_count == 2),
            label='train landed while its predecessor holds the host',
        )
        pred_held = not pred.result.done()
        release_pred.set()
        pred_outcome = await run.outcome(pred)
        train_outcome = await run.outcome(train.request)

    assert pred_held
    assert verifier.verified == ['pred', train.tip]
    assert pred_outcome.status == 'done', pred_outcome
    assert train_outcome.status == 'done', train_outcome


async def test_a_remote_green_the_local_cross_check_contradicts_benches_the_remote(
    tmp_path: Path,
) -> None:
    """With the cross-check on, a train's remote green is re-run on local; a
    red local verdict withholds the land, quarantines the remote for
    divergence and halts the lane, exactly as for a single."""
    repo = seed_repo(tmp_path / 'repo', _SEED)
    scene = _Scene(
        repo=repo, git_ops=GitOps(_GIT, repo),
        config=lane_scene_config(
            repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=True,
        ),
    )
    train = await _stacked_train(scene, 2)
    remote = _FakeRemote()
    verifier = FakeVerifier(default=fails(category='pytest', summary='red on local'))
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), remote_runners=[remote],
    )
    main_before = await scene.git_ops.get_main_sha()

    async with running_lane(lane) as run:
        await queue.put(train.request)
        outcome = await run.outcome(train.request)
        halted = lane.is_wip_halted

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), outcome
    assert await scene.git_ops.get_main_sha() == main_before
    assert len(remote.calls) == 1
    assert verifier.verified == [train.tip]
    (laptop,) = [host for host in lane.snapshot()['hosts'] if host['name'] == 'laptop']
    assert (laptop['quarantined'], laptop['quarantine_class']) == (True, 'divergence')
    assert halted is True


async def test_an_unreachable_remote_moves_the_train_verify_to_local(
    remote_scene: _Scene,
) -> None:
    """A transport fault on the remote is never a derail: the remote is benched
    for reprobe and the same verify re-runs on local."""
    train = await _stacked_train(remote_scene, 2, train_id='coalesce-6160')
    remote = _FakeRemote(unreachable=True)
    verifier = FakeVerifier()
    store = _RecordingEventStore()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        remote_scene.git_ops, queue,
        verifier=verifier, clock=healthy_verify_clock(), remote_runners=[remote],
        event_store=store,
    )

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'done', outcome
    assert len(remote.calls) == 1
    assert verifier.verified == [train.tip]
    assert train.mark_member_done.await_args_list == [
        call(member, outcome.merge_sha) for member in train.request.member_task_ids
    ]
    train.redrive_member.assert_not_awaited()
    assert [name for name, _ in store.events if name == 'train_derailed'] == []
    (laptop,) = [host for host in lane.snapshot()['hosts'] if host['name'] == 'laptop']
    assert (laptop['quarantined'], laptop['quarantine_class'], laptop['slot_state']) == (
        True, 'ru', 'free',
    )
