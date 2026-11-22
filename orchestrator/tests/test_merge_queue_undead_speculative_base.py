"""A successor never verifies on the merge commit of a predecessor the lane abandoned (task 8933).

When the lane gives a built item's request back to the queue, or drops it,
that item's speculative merge commit will never reach main. A successor
already built on it must not verify on it: the INV-3 dispatch re-check voids
the successor's base and re-merges it, and the §5.3 detector stays silent.
A successor that already verified on it is re-verified without it, never
adopted on that tree.

Live merger and verifier loops over a real tmp git repo. Head H verifies on
the remote 'head-host'; S, merged speculatively on H, verifies on 'local';
T, merged speculatively on S, waits for a host. Every observation is public:
request futures, ``snapshot()`` states, the event store, the recorded
escalations and the trees the fake hosts were handed.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    RecordingEscalations,
    VerifyScript,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    lane_state,
    make_lane,
    raises,
    running_lane,
)
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, MERGE_RESULT_TIMEOUT, wait_responsive

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeVerifyLeaseHeld
from orchestrator.merge_lane import MergeLane, MergeRequest, QueuedBranch
from orchestrator.merge_lane.verify_base_detector import VERIFY_BASE_DIVERGENCE_SENTINEL
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_HEAD_HOST = 'head-host'
_POLL_SECS = 0.005
_AWAITING_DISPATCH = ('awaiting_verify', 'awaiting_host')

Tree = frozenset[str]
_ON_S_LIVE_COMMIT: Tree = frozenset({'H.txt', 'S.txt', 'T.txt'})


async def _git(repo: Path, *args: str) -> tuple[int, str]:
    """``git *args`` in *repo*, off the event loop: its exit code and stdout."""
    proc = await asyncio.create_subprocess_exec(
        'git', *args, cwd=repo,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, stdout.decode()


def _task_files(listing: str) -> Tree:
    return frozenset(name for name in listing.split() if name.endswith('.txt'))


async def _tree_of(repo: Path, rev: str) -> Tree:
    """The task files (``*.txt``) in *rev*'s tree."""
    _, listing = await _git(repo, 'ls-tree', '--name-only', rev)
    return _task_files(listing)


async def _change_of(repo: Path, rev: str) -> Tree:
    """The task files (``*.txt``) *rev* changes over its first parent."""
    rc, listing = await _git(repo, 'diff', '--name-only', f'{rev}^1', rev)
    assert rc == 0, f'git diff {rev}^1 {rev} failed in {repo}'
    return _task_files(listing)


class _TreeRecordingVerifier(FakeVerifier):
    """The local host. Each call records its task and the ``*.txt`` names its
    worktree held at call time, then follows that task's next queued script
    (passing once none are left)."""

    def __init__(self, scripts: Mapping[str, Sequence[VerifyScript]]) -> None:
        super().__init__()
        self._queued = {task: collections.deque(s) for task, s in scripts.items()}
        self.trees: list[tuple[str | None, Tree]] = []

    def next_script(self, task_id: str | None) -> VerifyScript:
        queued = self._queued.get(task_id or '')
        return queued.popleft() if queued else super().next_script(task_id)

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ) -> VerifyResult:
        files = frozenset(p.name for p in Path(worktree).glob('*.txt'))
        self.trees.append((options.get('task_id'), files))
        return await super().run_scoped(worktree, config, module_configs, task_files, **options)


@dataclasses.dataclass(frozen=True)
class _RemoteVerify:
    """A commit a remote host was asked to verify: the ``*.txt`` names it
    changes over its first parent, which are the work of the one task branch
    it brings in, whether it is a ``--no-ff`` merge or a rebased replay, and
    the ``*.txt`` names of its tree."""

    change: Tree
    tree: Tree


class _TreeRecordingRemote(FakeRemoteRunner):
    """A remote host that records each commit it is asked to verify as a
    ``_RemoteVerify``, alongside the ``calls`` ledger. A verify is recorded on
    entry, so one handed and then cancelled still counts, and with nothing
    awaited between recording it and starting it: once ``release`` is set, a
    recorded verify with the host not ``live`` has returned."""

    def __init__(
        self, name: str, repo: Path, *, release: asyncio.Event, entered: asyncio.Event,
    ) -> None:
        super().__init__(name, release=release, entered=entered)
        self.repo = repo
        self.verifies: list[_RemoteVerify] = []

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        change = await _change_of(self.repo, merge_sha)
        tree = await _tree_of(self.repo, merge_sha)
        self.verifies.append(_RemoteVerify(change=change, tree=tree))
        return await super().run_merge_verify(merge_sha, spec)


@dataclasses.dataclass
class _Scene:
    git_ops: GitOps
    config: OrchestratorConfig
    events: EventStore
    escalations: RecordingEscalations

    @property
    def repo(self) -> Path:
        return self.git_ops.project_root


def _scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo')
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=False,
        merge_deep=MergeDeepConfig(chain_cap=0),
    )
    assert config.merge_deep.chain_cap == 0, (
        f'the operational yaml switched deep chains on in the scene: {config.merge_deep!r}'
    )
    return _Scene(
        git_ops=GitOps(_GIT, repo),
        config=config,
        events=EventStore(db_path=tmp_path / 'events.db', run_id='undead-speculative-base'),
        escalations=RecordingEscalations(),
    )


async def _request(scene: _Scene, name: str) -> MergeRequest:
    """The request merging task *name*'s branch, one commit adding ``<name>.txt``."""
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


async def _until(
    predicate: Callable[[], bool], *, label: str, timeout: float = MERGE_RESULT_TIMEOUT,
) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(_POLL_SECS)
    await wait_responsive(_poll(), timeout=timeout, label=label)


@dataclasses.dataclass
class _Run:
    """A running lane: H's verify on 'head-host' is held until ``release_h``."""

    scene: _Scene
    lane: MergeLane
    queue: asyncio.Queue[Any]
    reqs: dict[str, MergeRequest]
    changes: Mapping[str, Tree]
    local: _TreeRecordingVerifier
    head_host: _TreeRecordingRemote
    release_h: asyncio.Event

    def put(self, *names: str) -> None:
        for name in names:
            self.queue.put_nowait(self.reqs[name])

    async def awaiting_dispatch(self, *names: str) -> None:
        """Wait until each of *names* is built and waits for a host to verify on."""
        for name in names:
            await _until(
                lambda name=name: lane_state(self.lane, self.reqs[name].request_id)
                in _AWAITING_DISPATCH,
                label=f'{name} built and waiting for a host',
            )

    async def outcome_status(self, name: str) -> str:
        outcome = await wait_responsive(self.reqs[name].result, label=f'{name} outcome')
        return outcome.status

    def local_trees_of(self, task_id: str) -> list[Tree]:
        return [tree for task, tree in self.local.trees if task == task_id]

    def remote_trees_of(self, task_id: str) -> list[Tree]:
        """The trees 'head-host' verified for commits bringing in *task_id*'s branch."""
        return [v.tree for v in self.head_host.verifies if v.change == self.changes[task_id]]

    def first_merge_base(self, task_id: str) -> str:
        """The base_sha of *task_id*'s first speculative merge."""
        merges = [
            e for e in self.scene.events.fetch_events_by_type(EventType.speculative_merge)
            if e.get('task_id') == task_id
        ]
        assert merges, f'{task_id} never merged speculatively'
        return merges[0]['data']['base_sha']

    def dispatch_voids_of(self, task_id: str) -> list[dict[str, Any]]:
        return [
            e['data'] for e in self.scene.events.fetch_events_by_type(EventType.verdict_voided)
            if e.get('task_id') == task_id and e['data']['point'] == 'dispatch'
        ]

    def divergence_alarms(self) -> list[Any]:
        return [
            esc for esc in self.scene.escalations.filed
            if esc.task_id == VERIFY_BASE_DIVERGENCE_SENTINEL
        ]


@contextlib.asynccontextmanager
async def _running(
    scene: _Scene, names: Sequence[str], *, local_scripts: Mapping[str, Sequence[VerifyScript]],
    speculation_depth: int,
) -> AsyncIterator[_Run]:
    reqs = {name: await _request(scene, name) for name in names}
    changes = {
        name: await _change_of(scene.repo, req.branch.full_name) for name, req in reqs.items()
    }
    h_entered, release_h = asyncio.Event(), asyncio.Event()
    head_host = _TreeRecordingRemote(_HEAD_HOST, scene.repo, release=release_h, entered=h_entered)
    local = _TreeRecordingVerifier(local_scripts)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=local, clock=healthy_verify_clock(),
        speculation_depth=speculation_depth, remote_runners=[head_host],
        event_store=scene.events, escalation_queue=scene.escalations,
    )
    async with running_lane(lane):
        run = _Run(scene, lane, queue, reqs, changes, local, head_host, release_h)
        run.put('H')
        await wait_responsive(
            h_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT,
            label='H verifying on head-host',
        )
        yield run


async def _stage_behind_a_local_verify(
    run: _Run, verifying: str, entered: asyncio.Event, stacked: Sequence[str],
) -> str:
    """*verifying* merged on H and held on local, each of *stacked* merged
    speculatively behind it and waiting for a host, T last and on S; returns
    S's original merge commit, read off T's speculative merge."""
    run.put(verifying)
    await wait_responsive(
        entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT,
        label=f'{verifying} verifying on local',
    )
    run.put(*stacked)
    await run.awaiting_dispatch(*stacked)
    s_commit = run.first_merge_base('T')
    assert 'S.txt' in await _tree_of(run.scene.repo, s_commit), 'T was not stacked on S'
    on_main, _ = await _git(run.scene.repo, 'merge-base', '--is-ancestor', s_commit, 'main')
    assert on_main == 1, f"S's speculative merge {s_commit} is already on main"
    return s_commit


async def _t_verified(run: _Run) -> None:
    await _until(lambda: bool(run.local_trees_of('T')), label="T's local verify entered")


class TestARequeuedPredecessorsCommitIsDead:
    """A build whose request went back to the queue left a commit that will never land."""

    async def test_a_successor_of_a_requeued_predecessor_never_verifies_on_its_commit(
        self, tmp_path: Path,
    ) -> None:
        """S's local verify finds the merge-verify lane held and defers, so its
        request is requeued and the local host frees while H still verifies.
        T, built on S's commit, must be re-merged before it verifies."""
        scene = _scene(tmp_path)
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        lane_held = raises(MergeVerifyLeaseHeld(scene.repo / '_merge-verify', 4242))
        s_defers = dataclasses.replace(lane_held, release=release_s, entered=s_entered)
        async with _running(
            scene, ('H', 'S', 'T'), local_scripts={'S': [s_defers]}, speculation_depth=2,
        ) as run:
            s_commit = await _stage_behind_a_local_verify(run, 'S', s_entered, ('T',))

            release_s.set()
            await _t_verified(run)
            assert all('S.txt' not in tree for tree in run.local_trees_of('T')), (
                f"T verified on requeued S's commit: {run.local_trees_of('T')}"
            )
            voids = run.dispatch_voids_of('T')
            assert [(v['dead_link'], v['killer']) for v in voids] == [
                (s_commit, {'site': 'requeued', 'task_id': 'S'}),
            ], voids
            assert run.divergence_alarms() == []

            run.release_h.set()
            statuses = {name: await run.outcome_status(name) for name in ('H', 'T', 'S')}
            assert statuses == dict.fromkeys(('H', 'T', 'S'), 'done'), statuses
            assert await _tree_of(scene.repo, 'main') == {'H.txt', 'S.txt', 'T.txt'}
            assert run.divergence_alarms() == []


class TestAnAbandonedPredecessorsCommitIsDead:
    """A build whose request was abandoned left a commit that will never land."""

    async def test_a_successor_of_a_predecessor_cancelled_mid_verify_never_verifies_on_its_commit(
        self, tmp_path: Path,
    ) -> None:
        """S's waiter goes away while S verifies on local, so the lane drops
        S and the local host frees while H still verifies. T, built on S's
        commit, must be re-merged before it verifies."""
        scene = _scene(tmp_path)
        s_entered = asyncio.Event()
        s_holds = hangs_until(asyncio.Event(), entered=s_entered)
        async with _running(
            scene, ('H', 'S', 'T'), local_scripts={'S': [s_holds]}, speculation_depth=2,
        ) as run:
            s_commit = await _stage_behind_a_local_verify(run, 'S', s_entered, ('T',))

            run.reqs['S'].result.cancel()
            await _t_verified(run)
            assert all('S.txt' not in tree for tree in run.local_trees_of('T')), (
                f"T verified on abandoned S's commit: {run.local_trees_of('T')}"
            )
            voids = run.dispatch_voids_of('T')
            assert [(v['dead_link'], v['killer']) for v in voids] == [
                (s_commit, {'site': 'abandoned', 'task_id': 'S'}),
            ], voids
            assert run.divergence_alarms() == []

            run.release_h.set()
            statuses = {name: await run.outcome_status(name) for name in ('H', 'T')}
            assert statuses == dict.fromkeys(('H', 'T'), 'done'), statuses
            assert await _tree_of(scene.repo, 'main') == {'H.txt', 'T.txt'}
            assert run.divergence_alarms() == []

    async def test_a_successor_of_a_predecessor_cancelled_while_parked_never_verifies_on_its_commit(
        self, tmp_path: Path,
    ) -> None:
        """X verifies on local behind H; S, built on X, and T, built on S,
        wait for a host. S's waiter goes away, then H lands and frees
        'head-host' while X still verifies. S is abandoned at its dispatch,
        and T, dispatched on 'head-host', must be re-merged before it
        verifies."""
        scene = _scene(tmp_path)
        x_entered, release_x = asyncio.Event(), asyncio.Event()
        x_holds = hangs_until(release_x, entered=x_entered)
        async with _running(
            scene, ('H', 'X', 'S', 'T'), local_scripts={'X': [x_holds]}, speculation_depth=3,
        ) as run:
            s_commit = await _stage_behind_a_local_verify(run, 'X', x_entered, ('S', 'T'))

            run.reqs['S'].result.cancel()
            run.release_h.set()
            assert await run.outcome_status('H') == 'done'
            await _until(
                lambda: bool(run.remote_trees_of('T')), label="T's verify on head-host",
            )
            assert lane_state(run.lane, run.reqs['X'].request_id) == 'verifying', (
                'precondition: T dispatched while X still verified'
            )
            assert all('S.txt' not in tree for tree in run.remote_trees_of('T')), (
                f"T verified on abandoned S's commit: {run.remote_trees_of('T')}"
            )
            voids = run.dispatch_voids_of('T')
            assert [(v['dead_link'], v['killer']) for v in voids] == [
                (s_commit, {'site': 'abandoned', 'task_id': 'S'}),
            ], voids
            assert run.divergence_alarms() == []

            release_x.set()
            statuses = {name: await run.outcome_status(name) for name in ('X', 'T')}
            assert statuses == dict.fromkeys(('X', 'T'), 'done'), statuses
            assert await _tree_of(scene.repo, 'main') == {'H.txt', 'X.txt', 'T.txt'}
            assert run.divergence_alarms() == []


async def _t_verified_on_s_behind_it(run: _Run) -> None:
    """H lands and frees 'head-host'; T dispatches there on S's still-live
    commit while S verifies on local, and its verify passes."""
    run.release_h.set()
    assert await run.outcome_status('H') == 'done'
    await _until(
        lambda: bool(run.remote_trees_of('T')) and not run.head_host.live,
        label="T's verify on head-host passed",
    )
    assert run.remote_trees_of('T') == [_ON_S_LIVE_COMMIT]
    assert lane_state(run.lane, run.reqs['S'].request_id) == 'verifying', (
        'precondition: T verified while S still verified'
    )


def _assert_t_reverified_without_s(run: _Run) -> None:
    """T's only verify on S's work is the one taken on S's live commit."""
    t_trees = run.local_trees_of('T') + run.remote_trees_of('T')
    assert [tree for tree in t_trees if 'S.txt' in tree] == [_ON_S_LIVE_COMMIT], t_trees
    assert any('S.txt' not in tree for tree in t_trees), t_trees


class TestAVerdictTakenOnALivePredecessorIsNotAdoptedOnceItLeaves:
    """T verified on S's commit while S still verified; S then leaves the lane
    unlanded, so T's verdict describes a tree that will never be on main.
    T must be re-verified without S's work, never adopted on that tree."""

    async def test_a_verdict_on_a_predecessor_requeued_after_it_is_not_adopted(
        self, tmp_path: Path,
    ) -> None:
        """S's local verify defers once T's verify has passed on S's commit,
        so S's request is requeued and merges afresh."""
        scene = _scene(tmp_path)
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        lane_held = raises(MergeVerifyLeaseHeld(scene.repo / '_merge-verify', 4242))
        s_defers = dataclasses.replace(lane_held, release=release_s, entered=s_entered)
        async with _running(
            scene, ('H', 'S', 'T'), local_scripts={'S': [s_defers]}, speculation_depth=2,
        ) as run:
            await _stage_behind_a_local_verify(run, 'S', s_entered, ('T',))
            await _t_verified_on_s_behind_it(run)

            release_s.set()
            statuses = {name: await run.outcome_status(name) for name in ('T', 'S')}
            assert statuses == dict.fromkeys(('T', 'S'), 'done'), statuses
            _assert_t_reverified_without_s(run)
            assert await _tree_of(scene.repo, 'main') == {'H.txt', 'S.txt', 'T.txt'}
            assert run.divergence_alarms() == []

    async def test_a_verdict_on_a_predecessor_dropped_after_it_is_not_adopted(
        self, tmp_path: Path,
    ) -> None:
        """S's waiter goes away once T's verify has passed on S's commit, so
        the lane drops S mid-verify. Adopting T's verdict would land S's work
        with T."""
        scene = _scene(tmp_path)
        s_entered = asyncio.Event()
        s_holds = hangs_until(asyncio.Event(), entered=s_entered)
        async with _running(
            scene, ('H', 'S', 'T'), local_scripts={'S': [s_holds]}, speculation_depth=2,
        ) as run:
            await _stage_behind_a_local_verify(run, 'S', s_entered, ('T',))
            await _t_verified_on_s_behind_it(run)

            run.reqs['S'].result.cancel()
            assert await run.outcome_status('T') == 'done'
            _assert_t_reverified_without_s(run)
            assert await _tree_of(scene.repo, 'main') == {'H.txt', 'T.txt'}
            assert run.divergence_alarms() == []

