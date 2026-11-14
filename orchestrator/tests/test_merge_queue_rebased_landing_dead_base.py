"""A rebase-landed head's pre-rebase merge commit is a dead base (task 6273, INV-3).

When main moves under head A's in-flight verify, A lands REBASED, so the
merge commit A was verified as (``a_pre``) never reaches main. A successor B
that the merger stacked on ``a_pre`` is then built on an orphaned base.
INV-3 (``plans/merge-verdict-integrity-prd.md`` §1) names "superseded" as a
chain-breaking event, so ``a_pre`` must be recorded dead and B caught by the
shared chain predicate (``merge_lane/worker.py::SpeculativeMergeWorker._chain_dead_link``)
before B's verdict is spent or adopted:

* built-awaiting-host B (one host): re-merged onto real main at dispatch,
  so its only verify runs on the rebased landing, and B lands without a gate
  re-verify of its own;
* mid-verify B (two hosts): A's gate re-verify becomes a dispatch of A's
  rebased commit, so B's verify is cancelled and B re-anchors on that commit
  at its own dispatch, landing with no voided verdict and no gate of its own
  (task 8525);
* a rebased head parked with no free host waits first in line, ahead of a
  later arrival, and takes the first host that frees (task 8525);
* a head that lands by a plain advance supersedes nothing, so a successor on
  its merge commit is never voided, even after main moves on past it;
* a depth-2 follower C stacked on B, with no host free when A's gate
  re-verify becomes a dispatch, is re-merged onto real main at its dispatch,
  because tearing B down records B's own merge commit dead too (tasks 7352,
  8525);
* a request's gate attempts leave the lane with it, so a resubmission after
  a red rebased head counts from one again (task 8525);
* under a main that moves inside every verify, the dispatched rebased heads
  are bounded by the gate-retry cap; the request is then blocked and its
  in-flight successor is re-merged by the head-failure cascade and lands
  (task 8525);
* a successor whose teardown raises when the gate dispatches is resolved
  blocked, and the rebased head still lands (task 8525).

Each scene runs a real lane over a real tmp repo and observes only request
outcomes, the injected verifier and the injected event store.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, VerifyScript, fails, lane_scene_config, passes
from _merge_worker_drive import HEAVY_BARRIER_TEST_TIMEOUT
from _orch_helpers import wait_responsive
from test_merge_queue_concurrent_verify import (
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
)
from test_merge_speculation import _LateArrivalFakeEventStore

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_lane.types import QueuedBranch
from orchestrator.merge_lane.worker import MergeOutcome, MergeRequest
from orchestrator.verify_runner import HostAllocator, HostLease

_STOP_TIMEOUT = 30.0
_SETTLE_TIMEOUT = 30.0


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


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
    return lane_scene_config(git_repo, git_config)


class _BaseRecordingVerifier(FakeVerifier):
    """A ``FakeVerifier`` that also records the first parent of each tree it verifies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.verified_bases: list[tuple[str | None, str]] = []

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ):
        _, first_parent, _ = await _run(['git', 'rev-parse', 'HEAD^1'], cwd=worktree)
        self.verified_bases.append((options.get('task_id'), first_parent.strip()))
        return await super().run_scoped(
            worktree, config, module_configs, task_files, **options,
        )

    def bases_for(self, task_id: str) -> list[str]:
        return [base for tid, base in self.verified_bases if tid == task_id]


class _LocalSlotTakenOnceFreed(HostAllocator):
    """A concurrent acquirer outside the verifier loop takes the local slot once it frees.

    The first time a local lease is released, it re-takes the slot through the
    public ``acquire_local``, as ``merge_lane/drift.py`` does. There is no await
    between the release and the re-acquire, so the verifier loop never sees
    the slot FREE.
    """

    held_local: HostLease | None = None

    async def release(self, lease: HostLease) -> None:
        await super().release(lease)
        if lease.is_local and self.held_local is None:
            self.held_local = self.acquire_local(lambda: None)


class _MainMovesInsideEveryVerifyOf(_BaseRecordingVerifier):
    """Lands a foreign commit inside every verify of *task_id*, so each of its CAS attempts rebases.

    The k-th such verify first waits until *remote* has been entered k times,
    so the successor stacked on the previous head is verifying when this
    head's gate dispatches it again.
    """

    def __init__(self, repo: Path, task_id: str, remote: Any) -> None:
        super().__init__()
        self._repo = repo
        self._task_id = task_id
        self._remote = remote
        self.moves = 0

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ):
        if options.get('task_id') == self._task_id:
            self.moves += 1
            moves = self.moves
            await _until(
                lambda: self._remote.run_merge_verify.await_count >= moves,
                what=f'the successor to be verifying under {self._task_id} verify {moves}',
            )
            await _land_foreign_commit(self._repo, f'move {moves}')
        return await super().run_scoped(
            worktree, config, module_configs, task_files, **options,
        )


class _FirstRemoteCancelRaises(HostAllocator):
    """The first cancel-and-release of a remote lease raises; every later one succeeds."""

    raised = False

    async def cancel_and_release(self, lease: HostLease | None, **kwargs: Any) -> bool:
        if lease is not None and not lease.is_local and not self.raised:
            self.raised = True
            raise RuntimeError('cancel RPC lost')
        return await super().cancel_and_release(lease, **kwargs)


async def _submitted(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str,
) -> MergeRequest:
    worktree = await _make_branch_with_file(
        git_ops, task_id, f'{task_id}.py', f'{task_id.replace("-", "_")} = 1\n',
    )
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        merge_first_enqueued_at=1000.0,
    )


@contextlib.asynccontextmanager
async def _running_lane(git_ops: GitOps, **ports: Any):
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = MergeLane(git_ops, queue, **ports)
    lane_task = asyncio.ensure_future(lane.run())
    try:
        yield lane, queue
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(lane.stop(), timeout=_STOP_TIMEOUT)
        lane_task.cancel()
        await asyncio.gather(lane_task, return_exceptions=True)


async def _until(predicate, *, what: str, timeout: float = _SETTLE_TIMEOUT) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f'timed out after {timeout}s waiting for {what}')


async def _git_until_it_takes(repo: Path, *argv: str) -> None:
    """Run one git command, retrying while the live lane holds the repo's index lock."""
    deadline = asyncio.get_running_loop().time() + _SETTLE_TIMEOUT
    while True:
        rc, _, err = await _run(['git', *argv], cwd=repo)
        if rc == 0:
            return
        assert 'index.lock' in err and asyncio.get_running_loop().time() < deadline, err
        await asyncio.sleep(0.05)


async def _land_foreign_commit(repo: Path, note: str = 'landed elsewhere') -> None:
    """Move main the way a human direct commit does, disjoint from every branch.

    Stages only foreign.txt: ``add -A`` also stages the lane's live ``.worktrees/_merge-*``
    directories, and races their creation.  Each move needs its own *note*.
    """
    (repo / 'foreign.txt').write_text(f'{note}\n')
    await _git_until_it_takes(repo, 'add', 'foreign.txt')
    await _git_until_it_takes(repo, 'commit', '-m', f'Foreign landing: {note}')


def _events(store: _LateArrivalFakeEventStore, kind: EventType, task_id: str) -> list[dict]:
    return [e['data'] for e in store.speculative_events(kind) if e['task_id'] == task_id]


def _gate_retries(store: _LateArrivalFakeEventStore, task_id: str) -> list[dict]:
    return [
        data for data in _events(store, EventType.merge_attempt, task_id)
        if data.get('outcome') == 'gate_retry'
    ]


def _merge_attempts(store: _LateArrivalFakeEventStore, task_id: str, outcome: str) -> list[dict]:
    return [
        data for data in _events(store, EventType.merge_attempt, task_id)
        if data.get('outcome') == outcome
    ]


def _rows(store: _LateArrivalFakeEventStore, task_id: str) -> list[tuple[EventType, dict]]:
    """Every row *store* holds for *task_id*, in emit order."""
    return [(e['type'], e['data']) for e in store.events if e['task_id'] == task_id]


async def _first_parent(repo: Path, commit: str) -> str:
    rc, out, err = await _run(['git', 'rev-parse', f'{commit}^1'], cwd=repo)
    assert rc == 0, err
    return out.strip()


def _awaiting_host(lane: MergeLane) -> list[str]:
    """The task ids waiting for a host, in the order the lane will dispatch them."""
    return [e['task_id'] for e in lane.snapshot()['entries'] if e['state'] == 'awaiting_host']


def _leased_task_ids(lane: MergeLane) -> list[str]:
    """The task ids holding a host lease; an entry released at its verdict holds none."""
    return [
        row['task_id'] for row in lane.snapshot()['occupancy']['leases']
        if row['activity'] != 'verified_pending_landing'
    ]


async def _stacked_base(store: _LateArrivalFakeEventStore, task_id: str) -> str:
    await _until(
        lambda: bool(_events(store, EventType.speculative_merge, task_id)),
        what=f'{task_id} to be built speculatively',
    )
    return _events(store, EventType.speculative_merge, task_id)[0]['base_sha']


async def _assert_landed_rebased(
    store: _LateArrivalFakeEventStore, req: MergeRequest, a_pre: str,
) -> MergeOutcome:
    outcome = await wait_responsive(req.result, label=f'{req.task_id} outcome')
    assert outcome.status == 'done', outcome
    assert _gate_retries(store, req.task_id), (
        f'precondition: {req.task_id} must land through the rebased path'
    )
    assert outcome.merge_sha != a_pre
    return outcome


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestRebasedLandingRecordsItsPreRebaseCommitDead:

    async def test_built_awaiting_host_successor_is_remerged_onto_real_main_before_verifying(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        head_gate = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'rb-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            req_a = await _submitted(git_ops, config, 'rb-a')
            req_b = await _submitted(git_ops, config, 'rb-b')
            await queue.put(req_a)
            await queue.put(req_b)
            await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
            a_pre = await _stacked_base(store, 'rb-b')
            assert verifier.bases_for('rb-b') == [], 'precondition: B awaits the host'

            await _land_foreign_commit(git_repo)
            head_gate.set()
            outcome_a = await _assert_landed_rebased(store, req_a, a_pre)

            outcome_b = await wait_responsive(req_b.result, label='rb-b outcome')
            assert outcome_b.status == 'done', outcome_b
            assert verifier.bases_for('rb-b')[0] == outcome_a.merge_sha, (
                f'B verified on {verifier.bases_for("rb-b")[0]!r}, '
                f'expected the rebased landing {outcome_a.merge_sha!r}, not a_pre {a_pre!r}'
            )
            voids = _events(store, EventType.verdict_voided, 'rb-b')
            assert [(v['point'], v['reason'], v['dead_link'], v['killer']) for v in voids] == [
                ('dispatch', 'chain_dead', a_pre, {'site': 'gate_rebased', 'task_id': 'rb-a'}),
            ]
            assert _gate_retries(store, 'rb-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []

    async def test_mid_verify_successor_reanchors_on_the_dispatched_rebased_head(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        head_gate = asyncio.Event()
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'mv-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            try:
                _inject_two_host_allocator(
                    lane, _gated_runner(b_release, b_entered, passed=True, name='remote-mv'),
                )
                req_a = await _submitted(git_ops, config, 'mv-a')
                req_b = await _submitted(git_ops, config, 'mv-b')
                await queue.put(req_a)
                await queue.put(req_b)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await wait_responsive(b_entered.wait(), label='B to enter its remote verify')
                a_pre = await _stacked_base(store, 'mv-b')

                await _land_foreign_commit(git_repo)
                head_gate.set()
                outcome_a = await _assert_landed_rebased(store, req_a, a_pre)
                a_rebased = outcome_a.merge_sha
                assert a_rebased
                mid_flight = lane.two_layer_invariants(await git_ops.get_main_sha())
                assert [
                    v for v in mid_flight if req_b.request_id in v and 'INV-3' in v
                ] == [], mid_flight
                assert lane.snapshot()['verify_base_divergences'] == []

                outcome_b = await wait_responsive(req_b.result, label='mv-b outcome')
            finally:
                b_release.set()

            assert outcome_b.status == 'done', outcome_b
            landings = [
                e['task_id'] for e in store.events
                if e['type'] is EventType.merge_attempt and e['data'].get('outcome') == 'done'
            ]
            assert landings == ['mv-a', 'mv-b'], landings

            a_rows = _rows(store, 'mv-a')
            [gate_at] = [
                i for i, (kind, data) in enumerate(a_rows)
                if kind is EventType.merge_reverify_gate and data['reverify'] is True
            ]
            assert a_rows[gate_at][1]['merge_sha'] == a_rebased
            rebased_verifies = [
                (i, data) for i, (kind, data) in enumerate(a_rows)
                if kind is EventType.merge_verify and data.get('merge_sha') == a_rebased
            ]
            assert rebased_verifies, a_rows
            verify_at, a_verify = rebased_verifies[0]
            assert gate_at < verify_at, a_rows
            assert a_rebased != a_pre
            assert (a_verify['speculative'], a_verify['base_relation']) == (False, 'main'), a_verify
            assert a_verify['runner_reason'] != 'gate_on_head_host', a_verify
            assert [row['gate_clearance'] for row in _gate_retries(store, 'mv-a')] == [
                'reverified',
            ]

            b_rows = _rows(store, 'mv-b')
            cancels = [
                (i, data) for i, (kind, data) in enumerate(b_rows)
                if kind is EventType.merge_verify_cancelled
            ]
            assert [data['cause'] for _, data in cancels] == ['gate_reanchor'], b_rows
            b_reverify = next(
                data for i, (kind, data) in enumerate(b_rows)
                if i > cancels[0][0] and kind is EventType.merge_verify
            )
            assert b_reverify['base_relation'] == 'rebased_head', b_reverify
            assert await _first_parent(git_repo, b_reverify['merge_sha']) == a_rebased

            assert _events(store, EventType.verdict_voided, 'mv-b') == []
            reanchors = [
                data for data in _events(store, EventType.speculative_discard, 'mv-b')
                if data.get('reason') == 'gate_reanchor'
            ]
            assert [
                (d['point'], d['dead_link'], d['killer'], d['anchor_reason'], d['anchor_base'])
                for d in reanchors
            ] == [
                ('dispatch', a_pre, {'site': 'gate_rebased', 'task_id': 'mv-a'}, 'live_tip',
                 a_rebased),
            ]
            assert reanchors[0]['item_commit']
            assert _events(store, EventType.merge_reverify_gate, 'mv-b') == []
            assert _gate_retries(store, 'mv-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []

    async def test_plain_landing_supersedes_nothing(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A's merge commit IS main after a plain advance, so it is never dead.

        Main then moves past it before B's verdict is adopted, so a spurious
        record would void B at adoption; B instead takes the ordinary
        moved-main path.
        """
        head_gate = asyncio.Event()
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'pl-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            _inject_two_host_allocator(
                lane, _gated_runner(b_release, b_entered, passed=True, name='remote-pl'),
            )
            req_a = await _submitted(git_ops, config, 'pl-a')
            req_b = await _submitted(git_ops, config, 'pl-b')
            await queue.put(req_a)
            await queue.put(req_b)
            await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
            await wait_responsive(b_entered.wait(), label='B to enter its remote verify')
            a_pre = await _stacked_base(store, 'pl-b')

            head_gate.set()
            outcome_a = await wait_responsive(req_a.result, label='pl-a outcome')
            assert outcome_a.status == 'done', outcome_a
            assert _gate_retries(store, 'pl-a') == [], 'precondition: A lands by a plain advance'
            assert await git_ops.get_main_sha() == a_pre

            await _land_foreign_commit(git_repo)
            # B's base is A's LANDED commit, never dead: pure main drift under
            # an in-flight verify is silent (task 5302).
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []
            b_release.set()

            outcome_b = await wait_responsive(req_b.result, label='pl-b outcome')
            assert outcome_b.status == 'done', outcome_b
            assert _events(store, EventType.verdict_voided, 'pl-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestTornDownSuccessorRecordsItsCommitDead:

    async def test_follower_with_no_host_when_its_base_is_torn_down_is_remerged_at_dispatch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # Deep chains off (task 7720): under the operational yaml's chain_cap
        # fv-b's dispatch would chain the buffered fv-c, which then is never
        # built speculatively as the host-starved follower this scene needs.
        config = config.model_copy(update={'merge_deep': MergeDeepConfig(chain_cap=0)})
        head_gate = asyncio.Event()
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'fv-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(
            git_ops, verifier=verifier, event_store=store, speculation_depth=2,
        ) as (lane, queue):
            remote = _gated_runner(b_release, b_entered, passed=True, name='remote-fv')
            allocator = _inject_two_host_allocator(
                lane, remote, allocator_cls=_LocalSlotTakenOnceFreed,
            )
            assert isinstance(allocator, _LocalSlotTakenOnceFreed)
            try:
                req_a = await _submitted(git_ops, config, 'fv-a')
                req_b = await _submitted(git_ops, config, 'fv-b')
                req_c = await _submitted(git_ops, config, 'fv-c')
                await queue.put(req_a)
                await queue.put(req_b)
                await queue.put(req_c)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await wait_responsive(b_entered.wait(), label='B to enter its remote verify')
                a_pre = await _stacked_base(store, 'fv-b')
                b_pre = await _stacked_base(store, 'fv-c')
                assert remote.run_merge_verify.await_count == 1, (
                    'precondition: only B has verified, so C is not dispatched'
                )

                await _land_foreign_commit(git_repo)
                head_gate.set()
                await _assert_landed_rebased(store, req_a, a_pre)
                assert allocator.held_local is not None, (
                    'precondition: the concurrent acquirer took the slot A freed, '
                    'so B and C each wait for the remote'
                )

                outcome_b = await wait_responsive(req_b.result, label='fv-b outcome')
                outcome_c = await wait_responsive(req_c.result, label='fv-c outcome')
            finally:
                b_release.set()

            assert (outcome_b.status, outcome_c.status) == ('done', 'done'), (outcome_b, outcome_c)
            b_cancels = _events(store, EventType.merge_verify_cancelled, 'fv-b')
            assert [c['cause'] for c in b_cancels] == ['gate_reanchor']
            # A's rebased head landed before B had a host, so B fell to main.
            b_voids = _events(store, EventType.verdict_voided, 'fv-b')
            assert [
                (v['point'], v['reason'], v['dead_link'], v['item_commit'], v['killer'])
                for v in b_voids
            ] == [
                ('dispatch', 'chain_dead', a_pre, b_pre,
                 {'site': 'gate_rebased', 'task_id': 'fv-a'}),
            ]
            c_voids = _events(store, EventType.verdict_voided, 'fv-c')
            # C's killer names B's teardown: the void-chain provenance.
            assert [(v['point'], v['reason'], v['dead_link'], v['killer']) for v in c_voids] == [
                ('dispatch', 'chain_dead', b_pre, {'site': 'gate_reanchor', 'task_id': 'fv-b'}),
            ]
            assert c_voids[0]['item_commit'] and c_voids[0]['item_commit'] != b_pre
            assert _gate_retries(store, 'fv-c') == []

            assert allocator.held_local is not None
            await allocator.release(allocator.held_local)
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestRebasedHeadWaitsFirstForAHost:
    """PRD §7 boundary row 6, at the serial K = 1 and at K = hosts + 1 (task 8526)."""

    @pytest.mark.parametrize('speculation_depth', [
        pytest.param(1, id='serial-k'),
        pytest.param(3, id='k-hosts-plus-1'),
    ])
    async def test_a_rebased_head_with_no_free_host_waits_first_in_line_and_takes_the_first_free_host(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, speculation_depth: int,
    ) -> None:
        head_gate = asyncio.Event()
        remote_release = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'wh-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(
            git_ops, verifier=verifier, event_store=store, speculation_depth=speculation_depth,
        ) as (lane, queue):
            remote = _gated_runner(remote_release, name='remote-wh')
            allocator = _inject_two_host_allocator(
                lane, remote, allocator_cls=_LocalSlotTakenOnceFreed,
            )
            assert isinstance(allocator, _LocalSlotTakenOnceFreed)
            # Held before A dispatches: with a host free under A's verify the
            # fill-ahead race would leave a queue getter that harvests X.
            held_remote = allocator.acquire_remote()
            assert held_remote is not None, 'precondition: the remote was free to hold'
            local_handed_back = False
            try:
                req_a = await _submitted(git_ops, config, 'wh-a')
                await queue.put(req_a)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)

                await _land_foreign_commit(git_repo)
                head_gate.set()
                await _until(
                    lambda: _awaiting_host(lane) == ['wh-a'] and allocator.held_local is not None,
                    what="A's rebased head to wait for a host, with both hosts held",
                )
                assert verifier.verified == ['wh-a'], verifier.verified
                assert 'wh-a' not in _leased_task_ids(lane), lane.snapshot()['occupancy']

                req_x = await _submitted(git_ops, config, 'wh-x')
                await queue.put(req_x)
                await _until(lambda: 'wh-x' in _awaiting_host(lane), what='X to wait for a host')
                assert _awaiting_host(lane) == ['wh-a', 'wh-x'], lane.snapshot()['entries']
                assert verifier.verified == ['wh-a'], verifier.verified

                assert allocator.held_local is not None
                local_handed_back = True
                await allocator.release(allocator.held_local)
                outcome_a = await wait_responsive(req_a.result, label='wh-a outcome')
                outcome_x = await wait_responsive(req_x.result, label='wh-x outcome')
            finally:
                remote_release.set()
                head_gate.set()
                if allocator.held_local is not None and not local_handed_back:
                    await allocator.release(allocator.held_local)
                await allocator.release(held_remote)

            assert (outcome_a.status, outcome_x.status) == ('done', 'done'), (outcome_a, outcome_x)
            assert verifier.verified == ['wh-a', 'wh-a', 'wh-x'], verifier.verified
            a_verifies = [
                data['merge_sha'] for kind, data in _rows(store, 'wh-a')
                if kind is EventType.merge_verify
            ]
            assert a_verifies[-1] == outcome_a.merge_sha, a_verifies
            landings = [
                e['task_id'] for e in store.events
                if e['type'] is EventType.merge_attempt and e['data'].get('outcome') == 'done'
            ]
            assert landings == ['wh-a', 'wh-x'], landings
            remote.run_merge_verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestDispatchedRebasedHeadGateAttempts:

    async def test_a_resubmission_after_a_red_rebased_head_counts_its_gate_attempts_afresh(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        verifier = _BaseRecordingVerifier(sequence=[
            VerifyScript(result=passes().result, release=first_release),
            fails(category='test_failure', summary='the rebased tree is red'),
            VerifyScript(result=passes().result, release=second_release),
            passes(),
        ])
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (_lane, queue):
            try:
                req = await _submitted(git_ops, config, 'rs-a')
                await queue.put(req)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await _land_foreign_commit(git_repo, 'first move')
                first_release.set()
                failed = await wait_responsive(req.result, label='rs-a first outcome')
                assert failed.status != 'done', failed
                assert verifier.verified == ['rs-a', 'rs-a'], (
                    "precondition: the red verify was the dispatched rebased head's"
                )

                again = dataclasses.replace(
                    req, result=asyncio.get_running_loop().create_future(),
                    request_id=f'{req.request_id}-again',
                )
                await queue.put(again)
                await asyncio.wait_for(verifier.await_entry(3), timeout=_SETTLE_TIMEOUT)
                await _land_foreign_commit(git_repo, 'second move')
                second_release.set()
                landed = await wait_responsive(again.result, label='rs-a resubmission outcome')
            finally:
                first_release.set()
                second_release.set()

        assert landed.status == 'done', landed
        assert verifier.verified == ['rs-a'] * 4, verifier.verified
        assert [row['attempt'] for row in _gate_retries(store, 'rs-a')] == [1]

    async def test_a_main_moving_inside_every_verify_exhausts_the_cap_and_cascades_the_successor(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        remote = _gated_runner(b_release, b_entered, passed=True, name='remote-cap')
        verifier = _MainMovesInsideEveryVerifyOf(git_repo, 'cap-a', remote)
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            try:
                _inject_two_host_allocator(lane, remote)
                cap = lane.MAX_CAS_RETRIES
                req_a = await _submitted(git_ops, config, 'cap-a')
                req_b = await _submitted(git_ops, config, 'cap-b')
                await queue.put(req_a)
                await queue.put(req_b)

                outcome_a = await wait_responsive(req_a.result, label='cap-a outcome')
                b_release.set()
                outcome_b = await wait_responsive(req_b.result, label='cap-b outcome')
            finally:
                b_release.set()

            assert outcome_a.status == 'blocked', outcome_a
            assert 'Gate retry limit exhausted' in (outcome_a.reason or ''), outcome_a
            assert verifier.verified.count('cap-a') == cap + 1, verifier.verified
            assert [row['attempt'] for row in _gate_retries(store, 'cap-a')] == list(
                range(1, cap + 1),
            )
            assert [row['attempt'] for row in _merge_attempts(store, 'cap-a', 'cas_exhausted')] == [
                cap + 1,
            ]

            assert outcome_b.status == 'done', outcome_b
            # B re-anchors on each dispatched rebased head, then the cascade
            # re-merges it onto main once the cap ends A.
            assert remote.run_merge_verify.await_count == cap + 1
            reanchors = [
                d for d in _events(store, EventType.speculative_discard, 'cap-b')
                if d.get('reason') == 'gate_reanchor'
            ]
            assert [d['point'] for d in reanchors] == ['dispatch'] * cap, reanchors
            assert [
                d['base_relation'] for d in _events(store, EventType.merge_verify, 'cap-b')
            ] == ['rebased_head'] * cap + ['main']
            assert _events(store, EventType.verdict_voided, 'cap-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestGateSuccessorTeardownErrorIsContained:

    async def test_a_successor_whose_teardown_raises_is_blocked_and_the_rebased_head_lands(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        head_gate = asyncio.Event()
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'te-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            try:
                allocator = _inject_two_host_allocator(
                    lane, _gated_runner(b_release, b_entered, passed=True, name='remote-te'),
                    allocator_cls=_FirstRemoteCancelRaises,
                )
                assert isinstance(allocator, _FirstRemoteCancelRaises)
                req_a = await _submitted(git_ops, config, 'te-a')
                req_b = await _submitted(git_ops, config, 'te-b')
                await queue.put(req_a)
                await queue.put(req_b)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await wait_responsive(b_entered.wait(), label='B to enter its remote verify')
                a_pre = await _stacked_base(store, 'te-b')

                await _land_foreign_commit(git_repo)
                head_gate.set()
                outcome_b = await wait_responsive(req_b.result, label='te-b outcome')
                outcome_a = await _assert_landed_rebased(store, req_a, a_pre)
            finally:
                b_release.set()

            assert allocator.raised, 'precondition: B\'s teardown raised'
            assert outcome_b.status == 'blocked', outcome_b
            assert 'cancel RPC lost' in (outcome_b.reason or ''), outcome_b
            assert outcome_a.status == 'done', outcome_a
            assert _leased_task_ids(lane) == [], lane.snapshot()['occupancy']
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []


class _FirstWorktreeOnTipFails(GitOps):
    """The first ephemeral verify worktree asked for on a commit whose first parent is *tip* fails."""

    tip: str | None = None
    failed_for: str | None = None

    async def create_ephemeral_verify_worktree(self, sha: str) -> Path:
        armed = self.tip is not None and self.failed_for is None
        if armed and await _first_parent(self.project_root, sha) == self.tip:
            self.failed_for = sha
            raise OSError(28, 'No space left on device')
        return await super().create_ephemeral_verify_worktree(sha)


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestRebasedHeadWithNoWorktreeIsRequeued:

    async def test_a_rebased_head_whose_worktree_cannot_be_made_is_requeued_and_lands(
        self, git_config: GitConfig, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        git_ops = _FirstWorktreeOnTipFails(git_config, git_repo)
        head_gate = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'nw-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
            try:
                req = await _submitted(git_ops, config, 'nw-a')
                await queue.put(req)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await _land_foreign_commit(git_repo)
                git_ops.tip = await git_ops.get_main_sha()
                head_gate.set()
                outcome = await wait_responsive(req.result, label='nw-a outcome')
            finally:
                head_gate.set()

            assert git_ops.failed_for is not None, (
                "precondition: the rebased head's worktree could not be made"
            )
            assert outcome.status == 'done', outcome
            assert outcome.merge_sha and outcome.merge_sha != git_ops.failed_for
            assert await _first_parent(git_repo, outcome.merge_sha) == git_ops.tip
            assert verifier.verified == ['nw-a', 'nw-a'], verifier.verified
            assert _gate_retries(store, 'nw-a') == []
            assert _leased_task_ids(lane) == [], lane.snapshot()['occupancy']
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []
