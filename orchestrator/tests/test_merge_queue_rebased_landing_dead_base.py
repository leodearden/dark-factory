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
* mid-verify B (two hosts): its verdict is voided at adoption and re-run on
  real main, instead of being CAS-rebased through a gate re-verify;
* a head that lands by a plain advance supersedes nothing, so a successor on
  its merge commit is never voided, even after main moves on past it;
* a depth-2 follower C stacked on B, with no host free when B's verdict is
  voided at adoption, is re-merged onto real main at dispatch, because the
  void records B's own merge commit dead too (task 7352).

Each scene runs a real lane over a real tmp repo and observes only request
outcomes, the injected verifier and the injected event store.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, VerifyScript, lane_scene_config, passes
from _orch_helpers import wait_responsive
from test_merge_queue_concurrent_verify import (
    HEAVY_BARRIER_TEST_TIMEOUT,
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
)
from test_merge_speculation import _LateArrivalFakeEventStore

from orchestrator.config import GitConfig, OrchestratorConfig
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


async def _land_foreign_commit(repo: Path) -> None:
    """Move main the way a human direct commit does, disjoint from every branch.

    Stages only foreign.txt: ``add -A`` also stages the lane's live ``.worktrees/_merge-*``
    directories, and races their creation.
    """
    (repo / 'foreign.txt').write_text('landed elsewhere\n')
    await _git_until_it_takes(repo, 'add', 'foreign.txt')
    await _git_until_it_takes(repo, 'commit', '-m', 'Foreign landing')


def _events(store: _LateArrivalFakeEventStore, kind: EventType, task_id: str) -> list[dict]:
    return [e['data'] for e in store.speculative_events(kind) if e['task_id'] == task_id]


def _gate_retries(store: _LateArrivalFakeEventStore, task_id: str) -> list[dict]:
    return [
        data for data in _events(store, EventType.merge_attempt, task_id)
        if data.get('outcome') == 'gate_retry'
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
            assert [(v['point'], v['reason'], v['dead_link']) for v in voids] == [
                ('dispatch', 'chain_dead', a_pre),
            ]
            assert _gate_retries(store, 'rb-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []

    async def test_mid_verify_successor_verdict_is_voided_not_rebase_landed(
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
            await _assert_landed_rebased(store, req_a, a_pre)
            b_release.set()

            outcome_b = await wait_responsive(req_b.result, label='mv-b outcome')
            assert outcome_b.status == 'done', outcome_b
            voids = _events(store, EventType.verdict_voided, 'mv-b')
            assert [(v['point'], v['reason'], v['dead_link']) for v in voids] == [
                ('adoption', 'chain_dead', a_pre),
            ]
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
            b_release.set()

            outcome_b = await wait_responsive(req_b.result, label='pl-b outcome')
            assert outcome_b.status == 'done', outcome_b
            assert _events(store, EventType.verdict_voided, 'pl-b') == []
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestAdoptionVoidRecordsTheVoidedCommitDead:

    async def test_follower_with_no_host_when_its_base_is_voided_is_remerged_at_dispatch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
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

            await _land_foreign_commit(git_repo)
            head_gate.set()
            await _assert_landed_rebased(store, req_a, a_pre)
            await _until(
                lambda: allocator.held_local is not None,
                what='the concurrent acquirer to take the slot A freed',
            )
            assert allocator.free_host_count() == 0, 'precondition: C has no host'
            assert remote.run_merge_verify.await_count == 1, (
                'precondition: only B has verified, so C is not dispatched'
            )

            b_release.set()
            outcome_b = await wait_responsive(req_b.result, label='fv-b outcome')
            assert outcome_b.status == 'done', outcome_b
            b_voids = _events(store, EventType.verdict_voided, 'fv-b')
            assert [(v['point'], v['reason'], v['dead_link']) for v in b_voids] == [
                ('adoption', 'chain_dead', a_pre),
            ], 'precondition: B is voided at adoption'

            outcome_c = await wait_responsive(req_c.result, label='fv-c outcome')
            assert outcome_c.status == 'done', outcome_c
            c_voids = _events(store, EventType.verdict_voided, 'fv-c')
            assert [(v['point'], v['reason'], v['dead_link']) for v in c_voids] == [
                ('dispatch', 'chain_dead', b_pre),
            ]
            assert _gate_retries(store, 'fv-c') == []

            assert allocator.held_local is not None
            await allocator.release(allocator.held_local)
            assert lane.two_layer_invariants(await git_ops.get_main_sha()) == []
