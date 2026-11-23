"""The contract of ``merge_lane/verify_dispatch.py::VerifyDispatcher`` (PRD κ, task 5040).

The dispatcher dispatches one item onto a free verify host, runs its verify
on the lease it holds, and hands the item back through its ports when the
verify ends without a verdict. It is built here through its public constructor, over recording
port doubles, a ``FakeVerifier``, a ``FakeClock`` and a real merge on a real
repo, and observed only through what it returns and what the ports
recorded. Every expected value is today's ``SpeculativeMergeWorker``
behaviour; none is new.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeClock,
    FakeRemoteRunner,
    FakeVerifier,
    RecordingEscalations,
    fails,
    hangs_until,
    passes,
    raises,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeVerifyLeaseContended
from orchestrator.merge_lane.dead_base_ledger import DeadBaseSite, VoidPoint
from orchestrator.merge_lane.dispatch_ports import (
    ChainPlanning,
    DeadBases,
    DispatchCustody,
    DispatchRemerge,
    DispatchWorktrees,
    LaneVerifyContext,
    LeaseContention,
    MainTip,
    MergerAnchor,
    SpeculationPermits,
    VerifyFrontier,
    VerifyRetryBudgets,
)
from orchestrator.merge_lane.halt import HaltState
from orchestrator.merge_lane.intake import MERGE_LANES
from orchestrator.merge_lane.main_health_probe import MainHealthProbeHandles
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.types import (
    NO_DEEP_PLACEMENT,
    CapPermit,
    DecidedItem,
    InflightEntry,
    InflightStatus,
    InflightVerifyResult,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    NoHost,
    QueuedBranch,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
    SpecPermit,
)
from orchestrator.merge_lane.verify_dispatch import VerifyDispatcher
from orchestrator.merge_lane.verify_failure_outcome import DryRunInvestigationHandles
from orchestrator.verify_runner import HostLease, RunnerUnavailable

_LANE_SECS_PER_READING = 3600.0


def _git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


class _Ports:
    """Recording doubles of every port the dispatcher is built over."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.abandoned = False
        self.contended_elapsed = 0.0
        self.admission_halted = False
        self.in_verify = False
        self.previous_failed = False
        self.dead_link: str | None = None
        self.main = 'f' * 40
        self.remerged: Any = None
        self.placement_error: BaseException | None = None
        self._streak = 0
        self._cap_outs = 0

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]

    def record(self, *call: Any) -> None:
        self.calls.append(call)

    async def _teardown(self, lease: Any, verify_task: Any, task_id: str, **kwargs: Any) -> None:
        self.record('teardown_verify_task', task_id, kwargs.get('cancel'))
        if verify_task is not None and not verify_task.done():
            verify_task.cancel()
            with contextlib.suppress(BaseException):
                await verify_task

    async def _async_record(self, *call: Any) -> None:
        self.record(*call)

    def _advance(self, task_id: str, exc: BaseException) -> tuple[int, float]:
        self._streak += 1
        self.record('advance_streak', task_id)
        return self._streak, self.contended_elapsed

    def _next_cap_out(self) -> int:
        self._cap_outs += 1
        return self._cap_outs

    async def _release_at_verdict(self, verify: Any, slot: Any) -> InflightVerifyResult:
        return await verify()

    async def _main_sha(self, reader: str) -> str:
        self.record('main_sha_or_empty', reader)
        return self.main

    async def _remerge(self, req: MergeRequest, started: float | None, claim: RemergeClaim) -> Any:
        self.record('remerge', req.task_id)
        return self.remerged

    async def _deep_chain_placement(self, item: Any, depth: int, is_local: bool) -> Any:
        self.record('deep_chain_placement', depth)
        if self.placement_error is not None:
            raise self.placement_error
        return NO_DEEP_PLACEMENT

    def custody(self) -> DispatchCustody:
        return DispatchCustody(
            request_abandoned=lambda req: self.abandoned,
            note_verify_completed=lambda out, slot, lease: self.record('note_verify_completed', out),
            teardown_verify_task=self._teardown,
            drop_from_verify=lambda item, slot: self._async_record('drop_from_verify', item.request.task_id),
            requeue_from_verify=lambda item, slot: self._async_record(
                'requeue_from_verify', item.request.task_id,
            ),
            bench_from_verify=lambda slot, exc: self._async_record('bench_from_verify', str(exc)),
            bench_unavailable_runner=lambda name, exc: self.record('bench_unavailable_runner', name),
            adopt_head_on_tip_authority=lambda item: self._async_record('adopt', item.request.task_id),
            admission_halted=lambda: self.admission_halted,
            requeue_built_item=lambda item: self.record('requeue_built_item', item.request.task_id),
            retire_item=lambda rid: self.record('retire_item', rid),
            record_abandoned_item=lambda item: self.record('record_abandoned_item'),
            note_transition=lambda rid, before, after, live: self.record(
                'note_transition', before, after,
            ),
            release_at_verdict=self._release_at_verdict,
        )

    def frontier(self) -> VerifyFrontier:
        return VerifyFrontier(
            has_inflight_verify=lambda: self.in_verify,
            frontier_depth=lambda: 0,
            frontier_base_commits=lambda: (set(), set()),
            verify_base_divergence=lambda item, main: None,
        )

    def chain_planning(self) -> ChainPlanning:
        return ChainPlanning(
            probe_placement=lambda item: None,
            deep_chain_placement=self._deep_chain_placement,
            note_tip_result=lambda reason, depth, out: self.record('note_tip_result', depth),
            mark_next_dispatch_unchained=lambda ids: self.record('unchained', tuple(ids)),
            end_chain_reservation=lambda rid: self.record('end_chain_reservation', rid),
        )

    def permits(self) -> SpeculationPermits:
        return SpeculationPermits(
            release_cap_permit=lambda permit: self.record('release_cap_permit'),
            release_permit=lambda permit: self.record('release_permit'),
        )

    def remerge(self) -> DispatchRemerge:
        return DispatchRemerge(
            previous_failed=lambda: self.previous_failed,
            claim_remerge_base=lambda item, main, permit: RemergeClaim(
                anchor=RemergeBase(tip=None, reason=RemergeReason.NO_TIP), permit=None,
            ),
            emit_stale_discard=lambda task_id, reason, anchor: self.record(
                'emit_stale_discard', reason,
            ),
            remerge=self._remerge,
        )

    def main_tip(self) -> MainTip:
        return MainTip(
            main_sha_or_empty=self._main_sha,
            note_main_sha=lambda sha: self.record('note_main_sha'),
        )

    def dead_bases(self) -> DeadBases:
        return DeadBases(
            has_dead_bases=lambda: self.dead_link is not None,
            chain_dead_link=lambda item, main: self.dead_link,
            emit_dead_base_discard=lambda req, item, dead, point, anchor: self.record(
                'emit_dead_base_discard', dead, point,
            ),
            record_item_commit_dead=lambda item, site, notify: self.record(
                'record_item_commit_dead', site, notify,
            ),
        )

    def merger(self) -> MergerAnchor:
        return MergerAnchor(
            notify_merger_of_dead=lambda shas: self.record('notify_merger_of_dead'),
            reanchor_merger_on=lambda dead, replacement: self.record('reanchor_merger_on'),
        )

    def worktrees(self) -> DispatchWorktrees:
        return DispatchWorktrees(
            cleanup_owned=lambda wt: self._async_record('cleanup_owned', wt),
            release_or_cleanup=lambda wt, spec_warm: self._async_record('release_or_cleanup', wt),
            deregister_owned=lambda wt: self.record('deregister_owned', wt),
            owned=set,
        )

    def contention(self) -> LeaseContention:
        return LeaseContention(
            advance_streak=self._advance,
            clear_streak=lambda task_id: self.record('clear_streak', task_id),
            next_cap_out=self._next_cap_out,
        )


class _Hosts:
    """A one-slot host allocator double."""

    def __init__(self) -> None:
        self.free = 1
        self.acquire_error: BaseException | None = None
        self.acquired: list[HostLease] = []
        self.released: list[HostLease] = []

    def free_host_count(self) -> int:
        return self.free

    async def acquire(self, local_factory: Any, *, policy: Any = None) -> HostLease | None:
        if self.acquire_error is not None:
            raise self.acquire_error
        if not self.free:
            return None
        lease = _local_lease()
        self.acquired.append(lease)
        return lease

    async def release(self, lease: HostLease) -> None:
        self.released.append(lease)


class _Scene:
    def __init__(self, git_ops: GitOps, config: OrchestratorConfig) -> None:
        self.git_ops = git_ops
        self.config = config
        self.ports = _Ports()
        self.hosts = _Hosts()
        self.halts: list[str] = []
        self.escalations = RecordingEscalations()
        self.halt = HaltState(MERGE_LANES, on_resume=lambda: None)
        self._n = 0

    def dispatcher(self, verifier: FakeVerifier, clock: FakeClock | None = None) -> VerifyDispatcher:
        background: set[asyncio.Task[Any]] = set()
        ports = self.ports
        dispatcher = VerifyDispatcher(
            git_ops=lambda: self.git_ops,
            host_allocator=lambda config: cast(Any, self.hosts),
            clock=FakeClock() if clock is None else clock,
            verifier=verifier,
            halt=self.halt,
            lane_context=LaneVerifyContext(
                retry_budgets=lambda: VerifyRetryBudgets(
                    timeouts={}, enospc_retries={}, narrowed_retries={},
                    max_timeouts=3, max_enospc=1, max_narrowed=2,
                ),
                event_store=lambda: None,
                escalation_queue=lambda: self.escalations,
                quarantine_divergent=lambda name: ports.record('quarantine_divergent', name),
                halt_hook=self.halts.append,
                dry_run_handles=lambda: DryRunInvestigationHandles(
                    scheduler=None, background_tasks=background,
                ),
                main_health_probe=lambda: MainHealthProbeHandles(
                    background_tasks=background, auto_heal=None,
                ),
            ),
            custody=ports.custody(),
            frontier=ports.frontier(),
            chain_planning=ports.chain_planning(),
            permits=ports.permits(),
            remerge=ports.remerge(),
            main_tip=ports.main_tip(),
            dead_bases=ports.dead_bases(),
            merger=ports.merger(),
            worktrees=ports.worktrees(),
            contention=ports.contention(),
        )
        dispatcher.VERIFY_ABANDON_POLL_SECS = 0.02
        return dispatcher

    def decided_item(self, item: RealMergeItem) -> DecidedItem:
        return DecidedItem(
            request=item.request,
            immediate_outcome=MergeOutcome(status='conflict', reason='decided'),
            base_sha=item.base_sha,
            speculative=False,
        )

    async def merged_item(self, task_id: str = 'vd-1') -> RealMergeItem:
        self._n += 1
        branch_id = f'{task_id}-{self._n}'
        worktree = (await self.git_ops.create_worktree(branch_id)).path
        (worktree / f'feature{self._n}.txt').write_text(f'feature {self._n}\n')
        await self.git_ops.commit(worktree, 'Add feature')
        merged = await self.git_ops.merge_to_main(worktree, f'task/{branch_id}')
        assert merged.success and merged.merge_worktree is not None, merged
        req = MergeRequest(
            task_id=task_id,
            branch=QueuedBranch.parse(f'task/{branch_id}', self.config.git.branch_prefix),
            worktree=worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=self.config,
            result=asyncio.get_running_loop().create_future(),
        )
        return RealMergeItem(
            request=req,
            merge_result=merged,
            merge_wt=merged.merge_worktree,
            base_sha=await self.git_ops.get_main_sha(),
            speculative=False,
        )


@pytest_asyncio.fixture
async def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo')
    config = OrchestratorConfig(
        project_root=repo, git=_git_config(), escalate_preexisting_main_break=False,
    )
    return _Scene(GitOps(_git_config(), repo), config)


def _local_lease() -> HostLease:
    return HostLease(name='local', runner=cast(Any, None), is_local=True)


async def _verify(
    dispatcher: VerifyDispatcher, item: RealMergeItem, lease: HostLease,
) -> InflightVerifyResult:
    return await asyncio.wait_for(dispatcher.run_inflight_verify(item, lease), timeout=30.0)


@pytest.mark.asyncio
class TestRunInflightVerify:
    async def test_a_local_pass_returns_no_outcome_and_counts_the_attempt(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()

        result = await _verify(dispatcher, item, _local_lease())

        assert result.outcome is None and result.status is None
        assert result.merge_wt is not None and result.merge_wt.exists()
        assert dispatcher.verify_attempts() == 1
        assert ('note_tip_result', 1) in scene.ports.calls

    async def test_a_categorised_failure_keeps_its_frozen_reason_prefix(self, scene: _Scene) -> None:
        verifier = FakeVerifier(fails(category='test_failure', summary='2 failed'))
        dispatcher = scene.dispatcher(verifier)
        item = await scene.merged_item()

        result = await _verify(dispatcher, item, _local_lease())

        assert result.outcome is not None and result.outcome.status == 'blocked'
        assert result.outcome.reason.startswith(
            'Post-merge verification failed: 2 failed [category: test_failure]',
        ), result.outcome.reason
        assert verifier.verified == ['vd-1']

    async def test_an_unavailable_remote_is_benched_for_the_caller_to_redispatch(
        self, scene: _Scene,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        remote = FakeRemoteRunner('laptop', unreachable=True)
        lease = HostLease(name='laptop', runner=cast(Any, remote), is_local=False)

        result = await _verify(dispatcher, item, lease)

        assert result.status == InflightStatus.RUNNER_UNAVAILABLE
        assert result.merge_wt == item.merge_wt
        assert 'bench_from_verify' in scene.ports.names()
        assert 'bench_unavailable_runner' not in scene.ports.names()
        assert dispatcher.verify_attempts() == 0
        assert not item.request.result.done()

    async def test_an_operator_halt_mid_verify_requeues_and_closes_the_streak(
        self, scene: _Scene,
    ) -> None:
        entered = asyncio.Event()
        dispatcher = scene.dispatcher(FakeVerifier(hangs_until(asyncio.Event(), entered=entered)))
        item = await scene.merged_item()

        running = asyncio.ensure_future(dispatcher.run_inflight_verify(item, _local_lease()))
        await wait_responsive(entered.wait(), label='the verify entered its run')
        scene.halt.operator_halt('test halt')
        result = await asyncio.wait_for(running, timeout=30.0)

        assert result.status == InflightStatus.REQUEUED
        assert scene.ports.names()[-2:] == ['requeue_from_verify', 'clear_streak']
        assert not item.request.result.done()

    async def test_a_sole_waiter_giving_up_drops_the_request(self, scene: _Scene) -> None:
        entered = asyncio.Event()
        dispatcher = scene.dispatcher(FakeVerifier(hangs_until(asyncio.Event(), entered=entered)))
        item = await scene.merged_item()

        running = asyncio.ensure_future(dispatcher.run_inflight_verify(item, _local_lease()))
        await wait_responsive(entered.wait(), label='the verify entered its run')
        scene.ports.abandoned = True
        result = await asyncio.wait_for(running, timeout=30.0)

        assert result.status == InflightStatus.DROPPED
        assert scene.ports.names()[-1] == 'drop_from_verify'

    async def test_dead_verifies_requeue_then_cap_out_blocked(self, scene: _Scene) -> None:
        verifier = FakeVerifier(hangs_until(asyncio.Event()))
        dispatcher = scene.dispatcher(verifier, FakeClock(tick=_LANE_SECS_PER_READING))
        dispatcher.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2

        first = await _verify(dispatcher, await scene.merged_item(), _local_lease())
        assert first.status == InflightStatus.REQUEUED
        assert dispatcher.dead_verify_aborts('vd-1') == 1

        capped_item = await scene.merged_item()
        capped = await _verify(dispatcher, capped_item, _local_lease())

        assert capped.status is None
        assert capped.outcome is not None and capped.outcome.status == 'blocked'
        assert capped_item.request.result.result() is capped.outcome
        assert dispatcher.dead_verify_aborts('vd-1') == 0

    async def test_a_contended_lane_defers_then_caps_out(self, scene: _Scene) -> None:
        contended = MergeVerifyLeaseContended(Path('/x/_merge-verify.lock'), 300.0)
        dispatcher = scene.dispatcher(FakeVerifier(raises(contended)))

        deferred = await _verify(dispatcher, await scene.merged_item(), _local_lease())
        assert deferred.status == InflightStatus.REQUEUED
        assert scene.ports.names()[-1] == 'requeue_from_verify'

        scene.ports.contended_elapsed = dispatcher.MAX_CONTENDED_LEASE_DEFER_SECS
        capped = await _verify(dispatcher, await scene.merged_item(), _local_lease())

        assert capped.outcome is not None and capped.outcome.status == 'blocked'
        assert capped.outcome.reason.startswith(
            'merge-verify lane unavailable for 14400s across 2 consecutive deferred '
            'attempts (lane cap-out #1): ',
        ), capped.outcome.reason

    async def test_a_runner_unavailable_escaping_an_external_cancel_is_benched(
        self, scene: _Scene,
    ) -> None:
        release, entered = asyncio.Event(), asyncio.Event()
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        remote = _UnwindsUnavailable('laptop', release=release, entered=entered)
        lease = HostLease(name='laptop', runner=cast(Any, remote), is_local=False)

        running = asyncio.ensure_future(dispatcher.run_inflight_verify(item, lease))
        await wait_responsive(entered.wait(), label='the verify entered its run')
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=30.0)

        assert ('bench_unavailable_runner', 'laptop') in scene.ports.calls


class _UnwindsUnavailable(FakeRemoteRunner):
    """A remote whose verify raises RunnerUnavailable when it is cancelled."""

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> Any:
        try:
            return await super().run_merge_verify(merge_sha, spec)
        except asyncio.CancelledError:
            raise RunnerUnavailable(f'{self.name} went away') from None


@pytest.mark.asyncio
class TestPostMergeVerify:
    async def test_it_runs_on_the_dispatchers_own_verifier(self, scene: _Scene) -> None:
        verifier = FakeVerifier(passes())
        dispatcher = scene.dispatcher(verifier)
        item = await scene.merged_item()

        outcome = await dispatcher.post_merge_verify(
            scene.git_ops, item.request, item.merge_wt,
            timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
            merge_sha=item.merge_result.merge_commit,
        )

        assert outcome is None
        assert verifier.verified == ['vd-1']

    async def test_a_gate_history_leg_verifies_locally_without_a_slot(self, scene: _Scene) -> None:
        verifier = FakeVerifier(passes())
        dispatcher = scene.dispatcher(verifier)
        item = await scene.merged_item()

        outcome = await dispatcher.gate_history_leg(
            scene.git_ops, item.request, item.merge_wt,
            timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
            merge_sha=item.merge_result.merge_commit,
        )

        assert outcome is None
        assert verifier.verified == ['vd-1']


async def _launched(entry: InflightEntry | NoHost) -> InflightEntry:
    assert isinstance(entry, InflightEntry) and entry.verify_task is not None, entry
    await asyncio.wait_for(entry.verify_task, timeout=30.0)
    return entry


_DEAD_LINK = 'd' * 40
_INTO_REMERGE = ('note_transition', ItemLifecycleState.DISPATCHING, ItemLifecycleState.MERGING)
_OUT_OF_REMERGE = ('note_transition', ItemLifecycleState.MERGING, ItemLifecycleState.DISPATCHING)


def _stale_discards(ports: _Ports) -> list[str]:
    return [call[1] for call in ports.calls if call[0] == 'emit_stale_discard']


@pytest.mark.asyncio
class TestDispatch:
    async def test_the_cap_permit_is_released_before_any_branching(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = dataclasses.replace(await scene.merged_item(), cap_permit=CapPermit())
        scene.ports.abandoned = True

        entry = await dispatcher.dispatch(item)

        assert scene.ports.names()[0] == 'release_cap_permit'
        assert isinstance(entry, InflightEntry)
        assert isinstance(entry.item, RealMergeItem) and entry.item.cap_permit is None

    async def test_a_predispatch_abandon_cleans_up_and_retires_the_request(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        scene.ports.abandoned = True

        entry = await dispatcher.dispatch(item)

        assert isinstance(entry, InflightEntry)
        assert entry.status == InflightStatus.ABANDONED_PREDISPATCH
        assert entry.lease is None and entry.verify_task is None
        assert scene.ports.names() == ['cleanup_owned', 'record_abandoned_item', 'retire_item']
        assert scene.hosts.acquired == []

    @pytest.mark.parametrize('halt', ['operator', 'admission'])
    async def test_a_halt_requeues_a_built_item(self, scene: _Scene, halt: str) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        if halt == 'operator':
            scene.halt.operator_halt('test halt')
        else:
            scene.ports.admission_halted = True

        entry = await dispatcher.dispatch(item)

        assert isinstance(entry, InflightEntry)
        assert entry.status == InflightStatus.REQUEUED_PREDISPATCH
        assert scene.ports.names() == ['cleanup_owned', 'requeue_built_item']
        assert not item.request.result.done()

    async def test_a_decided_item_passes_through_a_halt(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        decided = scene.decided_item(await scene.merged_item())
        scene.halt.operator_halt('test halt')

        entry = await dispatcher.dispatch(decided)

        assert isinstance(entry, InflightEntry)
        assert entry.passthrough_outcome is decided.immediate_outcome and entry.status is None
        assert scene.ports.calls == []

    async def test_no_free_host_parks_the_item_before_any_remerge_work(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        scene.hosts.free = 0
        scene.ports.dead_link = _DEAD_LINK

        parked = await dispatcher.dispatch(item)

        assert parked == NoHost(item)
        assert scene.ports.calls == []

    async def test_a_free_host_launches_the_verify_on_its_lease(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        scene.ports.main = item.base_sha

        entry = await _launched(await dispatcher.dispatch(item))

        assert entry.lease is scene.hosts.acquired[0] and entry.item is item
        assert entry.verify_wt is not None and entry.started_at is not None
        assert entry.verify_task is not None and entry.verify_task.result().outcome is None
        assert 'remerge' not in scene.ports.names()
        assert ('deep_chain_placement', 0) in scene.ports.calls

    async def test_a_dead_base_straggler_is_voided_and_remerged_instead_of_verified(
        self, scene: _Scene,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = dataclasses.replace(await scene.merged_item(), speculative=True)
        scene.ports.dead_link = _DEAD_LINK
        scene.ports.remerged = scene.decided_item(item)

        entry = await dispatcher.dispatch(item)

        assert scene.ports.calls == [
            ('main_sha_or_empty', 'dispatch chain check'),
            _INTO_REMERGE,
            ('cleanup_owned', item.merge_wt),
            ('emit_dead_base_discard', _DEAD_LINK, VoidPoint.DISPATCH),
            ('record_item_commit_dead', DeadBaseSite.DISPATCH_VOID, False),
            ('remerge', 'vd-1'),
            ('reanchor_merger_on',),
            _OUT_OF_REMERGE,
        ]
        assert isinstance(entry, InflightEntry) and entry.passthrough_outcome is not None
        assert entry.was_speculative and entry.started_at is not None
        assert scene.hosts.acquired == []

    @pytest.mark.parametrize(
        ('speculative', 'previous_failed', 'chain_invalidated', 'reason'),
        [
            (True, True, False, 'previous_failed'),
            (True, True, True, 'previous_failed'),
            (True, False, True, 'chain_invalidated'),
            (False, False, False, 'main_advanced'),
        ],
    )
    async def test_a_stale_merge_is_remerged_with_its_reason(
        self, scene: _Scene, speculative: bool, previous_failed: bool,
        chain_invalidated: bool, reason: str,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = dataclasses.replace(await scene.merged_item(), speculative=speculative)
        scene.ports.previous_failed = previous_failed
        scene.ports.remerged = scene.decided_item(item)
        if chain_invalidated:
            dispatcher.note_chain_invalidated()

        entry = await dispatcher.dispatch(item)

        assert _stale_discards(scene.ports) == [reason]
        assert ('record_item_commit_dead', DeadBaseSite.STALE_DISCARD, False) in scene.ports.calls
        assert scene.ports.names().count('note_transition') == 2
        assert isinstance(entry, InflightEntry) and entry.passthrough_outcome is not None

    async def test_a_speculative_item_off_main_is_not_stale(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = dataclasses.replace(await scene.merged_item(), speculative=True)

        await _launched(await dispatcher.dispatch(item))

        assert 'remerge' not in scene.ports.names()
        assert 'main_sha_or_empty' not in scene.ports.names()

    async def test_no_staleness_remerge_while_a_verify_is_in_flight(self, scene: _Scene) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        scene.ports.in_verify = True
        scene.ports.previous_failed = True
        dispatcher.note_chain_invalidated()

        await _launched(await dispatcher.dispatch(item))

        assert 'remerge' not in scene.ports.names()
        assert 'main_sha_or_empty' not in scene.ports.names()

    async def test_chain_invalidation_lasts_until_a_dispatch_that_does_not_remerge(
        self, scene: _Scene,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        first = dataclasses.replace(await scene.merged_item(), speculative=True)
        scene.ports.remerged = scene.decided_item(first)
        dispatcher.note_chain_invalidated()

        await dispatcher.dispatch(first)
        await dispatcher.dispatch(dataclasses.replace(first))
        assert _stale_discards(scene.ports) == ['chain_invalidated', 'chain_invalidated']

        assert dispatcher.chain_invalidated()

        await dispatcher.dispatch(scene.decided_item(first))
        assert not dispatcher.chain_invalidated()
        await _launched(await dispatcher.dispatch(
            dataclasses.replace(await scene.merged_item(), speculative=True),
        ))
        assert _stale_discards(scene.ports) == ['chain_invalidated', 'chain_invalidated']

    async def test_a_cancel_in_deep_placement_releases_the_lease_and_permit(
        self, scene: _Scene,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = dataclasses.replace(await scene.merged_item(), permit=SpecPermit())
        scene.ports.main = item.base_sha
        scene.ports.placement_error = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await dispatcher.dispatch(item)

        assert scene.hosts.released == scene.hosts.acquired != []
        assert scene.ports.names()[-1] == 'release_permit'

    async def test_a_launch_raising_after_a_remerge_discards_the_unlaunched_replacement(
        self, scene: _Scene,
    ) -> None:
        dispatcher = scene.dispatcher(FakeVerifier(passes()))
        item = await scene.merged_item()
        replacement = dataclasses.replace(await scene.merged_item(), permit=SpecPermit())
        scene.ports.remerged = replacement
        scene.hosts.acquire_error = RuntimeError('acquire failed')

        with pytest.raises(RuntimeError, match='acquire failed'):
            await dispatcher.dispatch(item)

        assert _stale_discards(scene.ports) == ['main_advanced']
        assert scene.ports.calls[-4:] == [
            _OUT_OF_REMERGE,
            ('release_permit',),
            ('cleanup_owned', replacement.merge_wt),
            ('record_item_commit_dead', DeadBaseSite.DISPATCH_VOID, True),
        ]
        assert scene.hosts.acquired == []


_HOMES = {
    'orchestrator.merge_lane.verify_dispatch': ('VerifyDispatcher', 'GATE_HISTORY_LEG_FACTS'),
    'orchestrator.merge_lane.inflight_verify': (
        'InflightVerifyRun', 'InflightVerifyCall', 'InflightVerifyCounters',
        'resolve_dispatch_time_merge_base',
    ),
    'orchestrator.merge_lane.dispatch_ports': (
        'DispatchCustody', 'VerifyFrontier', 'ChainPlanning', 'SpeculationPermits',
        'DispatchRemerge', 'MainTip', 'DeadBases', 'MergerAnchor', 'DispatchWorktrees',
        'LeaseContention', 'LaneVerifyContext', 'VerifyRetryBudgets',
    ),
}
_GONE_FROM_WORKER_MODULE = (
    'resolve_dispatch_time_merge_base', '_slot_cancel', '_GATE_HISTORY_LEG_FACTS',
)
_GONE_FROM_WORKER_CLASS = (
    '_run_inflight_verify', '_lane_post_merge_verify', '_run_gate_history_leg',
    '_dispatch_item', '_launch_verify', '_remerge_at_dispatch', '_discard_unlaunched_replacement',
    'VERIFY_ABANDON_POLL_SECS', 'INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS',
    'INFLIGHT_VERIFY_PROGRESS_PROBE_SECS', 'ORPHAN_REAP_TIMEOUT_SECS',
    'MAX_INFLIGHT_DEAD_VERIFY_ABORTS', 'MAX_CONTENDED_LEASE_DEFER_SECS',
    'CONTENDED_LEASE_REQUEUE_WARN_STREAK', 'CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS',
)


@pytest.mark.parametrize(
    ('module', 'name'), [(m, n) for m, names in _HOMES.items() for n in names],
)
def test_each_verify_dispatch_name_lives_in_its_concern_module(module: str, name: str) -> None:
    assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize('name', _GONE_FROM_WORKER_MODULE)
def test_the_worker_module_no_longer_defines_the_moved_name(name: str) -> None:
    assert not hasattr(importlib.import_module('orchestrator.merge_lane.worker'), name)


@pytest.mark.parametrize('name', _GONE_FROM_WORKER_CLASS)
def test_the_worker_no_longer_defines_the_moved_member(name: str) -> None:
    worker = importlib.import_module('orchestrator.merge_lane.worker')
    assert not hasattr(worker.SpeculativeMergeWorker, name)
