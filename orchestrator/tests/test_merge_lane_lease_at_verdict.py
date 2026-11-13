"""A verify host is released at the verdict, not at landing (task 8526, policy δ).

``plans/merge-lane-policy-prd.md`` §7 boundary rows 5 and 6, at K = hosts + 1:

* row 5: a follower whose laptop verify has passed frees the laptop at its
  verdict, so the next follower verifies there while the head still verifies,
  and all three land in submission order with no gate re-verify;
* row 6: a rebased head that finds both hosts held still waits first in line
  and takes the first host that frees, holding no lease while it waits.

Each scene runs a real lane over a real tmp repo and observes only the lane's
snapshot, request outcomes, the injected verifier and the injected event store.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, VerifyScript, lane_scene_config, passes
from _merge_worker_drive import HEAVY_BARRIER_TEST_TIMEOUT
from _orch_helpers import wait_responsive
from _verify_result_doubles import real_verify_result
from test_merge_queue_concurrent_verify import _gated_runner, _inject_two_host_allocator
from test_merge_queue_rebased_landing_dead_base import (
    _awaiting_host,
    _BaseRecordingVerifier,
    _land_foreign_commit,
    _LocalSlotTakenOnceFreed,
    _running_lane,
    _submitted,
    _until,
)
from test_merge_speculation import _LateArrivalFakeEventStore

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.verify import VerifyResult

_SETTLE_TIMEOUT = 30.0
_RELEASED = 'verified_pending_landing'


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
    """Policy pinned, deep chains off (task 7720), so each follower is its own dispatch."""
    config = lane_scene_config(
        git_repo, git_config, verify_host_policy='prefer_local',
    ).model_copy(update={'merge_deep': MergeDeepConfig(chain_cap=0)})
    assert config.merge_deep.chain_cap == 0
    return config


class _PerCallGatedRemote:
    """A fake remote whose k-th ``run_merge_verify`` call waits on its own release.

    ``entered[k]`` is set when call k starts; the call returns a pass once
    ``release[k]`` is set.
    """

    def __init__(self, name: str, calls: int) -> None:
        self.entered = [asyncio.Event() for _ in range(calls)]
        self.release = [asyncio.Event() for _ in range(calls)]
        self.runner = MagicMock()
        self.runner.name = name
        self.runner.is_local = False
        self.runner.run_merge_verify = AsyncMock(side_effect=self._verify)
        self.runner.cancel_verify = AsyncMock(return_value=0)
        self.runner.probe_clean = AsyncMock(return_value=True)
        self._calls = 0

    async def _verify(self, *_args: Any, **_kwargs: Any) -> VerifyResult:
        call = self._calls
        self._calls += 1
        self.entered[call].set()
        await self.release[call].wait()
        return real_verify_result(True)

    def release_all(self) -> None:
        for event in self.release:
            event.set()


def _occupancy(lane: MergeLane) -> dict:
    """The lane's occupancy, checked: no more leased frontier entries than hosts."""
    occ = lane.snapshot()['occupancy']
    leased = sum(len(task_ids) for task_ids in occ['inflight_by_host'].values())
    assert leased <= occ['hosts_total'], occ
    return occ


def _row(lane: MergeLane, task_id: str) -> tuple[str, str] | None:
    """*task_id*'s (host, activity) occupancy row, or None."""
    rows = [r for r in _occupancy(lane)['leases'] if r['task_id'] == task_id]
    return (rows[0]['host'], rows[0]['activity']) if rows else None


def _holding_a_lease(lane: MergeLane) -> list[str]:
    return [r['task_id'] for r in _occupancy(lane)['leases'] if r['activity'] != _RELEASED]


def _rows(store: _LateArrivalFakeEventStore, kind: EventType, task_id: str) -> list[dict]:
    return [e['data'] for e in store.speculative_events(kind) if e['task_id'] == task_id]


def _landings(store: _LateArrivalFakeEventStore) -> list[str]:
    """Task ids in the order the lane landed them (its ``merge_attempt`` done rows)."""
    return [
        e['task_id'] for e in store.events
        if e['type'] is EventType.merge_attempt and e['data'].get('outcome') == 'done'
    ]


async def _first_parent_chain(repo: Path, count: int) -> list[str]:
    rc, out, err = await _run(['git', 'rev-list', '--first-parent', f'-n{count}', 'main'], cwd=repo)
    assert rc == 0, err
    return out.split()


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestLaptopReleasedAtVerdict:

    async def test_a_verified_follower_frees_the_laptop_for_the_next_while_the_head_verifies(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        a_release = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'lv-a': VerifyScript(result=passes().result, release=a_release),
        })
        laptop = _PerCallGatedRemote('laptop', calls=2)
        store = _LateArrivalFakeEventStore()

        async with _running_lane(
            git_ops, verifier=verifier, event_store=store,
            speculation_depth=3, remote_runners=[laptop.runner],
        ) as (lane, queue):
            lane.VERIFY_ABANDON_POLL_SECS = 0.02
            try:
                reqs = [await _submitted(git_ops, config, tid) for tid in ('lv-a', 'lv-b', 'lv-c')]
                for req in reqs:
                    await queue.put(req)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
                await wait_responsive(laptop.entered[0].wait(), label='B to enter the laptop')

                laptop.release[0].set()
                await _until(
                    lambda: _row(lane, 'lv-b') == ('laptop', _RELEASED),
                    what="B's laptop lease to be released at its verdict",
                )
                await wait_responsive(
                    laptop.entered[1].wait(), label='C to enter the laptop while A verifies',
                )
                occ = _occupancy(lane)
                assert not reqs[0].result.done(), 'precondition: A still verifies'
                assert (occ['inflight_total'], occ['hosts_busy']) == (3, 2), occ
                assert all(len(tids) == 1 for tids in occ['inflight_by_host'].values()), occ
                assert occ['inflight_by_host'] == {'local': ['lv-a'], 'laptop': ['lv-c']}, occ

                laptop.release[1].set()
                await _until(
                    lambda: bool(_rows(store, EventType.merge_verify, 'lv-c')),
                    what="C's merge_verify row",
                )
                assert not reqs[0].result.done(), 'precondition: A still verifies'
                assert [
                    (row['runner'], row['base_relation'])
                    for row in _rows(store, EventType.merge_verify, 'lv-c')
                ] == [('laptop', 'stacked')]

                a_release.set()
                outcomes = [
                    await wait_responsive(req.result, label=f'{req.task_id} outcome')
                    for req in reqs
                ]
                assert lane.speculation_accounting_violations() == []
                _occupancy(lane)
            finally:
                a_release.set()
                laptop.release_all()

        assert [o.status for o in outcomes] == ['done', 'done', 'done'], outcomes
        assert _landings(store) == ['lv-a', 'lv-b', 'lv-c']
        assert await _first_parent_chain(git_repo, 3) == [o.merge_sha for o in reversed(outcomes)]
        assert [
            row for row in _rows(store, EventType.merge_attempt, 'lv-b')
            if row.get('outcome') == 'gate_retry'
        ] == []
        assert _rows(store, EventType.merge_reverify_gate, 'lv-b') == []


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestRebasedHeadWaitsForAHostAtKHostsPlusOne:

    async def test_a_rebased_head_with_no_free_host_waits_first_and_takes_the_first_free_host(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        head_gate = asyncio.Event()
        remote_release = asyncio.Event()
        verifier = _BaseRecordingVerifier(scripts={
            'wk-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(
            git_ops, verifier=verifier, event_store=store, speculation_depth=3,
        ) as (lane, queue):
            remote = _gated_runner(remote_release, name='remote-wk')
            allocator = _inject_two_host_allocator(
                lane, remote, allocator_cls=_LocalSlotTakenOnceFreed,
            )
            assert isinstance(allocator, _LocalSlotTakenOnceFreed)
            held_remote = allocator.acquire_remote()
            assert held_remote is not None, 'precondition: the remote was free to hold'
            local_handed_back = False
            try:
                req_a = await _submitted(git_ops, config, 'wk-a')
                await queue.put(req_a)
                await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)

                await _land_foreign_commit(git_repo)
                head_gate.set()
                await _until(
                    lambda: _awaiting_host(lane) == ['wk-a'] and allocator.held_local is not None,
                    what="A's rebased head to wait for a host, with both hosts held",
                )
                assert verifier.verified == ['wk-a'], verifier.verified
                assert 'wk-a' not in _holding_a_lease(lane), lane.snapshot()['occupancy']

                req_x = await _submitted(git_ops, config, 'wk-x')
                await queue.put(req_x)
                await _until(lambda: 'wk-x' in _awaiting_host(lane), what='X to wait for a host')
                assert _awaiting_host(lane) == ['wk-a', 'wk-x'], lane.snapshot()['entries']
                assert verifier.verified == ['wk-a'], verifier.verified

                assert allocator.held_local is not None
                local_handed_back = True
                await allocator.release(allocator.held_local)
                outcome_a = await wait_responsive(req_a.result, label='wk-a outcome')
                outcome_x = await wait_responsive(req_x.result, label='wk-x outcome')
            finally:
                remote_release.set()
                head_gate.set()
                if allocator.held_local is not None and not local_handed_back:
                    await allocator.release(allocator.held_local)
                await allocator.release(held_remote)

            assert (outcome_a.status, outcome_x.status) == ('done', 'done'), (outcome_a, outcome_x)
            assert verifier.verified == ['wk-a', 'wk-a', 'wk-x'], verifier.verified
            assert _landings(store) == ['wk-a', 'wk-x']
            remote.run_merge_verify.assert_not_awaited()
