"""The merge lane's policy instrumentation (task 8523, ``plans/merge-lane-policy-prd.md`` D10).

Every ``merge_verify`` row the lane writes says why the verify ran on its host
(``runner_reason``) and what the verified merge was built on
(``base_relation``), and every running verify the lane cancels leaves one
``merge_verify_cancelled`` row. Each scene runs a real lane over a real tmp
repo and observes only request outcomes, the lane's snapshot and the injected
event store.
"""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, VerifyScript, fails, lane_scene_config, passes
from _merge_worker_drive import HEAVY_BARRIER_TEST_TIMEOUT
from _orch_helpers import wait_responsive
from test_merge_queue_concurrent_verify import _gated_runner, _GatedVerifier
from test_merge_queue_rebased_landing_dead_base import (
    _land_foreign_commit,
    _running_lane,
    _submitted,
    _until,
)
from test_merge_speculation import _LateArrivalFakeEventStore

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane

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
    """Policy pinned so the operational yaml's ``verify_host_policy`` cannot leak in."""
    return lane_scene_config(git_repo, git_config, verify_host_policy='prefer_local')


def _rows(store: _LateArrivalFakeEventStore, kind: EventType, task_id: str) -> list[dict]:
    return [e['data'] for e in store.speculative_events(kind) if e['task_id'] == task_id]


def _dispatch_facts(store: _LateArrivalFakeEventStore, task_id: str) -> list[tuple]:
    return [
        (row['runner_reason'], row['base_relation'])
        for row in _rows(store, EventType.merge_verify, task_id)
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestOrdinaryDispatchFacts:

    async def test_single_host_verify_on_quiet_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        store = _LateArrivalFakeEventStore()
        async with _running_lane(git_ops, verifier=FakeVerifier(), event_store=store) as (_, queue):
            req = await _submitted(git_ops, config, 'in-solo')
            await queue.put(req)
            outcome = await wait_responsive(req.result, label='in-solo outcome')

        assert outcome.status == 'done', outcome
        assert _dispatch_facts(store, 'in-solo') == [('prefer_local_free', 'main')]

    async def test_follower_stacked_on_a_verifying_head_overflows_to_the_remote(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        head_gate = asyncio.Event()
        b_release = asyncio.Event()
        b_entered = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'in-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()
        remote = _gated_runner(b_release, b_entered, passed=True, name='remote-in')

        async with _running_lane(
            git_ops, verifier=verifier, event_store=store, remote_runners=[remote],
        ) as (_, queue):
            req_a = await _submitted(git_ops, config, 'in-a')
            req_b = await _submitted(git_ops, config, 'in-b')
            await queue.put(req_a)
            await queue.put(req_b)
            await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
            await wait_responsive(b_entered.wait(), label='B to enter its remote verify')
            head_gate.set()
            b_release.set()
            outcome_a = await wait_responsive(req_a.result, label='in-a outcome')
            outcome_b = await wait_responsive(req_b.result, label='in-b outcome')

        assert outcome_a.status == 'done', outcome_a
        assert outcome_b.status == 'done', outcome_b
        assert _dispatch_facts(store, 'in-a') == [('prefer_local_free', 'main')]
        assert _dispatch_facts(store, 'in-b') == [('local_busy', 'stacked')]
        assert [row['runner'] for row in _rows(store, EventType.merge_verify, 'in-b')] == [
            'remote-in',
        ]

    async def test_infra_transient_retry_runs_on_the_held_host(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        verifier = FakeVerifier(sequence=[
            fails(category='semaphore_timeout', summary='semaphore timeout'),
            passes(),
        ])
        store = _LateArrivalFakeEventStore()
        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (_, queue):
            req = await _submitted(git_ops, config, 'in-retry')
            await queue.put(req)
            outcome = await wait_responsive(req.result, label='in-retry outcome')

        assert outcome.status == 'done', outcome
        rows = _rows(store, EventType.merge_verify, 'in-retry')
        assert [row['attempt'] for row in rows] == [0, 1]
        assert _dispatch_facts(store, 'in-retry') == [
            ('prefer_local_free', 'main'), ('retry_held_host', 'main'),
        ]


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestGateReverifyDispatchFacts:

    async def test_a_rebased_head_is_dispatched_by_the_host_policy_against_main(
        self, git_ops: GitOps, git_repo: Path, git_config: GitConfig,
    ) -> None:
        config = lane_scene_config(
            git_repo, git_config, verify_host_policy='prefer_local',
            merge_verify_breadth='full', merge_gate_history_pure=False,
        )
        head_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'gt-a': VerifyScript(result=passes().result, release=head_gate),
        })
        store = _LateArrivalFakeEventStore()

        async with _running_lane(git_ops, verifier=verifier, event_store=store) as (_, queue):
            req = await _submitted(git_ops, config, 'gt-a')
            await queue.put(req)
            await asyncio.wait_for(verifier.await_entry(1), timeout=_SETTLE_TIMEOUT)
            await _land_foreign_commit(git_repo)
            head_gate.set()
            outcome = await wait_responsive(req.result, label='gt-a outcome')

        assert outcome.status == 'done', outcome
        verified, gated = [row['merge_sha'] for row in _rows(store, EventType.merge_verify, 'gt-a')]
        assert verified != gated == outcome.merge_sha, 'precondition: A lands rebased'
        # The gate re-verify is a dispatch of the rebased head (task 8525).
        assert _dispatch_facts(store, 'gt-a') == [
            ('prefer_local_free', 'main'), ('prefer_local_free', 'main'),
        ]


def _lease_activity(lane: MergeLane, task_id: str) -> str | None:
    rows = [r for r in lane.snapshot()['occupancy']['leases'] if r['task_id'] == task_id]
    return rows[0]['activity'] if rows else None


async def _cascade(
    git_ops: GitOps, config: OrchestratorConfig, prefix: str, *, b_finishes_first: bool,
) -> _LateArrivalFakeEventStore:
    """Head A fails on local while B, stacked on A, verifies on the remote 'laptop'.

    With *b_finishes_first* B's verify completes before A fails, so the
    cascade tears down a finished verify rather than cancelling a running one.
    """
    a_release, a_entered = asyncio.Event(), asyncio.Event()
    verifier = _GatedVerifier(
        entered={None: a_entered},
        first=dataclasses.replace(
            fails(category='test_failure', summary='test_failure'), release=a_release,
        ),
    )
    b_release, b_entered = asyncio.Event(), asyncio.Event()
    store = _LateArrivalFakeEventStore()
    laptop = _gated_runner(b_release, b_entered, name='laptop')

    async with _running_lane(
        git_ops, verifier=verifier, event_store=store, remote_runners=[laptop],
    ) as (lane, queue):
        req_a = await _submitted(git_ops, config, f'{prefix}-a')
        req_b = await _submitted(git_ops, config, f'{prefix}-b')
        await queue.put(req_a)
        await queue.put(req_b)
        await wait_responsive(a_entered.wait(), label=f'{prefix}-a to enter its verify')
        await wait_responsive(b_entered.wait(), label=f'{prefix}-b to enter its verify')
        if b_finishes_first:
            b_release.set()
            await _until(
                lambda: _lease_activity(lane, f'{prefix}-b') == 'verified_pending_landing',
                what=f'{prefix}-b verify to finish',
            )
        a_release.set()
        outcome_a = await wait_responsive(req_a.result, label=f'{prefix}-a outcome')
        b_release.set()
        outcome_b = await wait_responsive(req_b.result, label=f'{prefix}-b outcome')

    assert outcome_a.status not in ('done', 'already_merged'), outcome_a
    assert outcome_b.status == 'done', outcome_b
    return store


async def _halted_mid_verify(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, *, warm_up: str | None = None,
) -> list[dict]:
    """Operator-halt the lane while *task_id*'s verify runs; its cancelled rows.

    *warm_up*, when given, first lands one request on the same host.
    """
    release = asyncio.Event()
    verifier = FakeVerifier(scripts={
        task_id: VerifyScript(result=passes().result, release=release),
    })
    store = _LateArrivalFakeEventStore()

    async with _running_lane(git_ops, verifier=verifier, event_store=store) as (lane, queue):
        lane.VERIFY_ABANDON_POLL_SECS = 0.02
        if warm_up is not None:
            warm = await _submitted(git_ops, config, warm_up)
            await queue.put(warm)
            outcome = await wait_responsive(warm.result, label=f'{warm_up} outcome')
            assert outcome.status == 'done', outcome
        req = await _submitted(git_ops, config, task_id)
        await queue.put(req)
        await asyncio.wait_for(
            verifier.await_entry(verifier.entered_count + 1), timeout=_SETTLE_TIMEOUT,
        )
        lane.operator_halt('test: halt mid-verify')
        await _until(
            lambda: bool(_rows(store, EventType.merge_verify_cancelled, task_id)),
            what=f'{task_id} cancelled row',
        )
        release.set()
    return _rows(store, EventType.merge_verify_cancelled, task_id)


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestCancelledVerifyRows:

    async def test_a_cascade_records_one_cancel_of_the_downstream_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        store = await _cascade(git_ops, config, 'cx', b_finishes_first=False)

        rows = _rows(store, EventType.merge_verify_cancelled, 'cx-b')
        assert len(rows) == 1, f'exactly one: the orphan-guard reap must not count again; {rows}'
        assert rows[0]['cause'] == 'cascade'
        assert rows[0]['host'] == 'laptop'
        assert type(rows[0]['elapsed_ms']) is int
        assert rows[0]['elapsed_ms'] >= 0
        assert _rows(store, EventType.merge_verify_cancelled, 'cx-a') == []

    async def test_a_cascade_over_a_finished_verify_records_nothing(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        store = await _cascade(git_ops, config, 'cd', b_finishes_first=True)

        assert _rows(store, EventType.merge_verify_cancelled, 'cd-b') == []
        assert _rows(store, EventType.merge_verify_cancelled, 'cd-a') == []

    async def test_an_operator_halt_records_its_cancel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        rows = await _halted_mid_verify(git_ops, config, 'oh-a')

        assert len(rows) == 1, rows
        assert rows[0]['cause'] == 'operator_halt'
        assert rows[0]['host'] == 'local'
        assert rows[0]['fraction'] is None, 'no verify has completed on this host yet'

    async def test_fraction_is_measured_once_the_host_completed_a_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        rows = await _halted_mid_verify(git_ops, config, 'oh-v', warm_up='oh-w')

        assert len(rows) == 1, rows
        assert type(rows[0]['fraction']) is float
        assert rows[0]['fraction'] > 0
