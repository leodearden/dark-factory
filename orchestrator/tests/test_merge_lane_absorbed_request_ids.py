"""merge_finalized names every request that resolves to the primary's outcome (task 4830).

A loser that attaches to, or coalesces onto, an in-flight merge never gets a
merge_finalized row of its own.  The primary's row lists it under
absorbed_request_ids, so a fresh run's EventStore.merge_finalized_absorbing
still resolves the loser (plans/merge-status-durable-non-landed-prd.md D7).

Every scene reads its outcome back through a SECOND EventStore on the same db
with a different run_id: the simulated restart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.merge_lane import (
    InFlightMergeRegistry,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    WaiterRecord,
    coalesce_or_enqueue_merge_request,
    enqueue_merge_request,
)
from orchestrator.merge_lane.worker import register_and_enqueue_merge_request

pytestmark = pytest.mark.asyncio

BRANCH = '4830'


def _real_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
    )


def _make_req(tmp_path: Path) -> MergeRequest:
    config = _real_config(tmp_path)
    return MergeRequest(
        task_id=BRANCH,
        branch=QueuedBranch.parse(BRANCH, config.git.branch_prefix),
        worktree=tmp_path,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        snapshot_tip='abc123',
    )


def _waiter(request_id: str) -> WaiterRecord:
    return WaiterRecord(
        request_id=request_id,
        future=asyncio.get_running_loop().create_future(),
        source='workflow',
    )


async def _finalize(req: MergeRequest, outcome: MergeOutcome) -> None:
    req.result.set_result(outcome)
    await asyncio.sleep(0)


def _first_run(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', 'run-1')


def _after_restart(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', 'run-2')


def _absorbed_by(store: EventStore, request_id: str) -> list[str] | None:
    row = store.latest_merge_finalized(request_id=request_id, cross_run=True)
    assert row is not None, f'no merge_finalized row for {request_id}'
    return row['absorbed_request_ids']


async def test_attach_loser_resolves_to_the_primary_after_restart(tmp_path: Path) -> None:
    registry = InFlightMergeRegistry()
    primary = _make_req(tmp_path)
    await register_and_enqueue_merge_request(
        asyncio.Queue(), primary, _first_run(tmp_path), registry,
    )
    assert registry.attach(BRANCH, _waiter('mr-loser'))

    await _finalize(primary, MergeOutcome('done', merge_sha='abc'))

    restarted = _after_restart(tmp_path)
    row = restarted.merge_finalized_absorbing('mr-loser', cross_run=True)
    assert row is not None
    assert (row['request_id'], row['state'], row['run_id']) == (
        primary.request_id, 'done', 'run-1',
    )
    assert _absorbed_by(restarted, primary.request_id) == ['mr-loser']


async def test_door_coalesced_loser_resolves_to_the_primary_after_restart(
    tmp_path: Path,
) -> None:
    registry = InFlightMergeRegistry()
    store = _first_run(tmp_path)
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    primary = _make_req(tmp_path)
    loser = _make_req(tmp_path)
    dispatched = await coalesce_or_enqueue_merge_request(queue, primary, store, registry)
    coalesced = await coalesce_or_enqueue_merge_request(queue, loser, store, registry)
    assert dispatched.dispatched
    assert coalesced.in_flight

    await _finalize(primary, MergeOutcome('blocked', reason='x'))

    row = _after_restart(tmp_path).merge_finalized_absorbing(
        loser.request_id, cross_run=True,
    )
    assert row is not None
    assert (row['request_id'], row['state']) == (primary.request_id, 'blocked')


async def test_a_detached_waiter_is_not_listed(tmp_path: Path) -> None:
    """A detached waiter never receives the primary's outcome, so it is not absorbed."""
    registry = InFlightMergeRegistry()
    primary = _make_req(tmp_path)
    await register_and_enqueue_merge_request(
        asyncio.Queue(), primary, _first_run(tmp_path), registry,
    )
    registry.attach(BRANCH, _waiter('mr-gone'))
    registry.detach(BRANCH, 'mr-gone')

    await _finalize(primary, MergeOutcome('done', merge_sha='abc'))

    restarted = _after_restart(tmp_path)
    assert _absorbed_by(restarted, primary.request_id) == []
    assert restarted.merge_finalized_absorbing('mr-gone', cross_run=True) is None


async def test_only_the_slot_primary_lists_the_slot_waiters(tmp_path: Path) -> None:
    registry = InFlightMergeRegistry()
    store = _first_run(tmp_path)
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    owner = _make_req(tmp_path)
    non_owner = _make_req(tmp_path)
    assert await register_and_enqueue_merge_request(queue, owner, store, registry)
    assert not await register_and_enqueue_merge_request(queue, non_owner, store, registry)
    registry.attach(BRANCH, _waiter('mr-peer'))

    await _finalize(non_owner, MergeOutcome('done', merge_sha='abc'))

    assert _absorbed_by(_after_restart(tmp_path), non_owner.request_id) == []


async def test_an_enqueue_without_a_registry_writes_an_empty_list(tmp_path: Path) -> None:
    """Every row written from now on carries the key, so [] means nothing was absorbed."""
    req = _make_req(tmp_path)
    await enqueue_merge_request(asyncio.Queue(), req, _first_run(tmp_path))

    await _finalize(req, MergeOutcome('done', merge_sha='abc'))

    assert _absorbed_by(_after_restart(tmp_path), req.request_id) == []
