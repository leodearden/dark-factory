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
import subprocess
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
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


def _make_req(tmp_path: Path, *, snapshot_tip: str = 'abc123') -> MergeRequest:
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
        snapshot_tip=snapshot_tip,
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


def _linear_history(repo: Path, length: int) -> list[str]:
    """A fresh repository whose every commit descends from the one before."""
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git('init', '-b', 'main')
    git('config', 'user.email', 'test@test.com')
    git('config', 'user.name', 'Test')
    tips = []
    for n in range(length):
        (repo / f'{n}.txt').write_text(f'{n}\n')
        git('add', '-A')
        git('commit', '-m', f'commit {n}')
        tips.append(git('rev-parse', 'HEAD'))
    return tips


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


async def test_waiters_cancelled_with_a_replaced_primary_resolve_to_its_row(
    tmp_path: Path,
) -> None:
    """A C3 REPLACE cancels the queued primary's waiters along with it; its row lists them.

    Each row is read off the entry its primary owned at enqueue time, so a
    waiter attached after the slot moved on is listed on the replacing row only.
    """
    repo = tmp_path / 'repo'
    seed_tip, replaced_tip, replacing_tip = _linear_history(repo, 3)
    classifier = GitOps(_real_config(tmp_path).git, repo)
    registry = InFlightMergeRegistry()
    store = _first_run(tmp_path)
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    assert registry.acquire(
        BRANCH, BRANCH, asyncio.get_running_loop().create_future(),
        request_id='mr-seed', snapshot_tip=seed_tip,
    )
    replaced = _make_req(tmp_path, snapshot_tip=replaced_tip)
    replacing = _make_req(tmp_path, snapshot_tip=replacing_tip)

    async def replace_with(req: MergeRequest) -> None:
        result = await coalesce_or_enqueue_merge_request(
            queue, req, store, registry, classifier_git_ops=classifier,
        )
        assert result.dispatched, result

    await replace_with(replaced)
    cancelled_peer = _waiter('mr-peer')
    assert registry.attach(BRANCH, cancelled_peer)
    await replace_with(replacing)
    assert registry.attach(BRANCH, _waiter('mr-late'))

    await _finalize(replacing, MergeOutcome('done', merge_sha='abc'))

    assert cancelled_peer.future.cancelled()
    restarted = _after_restart(tmp_path)
    row = restarted.merge_finalized_absorbing('mr-peer', cross_run=True)
    assert row is not None
    assert (row['request_id'], row['state']) == (replaced.request_id, 'abandoned')
    assert _absorbed_by(restarted, replaced.request_id) == ['mr-peer']
    assert _absorbed_by(restarted, replacing.request_id) == ['mr-late']
